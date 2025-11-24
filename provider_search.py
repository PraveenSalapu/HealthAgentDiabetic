"""
Provider search strategies for appointment scheduling.

This module defines an abstraction for fetching nearby clinicians using
multiple backends:
- Mock directory (for offline/testing)
- Google Places API (if USE_GOOGLE_PLACES flag is set)
- Gemini Browser Use (if ENABLE_GEMINI_BROWSER is set)

Each strategy exposes a common interface so the app can select the best
available option at runtime.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

LOGGER = logging.getLogger(__name__)

USE_GOOGLE_PLACES = os.getenv("USE_GOOGLE_PLACES", "false").lower() in {"1", "true", "yes"}
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
ENABLE_GEMINI_BROWSER = os.getenv("ENABLE_GEMINI_BROWSER", "false").lower() in {"1", "true", "yes"}
GEMINI_BROWSER_MODEL = os.getenv("GEMINI_BROWSER_MODEL", "gemini-2.5-pro-latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CACHE_TTL_SECONDS = int(os.getenv("PROVIDER_CACHE_TTL", "1800"))  # 30 minutes


@dataclass
class ProviderRecord:
    name: str
    specialty: str
    city: str
    zip: str
    distance: Optional[float] = None
    rating: Optional[float] = None
    next_availability: Optional[str] = None
    accepting_new: Optional[bool] = None
    url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class ProviderSearchStrategy(ABC):
    @abstractmethod
    def search(self, location: str, specialty: str, date_preference: str) -> List[ProviderRecord]:
        raise NotImplementedError


def cache_results(func):
    store: Dict[Tuple[str, str], Tuple[float, List[ProviderRecord]]] = {}

    @functools.wraps(func)
    def wrapper(self, location: str, specialty: str, date_preference: str) -> List[ProviderRecord]:
        key = (location.lower(), specialty.lower())
        now = time.time()
        if key in store:
            timestamp, data = store[key]
            if now - timestamp <= CACHE_TTL_SECONDS:
                return data
        results = func(self, location, specialty, date_preference)
        store[key] = (now, results)
        return results

    return wrapper


DEFAULT_MOCK_PROVIDERS: List[ProviderRecord] = [
    ProviderRecord(
        name="Dr. Alicia Gomez",
        specialty="endocrinologist",
        city="Austin, TX",
        zip="78701",
        distance=2.3,
        rating=4.9,
        next_availability="In 2 days",
        accepting_new=True,
        url="https://www.exampleclinic.com/alicia-gomez",
    ),
    ProviderRecord(
        name="Dr. Marcus Lee",
        specialty="primary care",
        city="Austin, TX",
        zip="78704",
        distance=3.8,
        rating=4.7,
        next_availability="In 4 days",
        accepting_new=True,
        url="https://www.exampleclinic.com/marcus-lee",
    ),
    ProviderRecord(
        name="Dr. Priya Natarajan",
        specialty="endocrinologist",
        city="Round Rock, TX",
        zip="78664",
        distance=18.5,
        rating=4.8,
        next_availability="Next week",
        accepting_new=True,
        url="https://www.exampleclinic.com/priya-natarajan",
    ),
]


class MockProviderStrategy(ProviderSearchStrategy):
    def __init__(self, providers: Optional[List[ProviderRecord]] = None):
        self.providers = providers or DEFAULT_MOCK_PROVIDERS

    @cache_results
    def search(self, location: str, specialty: str, date_preference: str) -> List[ProviderRecord]:
        location_lower = location.lower()
        specialty_lower = specialty.lower()

        def matches(record: ProviderRecord) -> bool:
            city_match = location_lower in record.city.lower() or location_lower in record.zip
            specialty_match = specialty_lower in record.specialty.lower()
            return city_match and specialty_match

        results = [rec for rec in self.providers if matches(rec)]
        if not results:
            results = [
                rec for rec in self.providers
                if location_lower in rec.city.lower() or location_lower in rec.zip
            ]
        return sorted(results, key=lambda r: (r.distance or 99, -(r.rating or 0)))


class GooglePlacesStrategy(ProviderSearchStrategy):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _geocode(self, location: str) -> Optional[Tuple[float, float]]:
        params = {"address": location, "key": self.api_key}
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params=params,
            timeout=10,
        )
        if resp.status_code != 200:
            LOGGER.warning("Geocode failed: %s", resp.text)
            return None
        payload = resp.json()
        if not payload.get("results"):
            return None
        loc = payload["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]

    @cache_results
    def search(self, location: str, specialty: str, date_preference: str) -> List[ProviderRecord]:
        coords = self._geocode(location)
        if coords is None:
            return []
        lat, lng = coords
        params = {
            "key": self.api_key,
            "location": f"{lat},{lng}",
            "radius": 20000,
            "keyword": specialty,
            "type": "doctor",
        }
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params=params,
            timeout=10,
        )
        if resp.status_code != 200:
            LOGGER.warning("Places search failed: %s", resp.text)
            return []
        payload = resp.json()
        records: List[ProviderRecord] = []
        for result in payload.get("results", [])[:10]:
            records.append(
                ProviderRecord(
                    name=result.get("name", "Clinic"),
                    specialty=specialty,
                    city=result.get("vicinity", location),
                    zip=location,
                    rating=result.get("rating"),
                    next_availability="Call for availability",
                    url=f"https://www.google.com/maps/place/?q=place_id:{result.get('place_id')}",
                    raw=result,
                )
            )
        if payload.get("status") == "OVER_QUERY_LIMIT":
            LOGGER.warning("Google Places quota exceeded")
        return records


class BrowserUseStrategy(ProviderSearchStrategy):
    def __init__(self, api_key: str, model_name: str = GEMINI_BROWSER_MODEL):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        genai.configure(api_key=api_key)
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                tools=["google_search_retrieval", "code_execution", "browser"],
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to initialise Gemini browser model: {exc}")

    @cache_results
    def search(self, location: str, specialty: str, date_preference: str) -> List[ProviderRecord]:
        prompt = (
            "You are a care concierge tasked with finding nearby clinicians.\n"
            "Requirements:\n"
            f"- Specialty: {specialty}\n"
            f"- Location: {location}\n"
            f"- Preferred appointment timing: {date_preference}\n"
            "Search web sources for clinics or physicians, prioritising those accepting new patients, good reviews, and close distance."
            " Return a concise list with booking URLs."
        )
        try:
            response = self.model.generate_content(prompt)
        except Exception as exc:
            LOGGER.warning("Browser search failed: %s", exc, exc_info=True)
            return []

        text = getattr(response, "text", "")
        records: List[ProviderRecord] = []
        for line in text.splitlines():
            if "http" not in line:
                continue
            parts = line.split("http", 1)
            name = parts[0].strip(" -*:") or "Clinic"
            url = "http" + parts[1].split()[0]
            snippet = line.replace(url, "").strip(" -")
            records.append(
                ProviderRecord(
                    name=name,
                    specialty=specialty,
                    city=location,
                    zip=location,
                    url=url,
                    next_availability="Check site",
                    raw={"snippet": snippet},
                )
            )
        return records

    def book(self, record: ProviderRecord, patient_profile: Dict[str, Any], date_preference: str) -> str:
        prompt = (
            "You are an AI care concierge. Use browser tools to help secure an appointment.\n"
            f"Clinic URL: {record.url or record.raw.get('url', 'N/A')}\n"
            f"Patient profile: {json.dumps(patient_profile, indent=2)}\n"
            f"Preferred timing: {date_preference}\n"
            "Open the site, attempt to schedule, and summarise the outcome. "
            "If online booking isn't possible, provide contact instructions."
        )
        response = self.model.generate_content(prompt)
        return getattr(response, "text", "") or "Booking attempt complete. Please confirm directly with the clinic."


def build_provider_search_chain() -> List[ProviderSearchStrategy]:
    strategies: List[ProviderSearchStrategy] = [MockProviderStrategy()]

    if USE_GOOGLE_PLACES and GOOGLE_PLACES_API_KEY:
        strategies.insert(0, GooglePlacesStrategy(GOOGLE_PLACES_API_KEY))

    if ENABLE_GEMINI_BROWSER and GEMINI_API_KEY:
        try:
            strategies.insert(0, BrowserUseStrategy(GEMINI_API_KEY))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to initialise browser strategy: %s", exc)

    return strategies


def search_providers(location: str, specialty: str, date_preference: str) -> Tuple[List[ProviderRecord], List[str]]:
    strategies = build_provider_search_chain()
    messages: List[str] = []

    for strategy in strategies:
        try:
            records = strategy.search(location, specialty, date_preference)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Provider search error via %s: %s", strategy.__class__.__name__, exc)
            messages.append(f"{strategy.__class__.__name__} unavailable: {exc}")
            continue

        if records:
            disclaimer = []
            if isinstance(strategy, MockProviderStrategy):
                disclaimer.append("Mock directory data; confirm details before booking.")
            elif isinstance(strategy, GooglePlacesStrategy):
                disclaimer.append("Powered by Google Places; details may change, confirm on provider site.")
            else:
                disclaimer.append("Live search via Gemini Browser Use; verify booking availability.")
            return records, disclaimer

    return [], messages or ["No provider strategies returned results."]


_browser_strategy: Optional[BrowserUseStrategy] = None


def get_browser_strategy() -> Optional[BrowserUseStrategy]:
    global _browser_strategy
    if not (ENABLE_GEMINI_BROWSER and GEMINI_API_KEY and genai):
        return None
    if _browser_strategy is None:
        try:
            _browser_strategy = BrowserUseStrategy(GEMINI_API_KEY)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Unable to initialise browser booking: %s", exc)
            return None
    return _browser_strategy


def attempt_browser_booking(record: ProviderRecord,
                            patient_profile: Dict[str, Any],
                            date_preference: str) -> Optional[str]:
    strategy = get_browser_strategy()
    if strategy is None:
        return None
    try:
        return strategy.book(record, patient_profile, date_preference)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Browser booking failed: %s", exc)
        return None
