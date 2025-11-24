"""
StreamLit Diabetes Predictor + Gemini Health Chatbot
=====================================================
A comprehensive health prediction application with AI-powered assistance.

Requirements:
    pip install streamlit pandas numpy plotly google-generativeai joblib xgboost imbalanced-learn

Environment Variables:
    GEMINI_API_KEY: Your Google Gemini API key
    GEMINI_MODEL: Model name (default: gemini-pro)

Model Files Required:
    - model_output/ml_xgboost_smoteenn_pipeline.pkl (your trained model)
    - diabetic_averages.json (feature averages for diabetic population)

Important Notes:
    - sklearn version warnings are SAFE to ignore - your pipeline will work correctly
    - The model was trained with sklearn 1.5.1, but works with 1.7.2
    - Your ImbPipeline handles all preprocessing automatically
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
import plotly.graph_objects as go
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional, Any

from provider_search import (
    ProviderRecord,
    search_providers,
    attempt_browser_booking,
)
import joblib
import sklearn
from xgboost import XGBClassifier
import re

# Suppress sklearn version warnings for better UX
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_JSON_PATH = "model_output2/xgboost_model.json"
PREPROCESSOR_PATH = "model_output2/preprocessor.pkl"
THRESHOLD_PATH = "model_output2/optimal_threshold.json"
AVERAGES_PATH = "model_output2/diabetic_averages.json"

FEATURE_CONFIGS = {
    "GenHlth": {"type": "select", "options": [1, 2, 3, 4, 5], 
                "labels": ["Excellent", "Very Good", "Good", "Fair", "Poor"]},
    "HighBP": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "DiffWalk": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "BMI": {"type": "number", "min": 10.0, "max": 70.0, "default": 25.0, "step": 0.1},
    "HighChol": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "Age": {"type": "number", "min": 1, "max": 13, "default": 7, "step": 1, 
            "help": "Age category (1=18-24, 2=25-29, ..., 13=80+)"},
    "HeartDiseaseorAttack": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "PhysHlth": {"type": "number", "min": 0, "max": 30, "default": 0, "step": 1,
                 "help": "Days of poor physical health in past 30 days"},
    "Income": {"type": "select", 
               "options": [1, 2, 3, 4, 5, 6, 7, 8],
               "labels": ["<$10k", "$10k-15k", "$15k-20k", "$20k-25k", "$25k-35k", "$35k-50k", "$50k-75k", ">$75k"]},
    "Education": {"type": "select",
                  "options": [1, 2, 3, 4, 5, 6],
                  "labels": ["Never attended", "Grades 1-8", "Grades 9-11", "High school/GED", "Some college", "College grad"]},
    "PhysActivity": {"type": "select", "options": [0, 1], "labels": ["No", "Yes"]}
}

# Feature display names
FEATURE_NAMES = {
    "GenHlth": "General Health",
    "HighBP": "High Blood Pressure",
    "DiffWalk": "Difficulty Walking",
    "BMI": "Body Mass Index",
    "HighChol": "High Cholesterol",
    "Age": "Age Category",
    "HeartDiseaseorAttack": "Heart Disease or Attack History",
    "PhysHlth": "Physical Health (poor days/month)",
    "Income": "Income Level",
    "Education": "Education Level",
    "PhysActivity": "Physical Activity (last 30 days)"
}

FEATURE_INFO = {
    "GenHlth": "1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor.",
    "HighBP": "0=No history of high blood pressure, 1=Diagnosed with high blood pressure.",
    "DiffWalk": "0=No difficulty walking or climbing stairs, 1=Difficulty present.",
    "BMI": "Body Mass Index (weight/height^2). Typical healthy adult range is 18.5-24.9.",
    "HighChol": "0=No high cholesterol diagnosis, 1=Diagnosed with high cholesterol.",
    "Age": "Categories: 1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+.",
    "HeartDiseaseorAttack": "0=No prior heart disease/attack, 1=History of heart disease or heart attack.",
    "PhysHlth": "Number of poor physical health days in the past 30 days (0-30).",
    "Income": "1=<10K, 2=10-15K, 3=15-20K, 4=20-25K, 5=25-35K, 6=35-50K, 7=50-75K, 8=>75K.",
    "Education": "1=Never attended/Kindergarten only, 2=Grades 1-8, 3=Grades 9-11, 4=High school/GED, 5=Some college, 6=College graduate.",
    "PhysActivity": "0=No physical activity in past 30 days, 1=Performed physical activity or exercise."
}

FORM_SECTIONS = [
    {
        "title": "Wellness Snapshot",
        "description": "Capture how you feel day to day alongside core wellness indicators.",
        "features": ["GenHlth", "BMI", "PhysHlth", "PhysActivity"],
    },
    {
        "title": "Cardiometabolic History",
        "description": "Document clinical risk factors that influence diabetes risk.",
        "features": ["HighBP", "HighChol", "HeartDiseaseorAttack", "DiffWalk"],
    },
    {
        "title": "Lifestyle & Support",
        "description": "Share information that shapes access to care and daily habits.",
        "features": ["Age", "Education", "Income"],
    },
]

RADAR_FEATURES = ["GenHlth", "BMI", "PhysHlth", "PhysActivity", "HighBP", "HighChol"]

IDEAL_PROFILE = {
    "GenHlth": 1,
    "BMI": 22.0,
    "PhysHlth": 0,
    "PhysActivity": 1,
    "HighBP": 0,
    "HighChol": 0,
}

AGENT_DEFINITIONS = {
    "coach": {
        "name": "Health Coach",
        "avatar": "🧭",
        "keywords": ["motivation", "routine", "habits", "exercise", "stress", "sleep", "plan"],
        "system_prompt": (
            "You are a supportive personal health coach helping someone interpret their diabetes risk results.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Focus on empowering next steps, mindset shifts, and habit coaching. Keep explanations warm and practical."
        ),
        "fallback_focus": (
            "Focus on motivation, small habit changes, and balancing lifestyle pillars like sleep, stress, and movement."
        ),
    },
    "doctor": {
        "name": "Clinical Advisor",
        "avatar": "🩺",
        "keywords": ["symptom", "medication", "diagnosis", "doctor", "treatment", "blood test", "lab"],
        "system_prompt": (
            "You are a cautious clinician interpreting model findings without giving direct medical orders.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Highlight questions to raise with a physician, relevant screenings or labs, and safety precautions."
        ),
        "fallback_focus": (
            "Suggest evidence-based checkpoints to review with a doctor and emphasize follow-up care."
        ),
    },
    "dietician": {
        "name": "Dietician",
        "avatar": "🥗",
        "keywords": ["diet", "meal", "food", "nutrition", "carb", "protein", "recipe", "eat"],
        "system_prompt": (
            "You are a registered dietician tailoring nutrition advice to the person's risk profile.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Offer practical meal planning, portion guidance, and nutrient strategies aligned with diabetes prevention."
        ),
        "fallback_focus": (
            "Share carbohydrate awareness tips, balanced meals, hydration guidance, and mindful eating strategies."
        ),
    },
    "scheduler": {
        "name": "Care Concierge",
        "avatar": "📅",
        "keywords": ["appointment", "schedule", "book", "nearby", "consultant", "specialist", "clinic"],
        "system_prompt": (
            "You are a care coordinator helping the user plan follow-up appointments.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Recommend how to choose local clinicians, what to bring to visits, and how to manage scheduling."
        ),
        "fallback_focus": (
            "Guide the user to gather insurance details, shortlist providers, prepare records, and set reminders."
        ),
    },
}

SCHEDULER_REQUIRED_FIELDS = [
    ("location", "To find nearby specialists, what city or ZIP code works best for you?"),
    ("specialty", "What type of clinician would you like to see (e.g., primary care, endocrinologist)?"),
    ("date", "When would you like the appointment? Share a specific date or a window such as 'next week'."),
]

def format_feature_value(feature: str, value: float) -> str:
    """Human readable representation for a feature value."""
    config = FEATURE_CONFIGS.get(feature, {})
    if not config:
        return str(value)

    if config.get("type") == "select":
        options = config.get("options", [])
        labels = config.get("labels", [])
        try:
            idx = options.index(int(value))
            return labels[idx]
        except (ValueError, IndexError):
            return str(value)

    if feature == "BMI":
        return f"{value:.1f}"

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value}"


def build_profile_summary(user_data: Dict[str, float]) -> str:
    """Create a bullet summary of key user metrics."""
    lines = []
    for feature in FEATURE_CONFIGS:
        if feature not in user_data:
            continue
        value = user_data[feature]
        display = format_feature_value(feature, value)
        lines.append(f"- {FEATURE_NAMES.get(feature, feature)}: {display}")
    return "\n".join(lines)


def determine_agent(user_message: str) -> str:
    """Route the user message to the best-fit agent."""
    text = user_message.lower()
    for key, agent in AGENT_DEFINITIONS.items():
        for keyword in agent["keywords"]:
            if keyword in text:
                return key
    return "coach"


def build_app_context(probability: float, user_data: Dict[str, float]) -> Dict[str, str]:
    """Assemble common context fields for agents."""
    risk_level, _, _ = classify_risk(probability)
    return {
        "probability": probability,
        "risk_level": risk_level,
        "profile_summary": build_profile_summary(user_data),
    }


def initialize_scheduler_flow():
    """Ensure scheduler flow state exists."""
    if "scheduler_flow" not in st.session_state:
        st.session_state.scheduler_flow = {
            "data": {},
            "awaiting": None,
            "stage": "collecting",
            "suggestions": [],
            "disclaimers": [],
        }


def reset_scheduler_flow():
    """Reset scheduler-specific state."""
    st.session_state.scheduler_flow = {
        "data": {},
        "awaiting": None,
        "stage": "collecting",
        "suggestions": [],
        "disclaimers": [],
    }


def normalize_specialty(text: str) -> str:
    value = text.lower().strip()
    synonyms = {
        "pcp": "primary care",
        "primary doctor": "primary care",
        "family doctor": "primary care",
        "diabetes doctor": "endocrinologist",
        "nutrition": "nutritionist",
        "diet": "nutritionist",
        "heart": "cardiologist",
    }
    for key, mapped in synonyms.items():
        if key in value:
            return mapped
    return value


def enrich_scheduler_data_from_message(flow: Dict[str, Any], message: str):
    """Heuristically populate scheduler data from free-form text."""
    text = message.strip()
    if not text:
        return

    # Zip code extraction
    if "location" not in flow["data"]:
        zip_matches = re.findall(r"\b\d{5}\b", text)
        if zip_matches:
            flow["data"]["location"] = zip_matches[0]
        else:
            # simple city extraction after 'in'
            lower = text.lower()
            if " in " in lower:
                potential = text.lower().split(" in ", 1)[1]
                flow["data"]["location"] = potential.strip().strip(".?!")

    if "specialty" not in flow["data"]:
        flow["data"]["specialty"] = normalize_specialty(text)

    if "date" not in flow["data"]:
        if any(word in text.lower() for word in ["today", "tomorrow", "week", "month", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            flow["data"]["date"] = text


def format_doctor_suggestions(doctors: List[Any], disclaimers: List[str]) -> str:
    """Build markdown list of doctor suggestions."""
    if not doctors:
        return (
            "I couldn't find a close match in the sample directory. Try broadening the location or specialty, "
            "or let me know the city and provider type again."
        )

    lines = ["Here are top matches near you. Reply with the number of the doctor you'd like to book:"]
    for idx, doc in enumerate(doctors, start=1):
        if isinstance(doc, ProviderRecord):
            accepting = "Accepting new patients" if doc.accepting_new else "Currently waitlisted"
            distance = f"{doc.distance} miles away" if doc.distance is not None else "Distance not available"
            rating = f"rating {doc.rating}/5" if doc.rating is not None else "no rating info"
            availability = doc.next_availability or "Check availability"
            city = doc.city
            specialty = doc.specialty.title()
        else:
            accepting = "Accepting new patients" if doc.get("accepting_new") else "Currently waitlisted"
            distance = f"{doc.get('distance')} miles away"
            rating = f"rating {doc.get('rating')}/5" if doc.get("rating") is not None else "no rating info"
            availability = doc.get("next_availability", "Check availability")
            city = doc.get("city", "Unknown location")
            specialty = doc.get("specialty", "clinic").title()
        lines.append(
            f"{idx}. **{doctor_to_display_name(doc)}** ({specialty}) — {city}, "
            f"{distance}, {rating}. {accepting}. Next availability: {availability}."
        )
    if disclaimers:
        lines.append("\n" + "\n".join(f"_Note: {text}_" for text in disclaimers))
    lines.append("\nIf none of these work, describe different preferences (another location, specialty, or date).")
    return "\n".join(lines)


def process_scheduler_message(user_message: str,
                              probability: float,
                              user_data: Dict[str, float]) -> str:
    """Handle structured scheduling workflow and return assistant reply."""
    initialize_scheduler_flow()
    flow = st.session_state.scheduler_flow

    if flow.get("stage") == "selection":
        return handle_scheduler_selection(flow, user_message, probability, user_data)

    awaiting = flow.get("awaiting")
    if awaiting:
        flow["data"][awaiting] = user_message.strip()
        flow["awaiting"] = None
    else:
        enrich_scheduler_data_from_message(flow, user_message)

    for field, prompt in SCHEDULER_REQUIRED_FIELDS:
        if not flow["data"].get(field):
            flow["awaiting"] = field
            return prompt

    location = flow["data"]["location"]
    specialty = flow["data"]["specialty"]
    date_pref = flow["data"].get("date", "earliest available")
    doctors, disclaimers = search_providers(location, specialty, date_pref)
    flow["suggestions"] = doctors
    flow["disclaimers"] = disclaimers

    if not doctors:
        flow["stage"] = "collecting"
        flow["awaiting"] = None
        message = "I couldn't find any nearby specialists with the current settings."
        if disclaimers:
            message += "\n" + "\n".join(disclaimers)
        message += "\nTry another ZIP code, broader location, or different specialty."
        return message

    flow["stage"] = "selection"
    flow["awaiting"] = None
    return format_doctor_suggestions(doctors, disclaimers)


def parse_doctor_selection(message: str, count: int) -> Optional[int]:
    """Return zero-based index of selected doctor or None."""
    matches = re.findall(r"\b(\d+)\b", message)
    if matches:
        idx = int(matches[0]) - 1
        if 0 <= idx < count:
            return idx
    return None


def doctor_to_display_name(doctor: Any) -> str:
    if isinstance(doctor, ProviderRecord):
        return doctor.name
    if isinstance(doctor, dict):
        return doctor.get("name", "the clinic")
    return "the clinic"


def doctor_to_url(doctor: Any) -> Optional[str]:
    if isinstance(doctor, ProviderRecord):
        return doctor.url
    if isinstance(doctor, dict):
        return doctor.get("url")
    return None


def handle_scheduler_selection(flow: Dict[str, Any],
                               message: str,
                               probability: float,
                               user_data: Dict[str, float]) -> str:
    doctors = flow.get("suggestions", [])
    if not doctors:
        flow["stage"] = "collecting"
        return "I lost track of the suggested doctors. Could you share your location and specialty again?"

    selection = parse_doctor_selection(message, len(doctors))
    if selection is None:
        return "Please reply with the number of the doctor you'd like to book (for example, '1' or 'Book option 2')."

    chosen = doctors[selection]
    url = doctor_to_url(chosen)
    booking_summary = ""

    booking_attempt = None
    if isinstance(chosen, ProviderRecord):
        try:
            booking_attempt = attempt_browser_booking(
                chosen,
                {"profile": build_profile_summary(user_data), "probability": probability},
                flow["data"].get("date", "earliest available"),
            )
        except Exception as exc:
            st.info(f"Automated booking attempt failed: {exc}")

    if booking_attempt:
        booking_summary = booking_attempt
    else:
        booking_summary = (
            "I'm unable to automate the booking directly. Here's how to proceed:\n"
            f"1. Visit the clinic site: {url or 'contact the office directly'}.\n"
            "2. Provide your preferred appointment window and confirm insurance coverage.\n"
            "3. Ask for a confirmation email or text.\n"
            "4. Add the appointment to your calendar and gather lab results or medications ahead of time."
        )

    display_name = doctor_to_display_name(chosen)
    disclaimers = flow.get("disclaimers", [])
    flow["stage"] = "collecting"
    flow["suggestions"] = []
    flow["disclaimers"] = []
    flow["data"] = {}
    flow["awaiting"] = None
    extra = "\n\n" + "\n".join(disclaimers) if disclaimers else ""
    return f"**Booking Summary for {display_name}**\n\n{booking_summary}{extra}"

# Default averages (fallback if file not found or missing features)
# Based on BRFSS 2015 diabetic population averages
DEFAULT_DIABETIC_AVERAGES = {
    "GenHlth": 3.29,
    "HighBP": 0.75,
    "DiffWalk": 0.37,
    "BMI": 31.94,
    "HighChol": 0.67,
    "Age": 9.38,
    "HeartDiseaseorAttack": 0.22,
    "PhysHlth": 7.95,
    "Income": 5.21,
    "Education": 4.75,
    "PhysActivity": 0.63
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model_components(model_json_path: str, preprocessor_path: str, threshold_path: str):
    """
    Load XGBoost model (JSON), preprocessor (pkl), and optimal threshold.
    
    Returns:
        tuple: (xgb_model, preprocessor, threshold) or (None, None, None) on error
    """
    try:
        # 1. Load preprocessor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocessor = joblib.load(preprocessor_path)
        
        # 2. Load XGBoost model from JSON
        xgb_model = XGBClassifier()
        xgb_model.load_model(model_json_path)
        
        # 3. Load optimal threshold
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            threshold = threshold_data.get("threshold", 0.5)
        
        # 4. Test prediction with dummy data
        test_features = list(FEATURE_CONFIGS.keys())
        test_data = pd.DataFrame([{f: 0 if FEATURE_CONFIGS[f]["type"] == "select" 
                                   else FEATURE_CONFIGS[f]["default"] 
                                   for f in test_features}])
        
        # Preprocess and predict
        X_processed = preprocessor.transform(test_data)
        _ = xgb_model.predict_proba(X_processed)
        
        st.success("✅ Model components loaded successfully!")
        return xgb_model, preprocessor, threshold
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("💡 Ensure all model files exist in the correct location.")
        return None, None, None
    
    except Exception as e:
        st.error(f"❌ Error loading model components: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def load_diabetic_averages(averages_path: str) -> Dict[str, float]:
    """Load average feature values for diabetic population."""
    try:
        with open(averages_path, 'r') as f:
            data = json.load(f)
        
        # Handle both formats:
        # Format 1: Simple dict {"feature": value, ...}
        # Format 2: Array of objects [{"feature": "name", "average_value": value}, ...]
        
        if isinstance(data, list):
            # Array format - convert to dict
            averages = {}
            for item in data:
                feature_name = item.get("feature")
                avg_value = item.get("average_value")
                if feature_name and avg_value is not None:
                    averages[feature_name] = float(avg_value)
            
            # Log what we found
            found_features = set(averages.keys())
            required_features = set(FEATURE_CONFIGS.keys())
            missing = required_features - found_features
            
            if missing:
                st.warning(f"⚠️ Missing features in averages file: {missing}. Using defaults for missing values.")
                # Fill in missing with defaults
                for feature in missing:
                    averages[feature] = DEFAULT_DIABETIC_AVERAGES.get(feature, 0)
            
            return averages
        else:
            # Simple dict format
            return data
            
    except FileNotFoundError:
        st.warning(f"Averages file not found at {averages_path}. Using defaults.")
        return DEFAULT_DIABETIC_AVERAGES
    except Exception as e:
        st.warning(f"Error loading averages: {e}. Using defaults.")
        return DEFAULT_DIABETIC_AVERAGES

def sanitize_input(value, feature_name: str) -> float:
    """Sanitize and validate user input."""
    config = FEATURE_CONFIGS[feature_name]
    
    if config["type"] == "number":
        try:
            val = float(value)
            return max(config["min"], min(config["max"], val))
        except:
            return config["default"]
    else:  # select
        return int(value) if value in config["options"] else config["options"][0]

def initialize_gemini(api_key: str, model_name: str = "gemini-pro"):
    """Initialize Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini API: {e}")
        return None

def get_system_prompt(prediction_prob: float, user_data: Dict[str, float]) -> str:
    """Generate system prompt with prediction context."""
    return f"""You are a compassionate healthcare assistant helping users understand their diabetes risk assessment results. 

CONTEXT:
- The user has completed a diabetes risk assessment
- Model prediction probability: {prediction_prob:.1f}%
- User's health metrics: {json.dumps(user_data, indent=2)}

YOUR ROLE:
- Provide empathetic, non-alarmist guidance based on the risk assessment
- Suggest lifestyle modifications (diet, exercise, sleep, stress management)
- Recommend appropriate follow-up actions
- Answer health-related questions in accessible language
- NEVER provide definitive medical diagnoses or prescribe treatments

CRITICAL SAFETY RULES:
1. Always clarify this is a risk estimation tool, not a diagnosis
2. Use language like "the model estimates" or "based on these factors"
3. Include this disclaimer in your first message: "This assessment is for informational purposes only. Please consult a healthcare provider for proper diagnosis and treatment."
4. Encourage professional medical consultation for any health concerns
5. Avoid creating alarm - focus on actionable, positive steps
6. If asked about medications or treatments, defer to healthcare providers

CONVERSATION STYLE:
- Warm and supportive, but scientifically accurate
- Use simple language, avoid excessive medical jargon
- Provide specific, actionable suggestions
- Ask clarifying questions when helpful
- Acknowledge emotions and concerns
- Keep every response within 200 words unless the user asks for more detail

Begin by providing a gentle, contextual interpretation of their {prediction_prob:.1f}% risk probability and offer to answer any questions."""

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparison_chart(user_data: Dict[str, float], 
                           avg_data: Dict[str, float]) -> go.Figure:
    """Create interactive comparison chart."""
    features = list(user_data.keys())
    user_values = [user_data[f] for f in features]
    avg_values = [avg_data.get(f, 0) for f in features]
    
    # Calculate differences
    differences = [user_values[i] - avg_values[i] for i in range(len(features))]
    
    fig = go.Figure()
    
    # User values
    fig.add_trace(go.Bar(
        name='Your Values',
        x=features,
        y=user_values,
        marker=dict(
            color='#00d9ff',
            line=dict(color='#00b8d4', width=1.5),
            pattern=dict(shape="")
        ),
        text=[f'{v:.1f}' for v in user_values],
        textposition='outside',
        textfont=dict(color='#f1f5f9', size=13, family='"Inter","Segoe UI",sans-serif', weight=600)
    ))

    # Average diabetic values
    fig.add_trace(go.Bar(
        name='Avg. Diabetic Population',
        x=features,
        y=avg_values,
        marker=dict(
            color='#7c3aed',
            line=dict(color='#6d28d9', width=1.5)
        ),
        text=[f'{v:.1f}' for v in avg_values],
        textposition='outside',
        textfont=dict(color='#cbd5e1', size=13, family='"Inter","Segoe UI",sans-serif', weight=500)
    ))
    
    fig.update_layout(
        title=dict(
            text='Your Health Metrics vs. Average Diabetic Population',
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        xaxis_title='Health Factors',
        yaxis_title='Value',
        barmode='group',
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(26, 35, 66, 0.85)',
            bordercolor='rgba(100, 116, 255, 0.3)',
            borderwidth=1.5,
            font=dict(color='#cbd5e1', size=12)
        ),
        uniformtext=dict(mode="show", minsize=12),
        paper_bgcolor='rgba(10, 14, 26, 0.5)',
        plot_bgcolor='rgba(21, 29, 53, 0.5)',
        font=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_xaxes(
        tickangle=-35,
        showgrid=False,
        linecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=12, family='"Inter","Segoe UI",sans-serif')
    )
    fig.update_yaxes(
        gridcolor='rgba(100, 116, 255, 0.15)',
        zerolinecolor='rgba(100, 116, 255, 0.2)',
        tickfont=dict(color='#cbd5e1', size=12, family='"Inter","Segoe UI",sans-serif')
    )
    
    return fig, differences

def classify_risk(probability: float) -> Tuple[str, str, str]:
    """Map probability to risk tier, CSS badge class, and guidance message."""
    if probability < 30:
        return (
            "Low",
            "risk-badge-low",
            "Your current inputs align with a lower likelihood of diabetes. Keep reinforcing healthy habits.",
        )
    if probability < 60:
        return (
            "Moderate",
            "risk-badge-medium",
            "Your profile shows a mix of protective and elevated factors. Focused lifestyle adjustments can help.",
        )
    return (
        "Elevated",
        "risk-badge-high",
        "Your risk signals are elevated. Schedule time with a healthcare professional to plan next steps.",
    )


def create_risk_gauge(probability: float) -> go.Figure:
    """Create risk level gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Diabetes Risk Probability",
            'font': {'size': 22, 'color': '#f1f5f9', 'family': '"Inter","Segoe UI",sans-serif', 'weight': 700}
        },
        number={
            'suffix': "%",
            'font': {'size': 52, 'color': '#00d9ff', 'family': '"Inter","Segoe UI",sans-serif', 'weight': 800}
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1.5,
                'tickcolor': "rgba(100, 116, 255, 0.3)",
                'tickfont': {'color': '#cbd5e1', 'family': '"Inter","Segoe UI",sans-serif', 'size': 11}
            },
            'bar': {
                'color': "#00d9ff",
                'thickness': 0.8
            },
            'bgcolor': "rgba(21, 29, 53, 0.5)",
            'borderwidth': 2,
            'bordercolor': "rgba(100, 116, 255, 0.3)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(248, 113, 113, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#7c3aed", 'width': 5},
                'thickness': 0.8,
                'value': probability
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor='rgba(10, 14, 26, 0)',
        font=dict(family='"Inter","Segoe UI",sans-serif')
    )
    return fig


def prepare_feature_contributions(user_data: Dict[str, float],
                                  avg_data: Dict[str, float]) -> List[Tuple[str, float]]:
    """Compute normalized deltas for each feature to feed the waterfall chart."""
    contributions: List[Tuple[str, float]] = []

    for feature, config in FEATURE_CONFIGS.items():
        user_value = float(user_data.get(feature, avg_data.get(feature, 0)))
        avg_value = float(avg_data.get(feature, DEFAULT_DIABETIC_AVERAGES.get(feature, user_value)))

        if config["type"] == "number":
            span = float(config["max"] - config["min"])
            if span == 0:
                continue
            delta = (user_value - avg_value) / span * 100
        else:
            delta = (user_value - avg_value) * 100

        contributions.append((feature, delta))

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    return contributions


def create_contribution_waterfall(user_data: Dict[str, float],
                                  avg_data: Dict[str, float]) -> go.Figure:
    """Build a waterfall chart showing how each factor shifts risk relative to diabetic average."""
    contributions = prepare_feature_contributions(user_data, avg_data)[:8]
    labels = [FEATURE_NAMES[f] for f, _ in contributions]
    deltas = [round(delta, 2) for _, delta in contributions]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(deltas),
            y=labels,
            x=deltas,
            connector={"mode": "spanning", "line": {"color": "rgba(100, 116, 255, 0.3)", "width": 2}},
            decreasing={"marker": {"color": "#10b981", "line": {"color": "#059669", "width": 1}}},
            increasing={"marker": {"color": "#f87171", "line": {"color": "#ef4444", "width": 1}}},
            textposition="outside",
            textfont=dict(color='#cbd5e1', family='"Inter","Segoe UI",sans-serif', size=11)
        )
    )

    fig.update_layout(
        title=dict(
            text="Feature shifts versus diabetic average (normalized)",
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        showlegend=False,
        height=420,
        margin=dict(l=120, r=30, t=60, b=40),
        paper_bgcolor="rgba(10, 14, 26, 0.5)",
        plot_bgcolor="rgba(21, 29, 53, 0.5)",
        font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
        xaxis=dict(
            title="Relative shift (percentage points)",
            gridcolor="rgba(100, 116, 255, 0.15)",
            zerolinecolor="rgba(100, 116, 255, 0.3)",
            tickfont=dict(color="#cbd5e1"),
            title_font=dict(color="#94a3b8")
        ),
        yaxis=dict(
            tickfont=dict(color="#cbd5e1"),
            gridcolor="rgba(100, 116, 255, 0.1)"
        ),
    )

    return fig


def score_feature_for_radar(value: float, feature: str) -> float:
    """Translate raw feature values into a 0-1 wellness score (1 = favorable)."""
    config = FEATURE_CONFIGS[feature]

    if feature == "GenHlth":
        min_val, max_val = 1, 5
        return max(0.0, min(1.0, 1 - (value - min_val) / (max_val - min_val)))

    if feature == "BMI":
        # Ideal BMI approximated at 22 within allowable range.
        ideal = 22.0
        spread = max(config["max"] - config["min"], 1)
        return max(0.0, min(1.0, 1 - abs(value - ideal) / (spread / 2)))

    if feature == "PhysHlth":
        return max(0.0, min(1.0, 1 - value / max(config["max"], 1)))

    if feature == "PhysActivity":
        return 1.0 if value >= 1 else 0.0

    if feature in {"HighBP", "HighChol"}:
        return 1.0 - min(1.0, max(0.0, value))

    return 0.5


def create_wellness_radar(user_data: Dict[str, float],
                          avg_data: Dict[str, float]) -> go.Figure:
    """Create a radar chart comparing the user to archetypes."""
    categories = [FEATURE_NAMES[f] for f in RADAR_FEATURES]

    user_scores = [
        score_feature_for_radar(float(user_data.get(f, avg_data.get(f, IDEAL_PROFILE.get(f, 0)))), f)
        for f in RADAR_FEATURES
    ]

    ideal_scores = [
        score_feature_for_radar(float(IDEAL_PROFILE.get(f, avg_data.get(f, 0))), f)
        for f in RADAR_FEATURES
    ]

    diabetic_scores = [
        score_feature_for_radar(float(avg_data.get(f, IDEAL_PROFILE.get(f, 0))), f)
        for f in RADAR_FEATURES
    ]

    # Close the loop for polar plot
    categories_closed = categories + [categories[0]]
    traces = {
        "Your profile": user_scores + [user_scores[0]],
        "Ideal baseline": ideal_scores + [ideal_scores[0]],
        "Diabetic average": diabetic_scores + [diabetic_scores[0]],
    }

    fig = go.Figure()
    palette = {
        "Your profile": "#00d9ff",
        "Ideal baseline": "#10b981",
        "Diabetic average": "#f97316",
    }

    for label, values in traces.items():
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill="toself",
                name=label,
                line=dict(color=palette[label], width=2),
                opacity=0.5 if label != "Your profile" else 0.7,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                showticklabels=True,
                ticks="",
                tickfont=dict(size=10, color="#cbd5e1"),
                gridcolor="rgba(100, 116, 255, 0.2)",
                linecolor="rgba(100, 116, 255, 0.3)",
            ),
            bgcolor="rgba(21, 29, 53, 0.5)",
            angularaxis=dict(
                gridcolor="rgba(100, 116, 255, 0.2)",
                linecolor="rgba(100, 116, 255, 0.3)",
                tickfont=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif', size=11)
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
            bgcolor="rgba(26, 35, 66, 0.7)",
            bordercolor="rgba(100, 116, 255, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="rgba(10, 14, 26, 0.5)",
        title=dict(
            text="Wellness balance across key factors",
            font=dict(size=18, color='#f1f5f9', family='"Inter","Segoe UI",sans-serif', weight=700)
        ),
        font=dict(color="#cbd5e1", family='"Inter","Segoe UI",sans-serif'),
        height=450,
    )

    return fig

def generate_insights(user_data: Dict[str, float], 
                      avg_data: Dict[str, float], 
                      differences: List[float]) -> List[str]:
    """Generate textual insights from comparison."""
    insights = []
    features = list(user_data.keys())
    
    for i, feature in enumerate(features):
        diff = differences[i]
        user_val = user_data[feature]
        avg_val = avg_data.get(feature, 0)
        
        if feature in ["Age", "BMI", "GenHlth"]:
            if abs(diff) > 0.1 * avg_val:  # 10% threshold
                direction = "higher" if diff > 0 else "lower"
                insights.append(
                    f"**{FEATURE_NAMES[feature]}**: Your value ({user_val:.1f}) is "
                    f"{direction} than the average diabetic population ({avg_val:.1f})"
                )
        else:  # Binary features
            if user_val == 1 and avg_val > 0.5:
                insights.append(
                    f"**{FEATURE_NAMES[feature]}**: You share this risk factor with "
                    f"{avg_val*100:.0f}% of the diabetic population"
                )
            elif user_val == 0 and avg_val > 0.5:
                insights.append(
                    f"**{FEATURE_NAMES[feature]}**: You don't have this risk factor, "
                    f"which is positive (present in {avg_val*100:.0f}% of diabetic population)"
                )
    
    return insights[:5]  # Top 5 insights

# ============================================================================
# CHATBOT FUNCTIONS
# ============================================================================

def send_message_to_gemini(model,
                           agent_key: str,
                           conversation: List[Dict[str, str]],
                           user_message: str,
                           context: Dict[str, str]) -> str:
    """Send message to Gemini and get response."""
    try:
        agent = AGENT_DEFINITIONS[agent_key]
        system_prompt = agent["system_prompt"].format(**context)

        # Build conversation history
        chat = model.start_chat(history=[])

        chat.send_message(system_prompt)

        # Send previous messages for this agent
        for msg in conversation:
            chat.send_message(msg['content'])

        # Send current message
        response = chat.send_message(user_message)
        return response.text
    
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."


def generate_fallback_response(probability: float,
                               user_data: Dict[str, float],
                               agent_key: str,
                               user_message: Optional[str] = None) -> str:
    """Provide an on-device assistant response when Gemini is unavailable."""
    agent = AGENT_DEFINITIONS[agent_key]
    context = build_app_context(probability, user_data)
    risk_level = context["risk_level"]
    summary = (
        f"{agent['name']} (fallback)\n"
        f"- Estimated diabetes probability: {probability:.1f}% ({risk_level.lower()} risk)\n"
        f"- Key health snapshot:\n{context['profile_summary']}"
    )

    bmi = user_data.get("BMI")
    phys_activity = user_data.get("PhysActivity")
    high_bp = user_data.get("HighBP")
    high_chol = user_data.get("HighChol")

    guidance: List[str] = []

    if agent_key == "coach":
        if phys_activity == 0:
            guidance.append("Start with 10-minute movement blocks after meals to wake up insulin sensitivity.")
        if bmi and bmi >= 30:
            guidance.append("Set a modest weight trend goal (e.g., 3-5% over 3 months) paired with resistance work.")
        guidance.append("Use a habit tracker this week: log sleep hours, stress triggers, and screen-time after 9pm.")

    elif agent_key == "doctor":
        guidance.append("Schedule an A1C or oral glucose tolerance test if it has been more than 12 months.")
        if high_bp == 1:
            guidance.append("Bring a log of home blood pressure readings to your next visit.")
        guidance.append("Ask whether cholesterol, kidney function (eGFR), and retinal screening are due.")
        guidance.append("Prepare a list of medications and supplements before the appointment.")

    elif agent_key == "dietician":
        if bmi and bmi >= 28:
            guidance.append("Build plates with half non-starchy vegetables, quarter lean protein, quarter high-fiber carbs.")
        guidance.append("Aim for ~25-30g fiber daily from beans, berries, chia, or whole grains.")
        if high_chol == 1:
            guidance.append("Swap saturated fats for olive oil, avocado, and nuts to support lipid levels.")
        guidance.append("Batch-prep breakfast options (e.g., chia pudding, veggie omelets) to prevent morning spikes.")

    elif agent_key == "scheduler":
        guidance.append("Verify insurance network and telehealth options, then shortlist endocrinologists or PCPs.")
        guidance.append("Compile labs, device readings, and symptom notes to share ahead of the visit.")
        guidance.append("Ask clinics about wait times and cancellation policies; set calendar reminders 24 hours prior.")
        guidance.append("If you need a dietician referral, request it during the primary appointment.")

    if not guidance:
        guidance.append(agent["fallback_focus"])

    action_plan = "Key suggestions:\n- " + "\n- ".join(guidance[:4])

    if user_message:
        return (
            f"You asked: \"{user_message}\".\n\n"
            f"{summary}\n\n"
            f"{action_plan}\n\n"
            "Always confirm these ideas with a healthcare professional who knows your full history."
        )

    return (
        f"{summary}\n\n"
        f"{action_plan}\n\n"
        "Let me know if you want to dive deeper into any of these steps."
    )

# ============================================================================
# STREAMLIT APP
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'prediction_prob' not in st.session_state:
        st.session_state.prediction_prob = 0.0
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = ""
    if 'optimal_threshold' not in st.session_state:
        st.session_state.optimal_threshold = 0.5
    if 'agent_histories' not in st.session_state:
        st.session_state.agent_histories = {key: [] for key in AGENT_DEFINITIONS}
    if 'active_agent' not in st.session_state:
        st.session_state.active_agent = "coach"
    initialize_scheduler_flow()

def reset_app():
    """Reset application state."""
    st.session_state.prediction_made = False
    st.session_state.user_data = {}
    st.session_state.prediction_prob = 0.0
    st.session_state.chat_history = []
    st.session_state.chatbot_initialized = False
    st.session_state.system_prompt = ""
    default_threshold = st.session_state.get("optimal_threshold_default", 0.5)
    st.session_state.optimal_threshold = default_threshold
    st.session_state.agent_histories = {key: [] for key in AGENT_DEFINITIONS}
    st.session_state.active_agent = "coach"
    reset_scheduler_flow()


def main():
    st.set_page_config(
        page_title="Diabetes Risk Navigator + AI Health Assistant",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {
            color-scheme: dark;
            --bg-primary: #0a0e1a;
            --bg-secondary: #0f1629;
            --bg-tertiary: #151d35;
            --bg-card: #1a2342;
            --accent-primary: #00d9ff;
            --accent-secondary: #7c3aed;
            --accent-tertiary: #f97316;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: rgba(100, 116, 255, 0.15);
            --glow-color: rgba(0, 217, 255, 0.3);
        }

        /* Main background with gradient */
        body, .stApp {
            background: linear-gradient(135deg, #0a0e1a 0%, #151d35 50%, #0f1320 100%) !important;
            color: var(--text-primary) !important;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        }

        div[data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0a0e1a 0%, #151d35 50%, #0f1320 100%) !important;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1629 0%, #1a2342 100%) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        /* Typography enhancements */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        }

        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown strong, .stMarkdown em {
            color: var(--text-secondary) !important;
        }

        /* Enhanced Primary Button */
        button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 16px !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            box-shadow:
                0 4px 16px rgba(0, 217, 255, 0.25),
                0 8px 32px rgba(124, 58, 237, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
        }

        button[kind="primary"]::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
            transition: left 0.5s !important;
        }

        button[kind="primary"]:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow:
                0 6px 24px rgba(0, 217, 255, 0.4),
                0 12px 48px rgba(124, 58, 237, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        }

        button[kind="primary"]:hover::before {
            left: 100% !important;
        }

        button[kind="primary"]:active {
            transform: translateY(0) scale(0.98) !important;
        }

        /* Enhanced Secondary Button */
        button[kind="secondary"] {
            background: rgba(0, 217, 255, 0.05) !important;
            color: var(--accent-primary) !important;
            border: 1.5px solid rgba(0, 217, 255, 0.3) !important;
            border-radius: 16px !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            backdrop-filter: blur(10px) !important;
        }

        button[kind="secondary"]:hover {
            background: rgba(0, 217, 255, 0.12) !important;
            border-color: var(--accent-primary) !important;
            box-shadow: 0 4px 16px rgba(0, 217, 255, 0.2) !important;
            transform: translateY(-1px) !important;
        }

        button[kind="secondary"]:active {
            transform: translateY(0) !important;
        }
        /* Glassmorphic Form */
        form[data-testid="stForm"] {
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.8), rgba(21, 29, 53, 0.9)) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1.5px solid rgba(100, 116, 255, 0.2) !important;
            border-radius: 28px !important;
            padding: 2.5rem !important;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.3),
                0 2px 8px rgba(124, 58, 237, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            position: relative !important;
            overflow: hidden !important;
        }

        form[data-testid="stForm"]::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -50% !important;
            width: 200% !important;
            height: 100% !important;
            background: radial-gradient(circle at 50% 50%, rgba(0, 217, 255, 0.03), transparent 70%) !important;
            animation: formGlow 8s ease-in-out infinite !important;
        }

        @keyframes formGlow {
            0%, 100% { transform: translateX(0) scale(1); opacity: 0.5; }
            50% { transform: translateX(25%) scale(1.1); opacity: 0.8; }
        }

        form[data-testid="stForm"] label {
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.01em !important;
            margin-bottom: 0.5rem !important;
            display: block !important;
        }

        .form-section-header h3 {
            margin-bottom: 0.5rem !important;
            color: var(--text-primary) !important;
            font-size: 1.35rem !important;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        .form-section-header p {
            margin-top: 0 !important;
            color: var(--text-muted) !important;
            font-size: 0.9rem !important;
            line-height: 1.6 !important;
        }

        .form-section-divider {
            border: 0 !important;
            height: 2px !important;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(0, 217, 255, 0.3) 20%,
                rgba(124, 58, 237, 0.3) 80%,
                transparent
            ) !important;
            margin: 2rem 0 !important;
            border-radius: 2px !important;
        }

        /* Enhanced Input Fields */
        .stApp input, .stApp textarea, div[data-baseweb="input"] input {
            background: rgba(10, 14, 26, 0.6) !important;
            color: var(--text-primary) !important;
            border: 1.5px solid rgba(100, 116, 255, 0.25) !important;
            border-radius: 14px !important;
            padding: 0.75rem 1rem !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow:
                0 2px 8px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
        }

        .stApp input:focus, .stApp textarea:focus, div[data-baseweb="input"] input:focus {
            background: rgba(10, 14, 26, 0.8) !important;
            border-color: var(--accent-primary) !important;
            box-shadow:
                0 0 0 3px rgba(0, 217, 255, 0.15),
                0 4px 16px rgba(0, 217, 255, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            outline: none !important;
        }

        /* Enhanced Select Dropdown */
        div[data-baseweb="select"] {
            background: transparent !important;
        }

        div[data-baseweb="select"] > div {
            background: rgba(10, 14, 26, 0.6) !important;
            color: var(--text-primary) !important;
            border: 1.5px solid rgba(100, 116, 255, 0.25) !important;
            border-radius: 14px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow:
                0 2px 8px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
            padding-right: 2.5rem !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="select"] > div:hover {
            border-color: rgba(0, 217, 255, 0.4) !important;
            box-shadow:
                0 4px 12px rgba(0, 217, 255, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            background: rgba(10, 14, 26, 0.8) !important;
        }

        div[data-baseweb="select"] svg {
            color: var(--accent-primary) !important;
            opacity: 0.8 !important;
        }

        /* Dropdown menu */
        div[data-baseweb="popover"] {
            background: rgba(15, 23, 42, 0.98) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1.5px solid rgba(100, 116, 255, 0.3) !important;
            border-radius: 14px !important;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.5),
                0 2px 8px rgba(0, 217, 255, 0.1) !important;
            overflow: hidden !important;
        }

        div[role="listbox"] {
            background: transparent !important;
            padding: 0.5rem !important;
        }

        div[role="option"] {
            color: var(--text-secondary) !important;
            transition: all 0.2s ease !important;
            padding: 0.75rem 1rem !important;
            border-radius: 10px !important;
            margin-bottom: 0.25rem !important;
            font-weight: 500 !important;
        }

        div[role="option"]:hover {
            background: rgba(0, 217, 255, 0.15) !important;
            color: var(--text-primary) !important;
            transform: translateX(4px) !important;
        }

        div[role="option"][aria-selected="true"] {
            background: rgba(0, 217, 255, 0.2) !important;
            color: var(--accent-primary) !important;
            font-weight: 600 !important;
        }

        /* Number Input Styling */
        div[data-testid="stNumberInput"] input {
            background: rgba(10, 14, 26, 0.6) !important;
            color: var(--text-primary) !important;
        }

        div[data-testid="stNumberInput"] button {
            background: rgba(0, 217, 255, 0.1) !important;
            color: var(--accent-primary) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(0, 217, 255, 0.2) !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: rgba(0, 217, 255, 0.2) !important;
            border-color: var(--accent-primary) !important;
            transform: scale(1.05) !important;
        }

        /* Selectbox text color */
        div[data-baseweb="select"] div[role="button"] {
            color: var(--text-primary) !important;
        }

        div[data-baseweb="select"] span {
            color: var(--text-primary) !important;
        }

        /* Help tooltip icon */
        div[data-testid="stTooltipIcon"] svg {
            color: var(--accent-primary) !important;
            opacity: 0.7 !important;
        }

        div[data-testid="stTooltipIcon"]:hover svg {
            opacity: 1 !important;
        }

        /* Input placeholder text */
        .stApp input::placeholder, .stApp textarea::placeholder {
            color: var(--text-muted) !important;
            opacity: 0.6 !important;
        }

        /* Labels and captions */
        .stApp label {
            color: var(--text-primary) !important;
        }

        .stApp .stMarkdown p, .stApp .stCaption {
            color: var(--text-secondary) !important;
        }

        div[data-testid="stCaptionContainer"] {
            color: var(--text-muted) !important;
        }

        /* Fix expander background */
        div[data-testid="stExpander"] {
            background: transparent !important;
            border: 1.5px solid rgba(100, 116, 255, 0.2) !important;
            border-radius: 14px !important;
        }

        div[data-testid="stExpander"] div[role="button"] {
            background: transparent !important;
            color: var(--text-primary) !important;
        }

        div[data-testid="stExpanderDetails"] {
            background: rgba(21, 29, 53, 0.3) !important;
            border-top: 1px solid rgba(100, 116, 255, 0.15) !important;
        }

        /* Alert boxes */
        div[data-testid="stAlert"] {
            background: transparent !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 14px !important;
        }

        div[data-testid="stSuccess"] {
            background: rgba(16, 185, 129, 0.1) !important;
            border: 1.5px solid rgba(16, 185, 129, 0.3) !important;
            color: #6ee7b7 !important;
        }

        div[data-testid="stWarning"] {
            background: rgba(251, 191, 36, 0.1) !important;
            border: 1.5px solid rgba(251, 191, 36, 0.3) !important;
            color: #fcd34d !important;
        }

        div[data-testid="stError"] {
            background: rgba(248, 113, 113, 0.1) !important;
            border: 1.5px solid rgba(248, 113, 113, 0.3) !important;
            color: #fca5a5 !important;
        }

        div[data-testid="stInfo"] {
            background: rgba(0, 217, 255, 0.1) !important;
            border: 1.5px solid rgba(0, 217, 255, 0.3) !important;
            color: var(--accent-primary) !important;
        }

        /* Dataframe styling */
        div[data-testid="stDataFrame"] {
            background: transparent !important;
        }

        div[data-testid="stDataFrame"] table {
            background: rgba(21, 29, 53, 0.5) !important;
            color: var(--text-secondary) !important;
        }

        div[data-testid="stDataFrame"] th {
            background: rgba(0, 217, 255, 0.15) !important;
            color: var(--text-primary) !important;
            border-color: rgba(100, 116, 255, 0.2) !important;
        }

        div[data-testid="stDataFrame"] td {
            border-color: rgba(100, 116, 255, 0.15) !important;
            color: var(--text-secondary) !important;
        }

        /* Hide any duplicate select elements or values */
        div[data-baseweb="select"] li[role="option"] span:last-child:empty {
            display: none !important;
        }

        /* Ensure clean single dropdown rendering */
        div[data-baseweb="select"] ul[role="listbox"] {
            list-style: none !important;
            padding: 0.5rem !important;
        }

        /* Clean dropdown button display - show only the formatted label */
        div[data-baseweb="select"] div[role="button"] > div {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }

        /* Ensure dropdowns show clean text without duplicates */
        .stSelectbox > div > div > div {
            background: rgba(10, 14, 26, 0.6) !important;
        }

        /* Caption styling improvements */
        .stApp [data-testid="stCaptionContainer"] {
            margin-top: 0.35rem !important;
            font-size: 0.85rem !important;
            line-height: 1.5 !important;
            color: var(--text-muted) !important;
        }
        /* Enhanced Hero Section with Glassmorphism */
        .app-hero {
            display: grid;
            grid-template-columns: minmax(0, 2.4fr) minmax(0, 1fr);
            gap: 2.5rem;
            padding: 3rem;
            border-radius: 32px;
            border: 1.5px solid rgba(100, 116, 255, 0.2);
            background: linear-gradient(
                135deg,
                rgba(26, 35, 66, 0.7),
                rgba(21, 29, 53, 0.85)
            );
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.4),
                0 8px 16px rgba(124, 58, 237, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .app-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(
                circle at center,
                rgba(0, 217, 255, 0.08) 0%,
                rgba(124, 58, 237, 0.05) 50%,
                transparent 70%
            );
            animation: heroGlow 10s ease-in-out infinite;
        }

        @keyframes heroGlow {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            50% { transform: translate(-10%, 10%) rotate(180deg); }
        }

        .hero-left {
            position: relative;
            z-index: 1;
        }

        .hero-left h1 {
            margin-bottom: 0.75rem;
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff, var(--accent-primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }

        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1.2rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.15), rgba(124, 58, 237, 0.15));
            border: 1px solid rgba(0, 217, 255, 0.3);
            color: var(--accent-primary);
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.15);
            animation: pillPulse 3s ease-in-out infinite;
        }

        @keyframes pillPulse {
            0%, 100% { box-shadow: 0 4px 12px rgba(0, 217, 255, 0.15); }
            50% { box-shadow: 0 6px 20px rgba(0, 217, 255, 0.3); }
        }

        .hero-subtitle {
            font-size: 1.08rem;
            color: var(--text-secondary);
            margin-top: 1rem;
            margin-bottom: 1.75rem;
            line-height: 1.7;
            font-weight: 400;
        }

        .hero-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem;
        }

        .hero-step {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.12), rgba(124, 58, 237, 0.08));
            border: 1px solid rgba(0, 217, 255, 0.25);
            color: var(--accent-primary);
            font-weight: 600;
            border-radius: 14px;
            padding: 0.7rem 1.2rem;
            font-size: 0.92rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
        }

        .hero-step:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 217, 255, 0.2);
            border-color: var(--accent-primary);
        }

        .hero-highlight {
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.9), rgba(21, 29, 53, 0.95));
            border-radius: 28px;
            border: 1.5px solid rgba(0, 217, 255, 0.3);
            padding: 2.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 0 0 1px rgba(0, 217, 255, 0.1);
            position: relative;
            z-index: 1;
            backdrop-filter: blur(10px);
        }

        .hero-metric-label {
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.15em;
            color: var(--text-muted);
            font-weight: 600;
        }

        .hero-metric-value {
            font-size: 3.5rem;
            font-weight: 800;
            color: var(--text-primary);
            line-height: 1;
            text-shadow: 0 2px 12px rgba(0, 217, 255, 0.3);
        }

        .hero-metric-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.5rem 1.2rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.02em;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        .hero-highlight-note {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        /* Enhanced Risk Badges */
        .risk-badge-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(5, 150, 105, 0.2));
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #6ee7b7;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        }

        .risk-badge-medium {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.25), rgba(245, 158, 11, 0.2));
            border: 1px solid rgba(251, 191, 36, 0.4);
            color: #fcd34d;
            box-shadow: 0 4px 12px rgba(251, 191, 36, 0.2);
        }

        .risk-badge-high {
            background: linear-gradient(135deg, rgba(248, 113, 113, 0.25), rgba(239, 68, 68, 0.2));
            border: 1px solid rgba(248, 113, 113, 0.4);
            color: #fca5a5;
            box-shadow: 0 4px 12px rgba(248, 113, 113, 0.2);
        }

        .risk-badge-neutral {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.25), rgba(124, 58, 237, 0.2));
            border: 1px solid rgba(0, 217, 255, 0.4);
            color: var(--accent-primary);
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.2);
        }

        /* Enhanced Status Cards */
        .status-card {
            display: flex;
            gap: 1rem;
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.6), rgba(21, 29, 53, 0.8));
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1.5px solid rgba(100, 116, 255, 0.2);
            padding: 1.5rem;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.3),
                0 2px 8px rgba(0, 217, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            height: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .status-card:hover {
            transform: translateY(-4px);
            border-color: rgba(0, 217, 255, 0.4);
            box-shadow:
                0 12px 40px rgba(0, 0, 0, 0.4),
                0 4px 16px rgba(0, 217, 255, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .status-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2.8rem;
            height: 2.8rem;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.2), rgba(124, 58, 237, 0.15));
            border: 1px solid rgba(0, 217, 255, 0.3);
            color: var(--accent-primary);
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.15);
        }

        .status-title {
            font-weight: 700;
            color: var(--text-primary);
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }

        .status-description {
            font-size: 0.88rem;
            color: var(--text-muted);
            line-height: 1.5;
        }
        /* Enhanced Risk Summary Card */
        .risk-summary-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 2rem;
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.7), rgba(21, 29, 53, 0.9));
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1.5px solid rgba(100, 116, 255, 0.25);
            padding: 2rem 2.5rem;
            box-shadow:
                0 12px 48px rgba(0, 0, 0, 0.35),
                0 4px 12px rgba(0, 217, 255, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
        }

        .risk-summary-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at top right, rgba(0, 217, 255, 0.05), transparent 60%);
            pointer-events: none;
        }

        .risk-summary-label {
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.15em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .risk-summary-value {
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
            text-shadow: 0 2px 12px rgba(0, 217, 255, 0.2);
        }

        /* Enhanced Insight Items */
        .insight-item {
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.5), rgba(21, 29, 53, 0.7));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1.5px solid rgba(100, 116, 255, 0.2);
            border-radius: 18px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.85rem;
            box-shadow:
                0 4px 16px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .insight-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            width: 4px;
            height: 60%;
            background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 0 4px 4px 0;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .insight-item:hover {
            transform: translateX(4px);
            border-color: rgba(0, 217, 255, 0.35);
            box-shadow:
                0 8px 24px rgba(0, 0, 0, 0.3),
                0 2px 8px rgba(0, 217, 255, 0.15);
        }

        .insight-item:hover::before {
            opacity: 1;
        }

        /* Enhanced Chat Messages */
        div[data-testid="stChatMessage"] {
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.6), rgba(21, 29, 53, 0.8)) !important;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            border: 1.5px solid rgba(100, 116, 255, 0.2) !important;
            border-radius: 20px !important;
            color: var(--text-secondary) !important;
            box-shadow:
                0 4px 16px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
        }

        div[data-testid="stChatMessage"] * {
            color: var(--text-secondary) !important;
        }

        /* Enhanced Chat Input */
        div[data-testid="stChatInput"] textarea {
            background: rgba(10, 14, 26, 0.7) !important;
            backdrop-filter: blur(15px) !important;
            -webkit-backdrop-filter: blur(15px) !important;
            color: var(--text-primary) !important;
            border-radius: 18px !important;
            border: 1.5px solid rgba(100, 116, 255, 0.25) !important;
            box-shadow:
                0 4px 16px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
            padding: 1rem !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }

        div[data-testid="stChatInput"] textarea:focus {
            background: rgba(10, 14, 26, 0.9) !important;
            border-color: var(--accent-primary) !important;
            box-shadow:
                0 0 0 3px rgba(0, 217, 255, 0.15),
                0 6px 24px rgba(0, 217, 255, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        }

        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
            color: #ffffff !important;
            border-radius: 14px !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.25) !important;
            transition: all 0.3s ease !important;
        }

        div[data-testid="stChatInput"] button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4) !important;
        }

        /* Enhanced Assistant Tip */
        .assistant-tip {
            margin-top: 1.25rem;
            background: linear-gradient(135deg, rgba(26, 35, 66, 0.5), rgba(21, 29, 53, 0.7));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1.5px dashed rgba(0, 217, 255, 0.3);
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            color: var(--text-secondary);
            box-shadow:
                0 4px 12px rgba(0, 0, 0, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .assistant-tip ul {
            margin: 0.5rem 0 0 1.5rem;
            padding: 0;
            color: var(--text-secondary);
        }

        .assistant-tip ul li {
            margin-bottom: 0.4rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-pro")

    xgb_model, preprocessor, optimal_threshold = load_model_components(
        MODEL_JSON_PATH,
        PREPROCESSOR_PATH,
        THRESHOLD_PATH,
    )
    if xgb_model is None or preprocessor is None:
        st.error("Unable to load model components. Verify the paths and try again.")
        st.stop()

    diabetic_averages = load_diabetic_averages(AVERAGES_PATH)
    gemini_model = initialize_gemini(api_key, model_name)

    if "optimal_threshold_default" not in st.session_state:
        st.session_state.optimal_threshold_default = optimal_threshold
    if (
        "optimal_threshold" not in st.session_state
        or st.session_state.optimal_threshold == 0.5
    ):
        st.session_state.optimal_threshold = optimal_threshold

    prediction_made = st.session_state.prediction_made
    probability = st.session_state.prediction_prob

    if prediction_made:
        risk_level, risk_badge_class, risk_message = classify_risk(probability)
        hero_metric = f"{probability:.0f}%"
        hero_badge_label = risk_level
    else:
        risk_level, risk_badge_class, risk_message = (
            "Pending",
            "risk-badge-neutral",
            "Complete the assessment to unlock your personalized guidance.",
        )
        hero_metric = "--"
        hero_badge_label = "Awaiting input"

    st.markdown(
        """
        <div class="app-hero">
            <div class="hero-left">
                <span class="hero-pill">AI-guided prevention</span>
                <h1>Diabetes Risk Navigator</h1>
                <p class="hero-subtitle">
                    Understand your health indicators, benchmark against diabetic population data,
                    and receive tailored talking points for your next appointment.
                </p>
                <div class="hero-steps">
                    <div class="hero-step">1. Share your profile</div>
                    <div class="hero-step">2. Explore insights</div>
                    <div class="hero-step">3. Chat with the assistant</div>
                </div>
            </div>
            <div class="hero-highlight">
                <div class="hero-metric-label">Current risk estimate</div>
                <div class="hero-metric-value">{}</div>
                <div class="hero-metric-badge {}">{}</div>
                <p class="hero-highlight-note">{}</p>
            </div>
        </div>
        """.format(hero_metric, risk_badge_class, hero_badge_label, risk_message),
        unsafe_allow_html=True,
    )

    status_cols = st.columns(3, gap="large")
    status_details = [
        ("SEC", "Data privacy", "No information leaves this device."),
        ("360", "Balanced inputs", "Clinical, lifestyle, and social factors in one place."),
        ("LIVE", "Dynamic guidance", "Insights and chatbot refresh whenever you update data."),
    ]
    for col, (icon, title, description) in zip(status_cols, status_details):
        col.markdown(
            f"""
            <div class="status-card">
                <span class="status-icon">{icon}</span>
                <div>
                    <div class="status-title">{title}</div>
                    <div class="status-description">{description}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    action_cols = st.columns([5, 1])
    with action_cols[1]:
        st.button(
            "Start new assessment",
            type="secondary",
            use_container_width=True,
            on_click=reset_app,
        )

    st.markdown("")

    if not prediction_made:
        st.subheader("Tell us about yourself")
        st.caption("Complete the sections below. Defaults reflect typical values and can be adjusted.")

        with st.form("health_form"):
            user_inputs = {}
            for section_index, section in enumerate(FORM_SECTIONS):
                st.markdown(
                    f"""
                    <div class="form-section-header">
                        <h3>{section["title"]}</h3>
                        <p>{section["description"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                section_cols = st.columns(2, gap="large")
                for idx, feature in enumerate(section["features"]):
                    config = FEATURE_CONFIGS[feature]
                    label = FEATURE_NAMES[feature]
                    info_text = FEATURE_INFO.get(feature)
                    target_col = section_cols[idx % 2]
                    with target_col:
                        if config["type"] == "number":
                            user_inputs[feature] = st.number_input(
                                label,
                                min_value=config["min"],
                                max_value=config["max"],
                                value=config["default"],
                                step=config["step"],
                                key=f"input_{feature}",
                            )
                            # Show help text only for number inputs
                            if config.get("help"):
                                st.caption(config["help"])
                        else:
                            # Select/dropdown field
                            options = config["options"]
                            labels = config["labels"]
                            default_value = config.get("default", options[0])
                            default_index = (
                                options.index(default_value)
                                if default_value in options
                                else 0
                            )
                            label_lookup = dict(zip(options, labels))

                            # Add help text if available (for context, not code mappings)
                            help_text = config.get("help") if config.get("help") else None

                            selected_value = st.selectbox(
                                label,
                                options=options,
                                index=default_index,
                                format_func=lambda opt, lookup=label_lookup: lookup[opt],
                                key=f"input_{feature}",
                                help=help_text
                            )
                            user_inputs[feature] = selected_value
                if section_index < len(FORM_SECTIONS) - 1:
                    st.markdown("<hr class='form-section-divider' />", unsafe_allow_html=True)

            submitted = st.form_submit_button("Analyze my risk", use_container_width=True)

            if submitted:
                clean_data = {k: sanitize_input(v, k) for k, v in user_inputs.items()}
                feature_order = list(FEATURE_CONFIGS.keys())
                input_df = pd.DataFrame([clean_data])[feature_order]

                with st.expander("Model input preview", expanded=False):
                    st.dataframe(input_df)

                try:
                    X_processed = preprocessor.transform(input_df)
                    prediction_proba = xgb_model.predict_proba(X_processed)[0]
                    probability = float(prediction_proba[1] * 100)
                    threshold_value = st.session_state.get(
                        "optimal_threshold",
                        optimal_threshold,
                    )
                    prediction = 1 if probability >= threshold_value * 100 else 0

                    with st.expander("Model output preview", expanded=False):
                        st.write(f"**Prediction:** {prediction}")
                        st.write(f"**Probabilities:** {prediction_proba}")
                        st.write(f"**Diabetes Probability:** {probability:.2f}%")

                    st.session_state.user_data = clean_data
                    st.session_state.prediction_prob = probability
                    st.session_state.prediction_made = True
                    st.session_state.system_prompt = get_system_prompt(probability, clean_data)
                    st.session_state.chat_history = []
                    st.session_state.chatbot_initialized = False
                    st.session_state.optimal_threshold = threshold_value

                    st.rerun()
                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                    st.info("Ensure the preprocessor and model files are accessible and compatible.")
    else:
        probability = st.session_state.prediction_prob
        risk_level, badge_class, risk_message = classify_risk(probability)

        st.success("Assessment captured. Explore your personalized visuals and guided assistant below.")

        with st.expander("Review your submitted profile", expanded=False):
            summary_df = pd.DataFrame([st.session_state.user_data])
            summary_df.rename(columns=FEATURE_NAMES, inplace=True)
            st.dataframe(summary_df)

        results_col, assistant_col = st.columns([7, 5], gap="large")

        with results_col:
            st.subheader("Personalized risk summary")
            st.markdown(
                """
                <div class="risk-summary-card">
                    <div>
                        <div class="risk-summary-label">Estimated probability</div>
                        <div class="risk-summary-value">{:.1f}%</div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:0.65rem; align-items:flex-start;">
                        <span class="hero-metric-badge {}">{}</span>
                        <p class="hero-highlight-note">{}</p>
                    </div>
                </div>
                """.format(probability, badge_class, risk_level, risk_message),
                unsafe_allow_html=True,
            )

            gauge_fig = create_risk_gauge(probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown("### How you compare")
            comparison_fig, differences = create_comparison_chart(
                st.session_state.user_data,
                diabetic_averages,
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

            st.markdown("### Key takeaways")
            insights = generate_insights(
                st.session_state.user_data,
                diabetic_averages,
                differences,
            )
            if insights:
                for insight in insights:
                    st.markdown(
                        f"<div class='insight-item'>&bull; {insight}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div class='insight-item'>Your profile closely matches the reference population with no standout differences.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("### What drives your risk estimate")
            waterfall_fig = create_contribution_waterfall(
                st.session_state.user_data,
                diabetic_averages,
            )
            st.plotly_chart(waterfall_fig, use_container_width=True)

            st.markdown("### Wellness radar")
            radar_fig = create_wellness_radar(
                st.session_state.user_data,
                diabetic_averages,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        with assistant_col:
            st.subheader("AI health assistant")
            context = build_app_context(probability, st.session_state.user_data)
            if gemini_model is None:
                st.caption(
                    "No Gemini API key detected. Responses are generated locally from the assessment summary."
                )

            if not st.session_state.chatbot_initialized:
                intro_prompt = (
                    "Share a concise welcome summary highlighting the user's risk score and the first three actions "
                    "they should consider this week."
                )
                if gemini_model is not None:
                    initial_response = send_message_to_gemini(
                        gemini_model,
                        "coach",
                        st.session_state.agent_histories.get("coach", []),
                        intro_prompt,
                        context,
                    )
                else:
                    initial_response = generate_fallback_response(
                        probability,
                        st.session_state.user_data,
                        "coach",
                    )
                st.session_state.agent_histories["coach"].append(
                    {"role": "assistant", "content": initial_response}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": initial_response, "agent": "coach"}
                )
                st.session_state.chatbot_initialized = True

            chat_container = st.container(height=480)
            with chat_container:
                for message in st.session_state.chat_history:
                    agent_key = message.get("agent", "coach")
                    agent_meta = AGENT_DEFINITIONS.get(agent_key, AGENT_DEFINITIONS["coach"])
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.caption(f"Routed to {agent_meta['name']}")
                            st.markdown(message["content"])
                    else:
                        with st.chat_message("assistant", avatar=agent_meta.get("avatar", "\U0001F916")):
                            st.markdown(f"**{agent_meta['name']}**\n\n{message['content']}")

            st.markdown(
                """
                <div class="assistant-tip">
                    Try asking:
                    <ul>
                        <li>What lifestyle changes would have the biggest impact for me?</li>
                        <li>How should I discuss these results with my doctor?</li>
                        <li>What labs should I request at my next visit?</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            user_message = st.chat_input("Ask about your risk, prevention, or next steps...")
            if user_message:
                agent_key = determine_agent(user_message)
                st.session_state.active_agent = agent_key
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_message, "agent": agent_key}
                )
                agent_history = st.session_state.agent_histories.setdefault(agent_key, [])
                agent_history.append({"role": "user", "content": user_message})
                history_before_user = agent_history[:-1]
                if agent_key == "scheduler":
                    assistant_response = process_scheduler_message(
                        user_message,
                        probability,
                        st.session_state.user_data,
                    )
                elif gemini_model is not None:
                    with st.spinner("Composing guidance..."):
                        assistant_response = send_message_to_gemini(
                            gemini_model,
                            agent_key,
                            history_before_user,
                            user_message,
                            context,
                        )
                else:
                    assistant_response = generate_fallback_response(
                        probability,
                        st.session_state.user_data,
                        agent_key,
                        user_message,
                    )
                agent_history.append({"role": "assistant", "content": assistant_response})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_response, "agent": agent_key}
                )
                st.rerun()

    st.markdown("---")
    st.markdown(
        "This tool offers educational guidance and does not replace professional medical advice. "
        "Consult your care team before making treatment decisions."
    )

if __name__ == "__main__":
    main()
