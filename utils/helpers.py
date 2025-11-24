"""
Utility helper functions for the Health AI Chatbot.

This module contains helper functions for:
- Feature value formatting
- Profile summary generation
- Risk classification
- Input sanitization
"""

from typing import Dict, Tuple
from config.settings import FEATURE_CONFIGS, FEATURE_NAMES


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
