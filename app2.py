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
from typing import Dict, List, Tuple, Optional
import joblib
import sklearn
from xgboost import XGBClassifier

# Suppress sklearn version warnings for better UX
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths (adjust as needed)
# MODEL_PATH = "model_output/ml_xgboost_smoteenn_pipeline.pkl"

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
    "Income": {"type": "number", "min": 1, "max": 8, "default": 4, "step": 1,
               "help": "Income category (1=<$10k, 8=>$75k)"},
    "Education": {"type": "number", "min": 1, "max": 6, "default": 4, "step": 1,
                  "help": "Education level (1=Never, 6=College grad)"},
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

# def load_model(model_path: str):
#     """Load the trained model from disk with version compatibility checks."""
#     try:
#         # Suppress sklearn warnings during loading
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             # Try joblib first, then pickle
#             try:
#                 model = joblib.load(model_path)
#             except:
#                 import pickle
#                 with open(model_path, 'rb') as f:
#                     model = pickle.load(f)
        
#         # Check if it's a dict or other container
#         if isinstance(model, dict):
#             # Try to find the pipeline in the dict
#             if 'pipeline' in model:
#                 model = model['pipeline']
#             elif 'model' in model:
#                 model = model['model']
#             else:
#                 st.error("❌ Could not find pipeline in dictionary")
#                 return None
        
#         # Check model type
#         from sklearn.pipeline import Pipeline
#         try:
#             from imblearn.pipeline import Pipeline as ImbPipeline
#             is_imblearn_pipeline = isinstance(model, ImbPipeline)
#         except:
#             is_imblearn_pipeline = False
        
#         is_sklearn_pipeline = isinstance(model, Pipeline)
#         is_pipeline = is_sklearn_pipeline or is_imblearn_pipeline
        
#         # Verify it has predict method
#         if not hasattr(model, 'predict'):
#             st.error(f"❌ Loaded object is {type(model)} but has no 'predict' method")
#             st.error("This might not be a trained model. Check how the model was saved.")
#             return None
        
#         # Verify model can make predictions
#         try:
#             # Test prediction with dummy data in CORRECT ORDER
#             test_features = list(FEATURE_CONFIGS.keys())
#             test_data = pd.DataFrame([{f: 0 if FEATURE_CONFIGS[f]["type"] == "select" 
#                                        else FEATURE_CONFIGS[f]["default"] 
#                                        for f in test_features}])
            
#             _ = model.predict(test_data)
            
#             return model
#         except Exception as pred_error:
#             st.error(f"⚠️ Model loaded but prediction test failed: {pred_error}")
#             st.error(f"Error type: {type(pred_error).__name__}")
#             import traceback
#             st.code(traceback.format_exc())
#             st.warning("The model may be incompatible with the current scikit-learn version or feature order is wrong.")
#             return None
            
#     except Exception as e:
#         st.error(f"❌ Error loading model from {model_path}: {e}")
#         st.info("💡 Tip: Ensure the model file exists and was created with a compatible scikit-learn version.")
#         import traceback
#         st.code(traceback.format_exc())
#         return None

# NEW FUNCTION: Load model components separately
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
            color='#2563EB',
            line=dict(color='#1E3A8A', width=1.2)
        ),
        text=[f'{v:.1f}' for v in user_values],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=13, family='"Inter","Segoe UI",sans-serif')
    ))
    
    # Average diabetic values
    fig.add_trace(go.Bar(
        name='Avg. Diabetic Population',
        x=features,
        y=avg_values,
        marker=dict(
            color='#A855F7',
            line=dict(color='#6D28D9', width=1.2)
        ),
        text=[f'{v:.1f}' for v in avg_values],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=13, family='"Inter","Segoe UI",sans-serif')
    ))
    
    fig.update_layout(
        title='Your Health Metrics vs. Average Diabetic Population',
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
            bgcolor='rgba(11,17,32,0.85)',
            bordercolor='#1f2a4d',
            borderwidth=1
        ),
        uniformtext=dict(mode="show", minsize=12),
        paper_bgcolor='#0b1120',
        plot_bgcolor='#111c3a',
        font=dict(color='#e2e8f0', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    fig.update_xaxes(
        tickangle=-35,
        showgrid=False,
        linecolor='#1f2a4d',
        tickfont=dict(color='#cbd5f5', size=12, family='"Inter","Segoe UI",sans-serif')
    )
    fig.update_yaxes(
        gridcolor='#1f2a4d',
        zerolinecolor='#1f2a4d',
        tickfont=dict(color='#cbd5f5', size=12, family='"Inter","Segoe UI",sans-serif')
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
            'font': {'size': 24, 'color': '#f8fafc', 'family': '"Inter","Segoe UI",sans-serif'}
        },
        number={'suffix': "%", 'font': {'size': 48, 'color': '#f8fafc', 'family': '"Inter","Segoe UI",sans-serif'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#1e293b",
                'tickfont': {'color': '#cbd5f5', 'family': '"Inter","Segoe UI",sans-serif'}
            },
            'bar': {'color': "#38bdf8"},
            'bgcolor': "#111c3a",
            'borderwidth': 2,
            'bordercolor': "#1f2a4d",
            'steps': [
                {'range': [0, 30], 'color': '#134e4a'},
                {'range': [30, 60], 'color': '#78350f'},
                {'range': [60, 100], 'color': '#7f1d1d'}
            ],
            'threshold': {
                'line': {'color': "#f97316", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='#0b1120'
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
            connector={"mode": "spanning", "line": {"color": "#1f2a4d", "width": 1}},
            decreasing={"marker": {"color": "#f87171"}},
            increasing={"marker": {"color": "#38bdf8"}},
        )
    )

    fig.update_layout(
        title="Feature shifts versus diabetic average (normalized)",
        showlegend=False,
        height=420,
        margin=dict(l=120, r=30, t=60, b=40),
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111c3a",
        font=dict(color="#e2e8f0", family='"Inter","Segoe UI",sans-serif'),
        xaxis=dict(
            title="Relative shift (percentage points)",
            gridcolor="#1f2a4d",
            zerolinecolor="#1f2a4d",
            tickfont=dict(color="#cbd5f5"),
        ),
        yaxis=dict(tickfont=dict(color="#cbd5f5")),
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
        "Your profile": "#38bdf8",
        "Ideal baseline": "#22d3ee",
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
                tickfont=dict(size=10, color="#cbd5f5"),
                gridcolor="#1f2a4d",
                linecolor="#1f2a4d",
            ),
            bgcolor="#111c3a",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5f5"),
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#0b1120",
        title="Wellness balance across key factors",
        font=dict(color="#e2e8f0", family='"Inter","Segoe UI",sans-serif'),
        height=420,
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

def send_message_to_gemini(model, chat_history: List[Dict[str, str]], 
                          user_message: str) -> str:
    """Send message to Gemini and get response."""
    try:
        # Build conversation history
        chat = model.start_chat(history=[])
        
        # Send system prompt as first message if this is the start
        if len(chat_history) == 0:
            system_msg = st.session_state.get('system_prompt', '')
            if system_msg:
                chat.send_message(system_msg)
        
        # Send previous messages
        for msg in chat_history:
            if msg['role'] == 'user':
                chat.send_message(msg['content'])
            # Assistant messages are already in history
        
        # Send current message
        response = chat.send_message(user_message)
        return response.text
    
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."


def generate_fallback_response(probability: float,
                               user_data: Dict[str, float],
                               user_message: Optional[str] = None) -> str:
    """Provide an on-device assistant response when Gemini is unavailable."""
    risk_level, _, risk_message = classify_risk(probability)
    summary = (
        f"The assessment estimates a {probability:.1f}% likelihood of diabetes, "
        f"which places you in the {risk_level.lower()} risk range. {risk_message}"
    )

    lifestyle_hints: List[str] = []
    bmi = user_data.get("BMI")
    if bmi is not None and bmi >= 30:
        lifestyle_hints.append(
            "Set a realistic weekly movement goal and focus on gradual weight management."
        )
    if user_data.get("PhysActivity") == 0:
        lifestyle_hints.append(
            "Incorporate at least 150 minutes of moderate activity per week, even if you start with 10-minute walks."
        )
    if user_data.get("HighBP") == 1:
        lifestyle_hints.append(
            "Monitor blood pressure regularly and discuss medication adherence with your clinician."
        )
    if user_data.get("HighChol") == 1:
        lifestyle_hints.append(
            "Review cholesterol numbers with your care team and ask whether dietary adjustments could help."
        )

    if not lifestyle_hints:
        lifestyle_hints.append(
            "Keep reinforcing balanced nutrition, regular movement, quality sleep, and stress reduction."
        )

    action_plan = "Here are next steps to consider:\n- " + "\n- ".join(lifestyle_hints[:3])

    if user_message:
        return (
            f"You asked: \"{user_message}\".\n\n"
            f"{summary}\n\n"
            f"{action_plan}\n\n"
            "Always review these ideas with a healthcare professional who knows your full history."
        )

    return (
        f"{summary}\n\n"
        f"{action_plan}\n\n"
        "Use the chat to dig deeper into lifestyle changes, lab tests, or questions for your next appointment."
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


def main():
    st.set_page_config(
        page_title="Diabetes Risk Navigator + AI Health Assistant",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        body, .stApp {
            background-color: #0b1120 !important;
            color: #e2e8f0 !important;
            font-family: "Inter", "Segoe UI", sans-serif !important;
        }
        div[data-testid="stAppViewContainer"] {
            background-color: #0b1120 !important;
        }
        div[data-testid="stSidebar"] {
            background-color: #111c3a !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #f8fafc !important;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown strong, .stMarkdown em {
            color: #e2e8f0 !important;
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #38bdf8, #2563eb) !important;
            color: #0b1120 !important;
            border: 1px solid #38bdf8 !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            box-shadow: 0 18px 38px rgba(37, 99, 235, 0.35) !important;
        }
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        }
        button[kind="secondary"] {
            background: transparent !important;
            color: #60a5fa !important;
            border: 1px solid #1d4ed8 !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
        }
        button[kind="secondary"]:hover {
            background: rgba(37, 99, 235, 0.15) !important;
        }
        form[data-testid="stForm"] {
            background: #111c3a !important;
            border: 1px solid #1f2a4d !important;
            border-radius: 22px !important;
            padding: 2.2rem !important;
            box-shadow: 0 28px 55px rgba(15, 23, 42, 0.5) !important;
        }
        form[data-testid="stForm"] label {
            font-weight: 600 !important;
            color: #e2e8f0 !important;
        }
        .form-section-header h3 {
            margin-bottom: 0.25rem;
            color: #f1f5f9;
        }
        .form-section-header p {
            margin-top: 0;
            color: #94a3b8;
        }
        .form-section-divider {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, rgba(148, 163, 184, 0.1), rgba(148, 163, 184, 0));
        }
        .stApp input, .stApp textarea, div[data-baseweb="input"] input {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
        }
        div[data-testid="stNumberInput"] input {
            background-color: #0f172a !important;
            color: #f8fafc !important;
        }
        div[data-testid="stNumberInput"] button {
            background-color: #1e293b !important;
            color: #cbd5f5 !important;
        }
        .app-hero {
            display: grid;
            grid-template-columns: minmax(0, 2.4fr) minmax(0, 1fr);
            gap: 2.5rem;
            padding: 3rem;
            border-radius: 30px;
            border: 1px solid #1f2a4d;
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.95), rgba(15, 23, 42, 0.95));
            box-shadow: 0 32px 70px rgba(15, 23, 42, 0.6);
            margin-bottom: 1.75rem;
        }
        .hero-left h1 {
            margin-bottom: 0.75rem;
            font-size: 2.4rem;
            font-weight: 700;
        }
        .hero-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(37, 214, 238, 0.15);
            color: #38bdf8;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #cbd5f5;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .hero-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .hero-step {
            background: rgba(37, 99, 235, 0.18);
            color: #93c5fd;
            font-weight: 600;
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
        }
        .hero-highlight {
            background: #111c3a;
            border-radius: 24px;
            border: 1px solid #1f2a4d;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.25);
        }
        .hero-metric-label {
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            color: #94a3b8;
        }
        .hero-metric-value {
            font-size: 3.1rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .hero-metric-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 600;
        }
        .hero-highlight-note {
            font-size: 0.92rem;
            color: #cbd5f5;
        }
        .risk-badge-low {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
        }
        .risk-badge-medium {
            background: rgba(251, 191, 36, 0.25);
            color: #facc15;
        }
        .risk-badge-high {
            background: rgba(248, 113, 113, 0.25);
            color: #f87171;
        }
        .risk-badge-neutral {
            background: rgba(59, 130, 246, 0.25);
            color: #60a5fa;
        }
        .status-card {
            display: flex;
            gap: 0.75rem;
            background: #111c3a;
            border-radius: 18px;
            border: 1px solid #1f2a4d;
            padding: 1rem 1.2rem;
            box-shadow: 0 22px 46px rgba(15, 23, 42, 0.45);
            height: 100%;
        }
        .status-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2.5rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.2);
            color: #38bdf8;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .status-title {
            font-weight: 600;
            color: #f1f5f9;
        }
        .status-description {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }
        .risk-summary-card {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            background: #111c3a;
            border-radius: 20px;
            border: 1px solid #1f2a4d;
            padding: 1.75rem 2rem;
            box-shadow: 0 22px 46px rgba(15, 23, 42, 0.45);
            margin-bottom: 1.5rem;
        }
        .risk-summary-label {
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            color: #94a3b8;
        }
        .risk-summary-value {
            font-size: 2.8rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .insight-item {
            background: #111c3a;
            border: 1px solid #1f2a4d;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.4);
            color: #f1f5f9;
            font-weight: 500;
        }
        div[data-testid="stChatMessage"] {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95)) !important;
            border: 1px solid #1f2a4d !important;
            border-radius: 18px !important;
            color: #e2e8f0 !important;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.45) !important;
        }
        div[data-testid="stChatMessage"] * {
            color: #e2e8f0 !important;
        }
        div[data-testid="stChatInput"] textarea {
            background: #111c3a !important;
            color: #e2e8f0 !important;
            border-radius: 16px !important;
            border: 1px solid #1f2a4d !important;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.5) !important;
        }
        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
            color: #0b1120 !important;
            border-radius: 12px !important;
        }
        .assistant-tip {
            margin-top: 1rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px dashed rgba(59, 130, 246, 0.6);
            border-radius: 14px;
            padding: 0.85rem 1.1rem;
            color: #cbd5f5;
        }
        .assistant-tip ul {
            margin: 0.35rem 0 0 1.1rem;
            padding: 0;
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
                        else:
                            options = config["options"]
                            labels = config["labels"]
                            default_value = config.get("default", options[0])
                            default_index = (
                                options.index(default_value)
                                if default_value in options
                                else 0
                            )
                            label_lookup = dict(zip(options, labels))
                            selected_value = st.selectbox(
                                label,
                                options=options,
                                index=default_index,
                                format_func=lambda opt, lookup=label_lookup: lookup[opt],
                                key=f"input_{feature}",
                            )
                            user_inputs[feature] = selected_value
                        if info_text:
                            st.caption(info_text)
                        elif config.get("help"):
                            st.caption(config["help"])
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
            if gemini_model is None:
                st.caption(
                    "No Gemini API key detected. Responses are generated locally from the assessment summary."
                )

            if not st.session_state.chatbot_initialized:
                if gemini_model is not None:
                    initial_response = send_message_to_gemini(
                        gemini_model,
                        [],
                        st.session_state.system_prompt,
                    )
                else:
                    initial_response = generate_fallback_response(
                        probability,
                        st.session_state.user_data,
                    )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": initial_response}
                )
                st.session_state.chatbot_initialized = True

            chat_container = st.container(height=480)
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(message["content"])
                    else:
                        with st.chat_message("assistant", avatar="\U0001F916"):
                            st.markdown(message["content"])

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
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_message}
                )
                if gemini_model is not None:
                    with st.spinner("Composing guidance..."):
                        assistant_response = send_message_to_gemini(
                            gemini_model,
                            st.session_state.chat_history[:-1],
                            user_message,
                        )
                else:
                    assistant_response = generate_fallback_response(
                        probability,
                        st.session_state.user_data,
                        user_message,
                    )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_response}
                )
                st.rerun()

    st.markdown("---")
    st.markdown(
        "This tool offers educational guidance and does not replace professional medical advice. "
        "Consult your care team before making treatment decisions."
    )

if __name__ == "__main__":
    main()
