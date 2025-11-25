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

# Suppress sklearn version warnings for better UX
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths (adjust as needed)
MODEL_PATH = "model_output/ml_xgboost_smoteenn_pipeline.pkl"
THRESHOLD_PATH = "model_output/optimal_threshold.json"
AVERAGES_PATH = "model_output/diabetic_averages.json"

# PREPROCESSING NOTES:
# ====================
# Your model is an imblearn Pipeline with:
#   1. ColumnTransformer (scales numeric: BMI, GenHlth, PhysHlth, Age, Education, Income)
#   2. SMOTEENN sampler (only active during training, skipped during prediction)
#   3. XGBoost classifier
#
# The app passes RAW user input → Pipeline handles scaling automatically.
# Feature order MUST match training: GenHlth, HighBP, DiffWalk, BMI, HighChol,
#                                     Age, HeartDiseaseorAttack, PhysHlth, Income,
#                                     Education, PhysActivity

# Feature definitions - MUST MATCH TRAINING ORDER
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

# Multi-Agent System Definitions
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
    "wellness": {
        "name": "Wellness Advisor",
        "avatar": "🌟",
        "keywords": ["mental", "mindfulness", "meditation", "wellbeing", "balance", "lifestyle"],
        "system_prompt": (
            "You are a wellness advisor focusing on holistic health and mental wellbeing.\n"
            "Risk probability: {probability:.1f}% ({risk_level}).\n"
            "Key metrics:\n{profile_summary}\n\n"
            "Provide guidance on stress management, mental health, mindfulness, and overall wellbeing as it relates to diabetes prevention."
        ),
        "fallback_focus": (
            "Guide on stress reduction, mindfulness practices, work-life balance, and mental health support."
        ),
    },
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

def load_model(model_path: str):
    """Load the trained model from disk with version compatibility checks."""
    try:
        # Suppress sklearn warnings during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try joblib first, then pickle
            try:
                model = joblib.load(model_path)
            except:
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        
        # Check if it's a dict or other container
        if isinstance(model, dict):
            # Try to find the pipeline in the dict
            if 'pipeline' in model:
                model = model['pipeline']
            elif 'model' in model:
                model = model['model']
            else:
                st.error("❌ Could not find pipeline in dictionary")
                return None
        
        # Check model type
        from sklearn.pipeline import Pipeline
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            is_imblearn_pipeline = isinstance(model, ImbPipeline)
        except:
            is_imblearn_pipeline = False
        
        is_sklearn_pipeline = isinstance(model, Pipeline)
        is_pipeline = is_sklearn_pipeline or is_imblearn_pipeline
        
        # Verify it has predict method
        if not hasattr(model, 'predict'):
            st.error(f"❌ Loaded object is {type(model)} but has no 'predict' method")
            st.error("This might not be a trained model. Check how the model was saved.")
            return None
        
        # Verify model can make predictions
        try:
            # Test prediction with dummy data in CORRECT ORDER
            test_features = list(FEATURE_CONFIGS.keys())
            test_data = pd.DataFrame([{f: 0 if FEATURE_CONFIGS[f]["type"] == "select" 
                                       else FEATURE_CONFIGS[f]["default"] 
                                       for f in test_features}])
            
            _ = model.predict(test_data)
            
            return model
        except Exception as pred_error:
            st.error(f"⚠️ Model loaded but prediction test failed: {pred_error}")
            st.error(f"Error type: {type(pred_error).__name__}")
            import traceback
            st.code(traceback.format_exc())
            st.warning("The model may be incompatible with the current scikit-learn version or feature order is wrong.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model from {model_path}: {e}")
        st.info("💡 Tip: Ensure the model file exists and was created with a compatible scikit-learn version.")
        import traceback
        st.code(traceback.format_exc())
        return None

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

def determine_agent(user_message: str, current_agent: str = "coach") -> str:
    """Route the user message to the best-fit agent based on keywords."""
    text = user_message.lower()
    for key, agent in AGENT_DEFINITIONS.items():
        for keyword in agent["keywords"]:
            if keyword in text:
                return key
    return current_agent  # Keep current agent if no keyword match

def build_app_context(probability: float, user_data: Dict[str, float]) -> Dict[str, str]:
    """Assemble common context fields for agents."""
    risk_level, _, _ = classify_risk(probability)
    return {
        "probability": probability,
        "risk_level": risk_level,
        "profile_summary": build_profile_summary(user_data),
    }

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
    """Generate system prompt with prediction context - DEPRECATED, use agent-specific prompts."""
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

def generate_fallback_response(probability: float,
                               user_data: Dict[str, float],
                               agent_key: str,
                               user_message: Optional[str] = None) -> str:
    """Provide an on-device assistant response when Gemini is unavailable."""
    agent = AGENT_DEFINITIONS[agent_key]
    context = build_app_context(probability, user_data)
    risk_level = context["risk_level"]
    summary = (
        f"{agent['name']} (offline mode)\n"
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

    elif agent_key == "wellness":
        guidance.append("Practice daily mindfulness or meditation for 10-15 minutes to reduce stress hormones.")
        guidance.append("Maintain a consistent sleep schedule; aim for 7-9 hours to support metabolic health.")
        guidance.append("Consider journaling to track emotional eating patterns and stress triggers.")
        guidance.append("Explore yoga or tai chi for combined physical and mental benefits.")

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
            color='#2563eb',
            line=dict(color='#1d4ed8', width=1.2)
        ),
        text=[f'{v:.1f}' for v in user_values],
        textposition='outside',
        textfont=dict(color='#0f172a', size=13, family='"Inter","Segoe UI",sans-serif')
    ))
    
    # Average diabetic values
    fig.add_trace(go.Bar(
        name='Avg. Diabetic Population',
        x=features,
        y=avg_values,
        marker=dict(
            color='#f97316',
            line=dict(color='#ea580c', width=1.2)
        ),
        text=[f'{v:.1f}' for v in avg_values],
        textposition='outside',
        textfont=dict(color='#0f172a', size=13, family='"Inter","Segoe UI",sans-serif')
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
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#dbeafe',
            borderwidth=1
        ),
        uniformtext=dict(mode="show", minsize=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f172a', family='"Inter","Segoe UI",sans-serif'),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    fig.update_xaxes(
        tickangle=-35,
        showgrid=False,
        linecolor='#cbd5f5',
        tickfont=dict(color='#0f172a', size=12, family='"Inter","Segoe UI",sans-serif')
    )
    fig.update_yaxes(
        gridcolor='#e2e8f0',
        zerolinecolor="#cbd5f5",
        tickfont=dict(color='#0f172a', size=12, family='"Inter","Segoe UI",sans-serif')
    )

    return fig, differences

def classify_risk(probability: float) -> Tuple[str, str, str]:
    """Translate probability into a labeled risk tier, CSS badge class, and guidance."""
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
            'font': {'size': 24, 'color': '#0f172a', 'family': '"Inter","Segoe UI",sans-serif'}
        },
        number={'suffix': "%", 'font': {'size': 48, 'color': '#0f172a', 'family': '"Inter","Segoe UI",sans-serif'}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "#0f172a",
                'tickfont': {'color': '#0f172a', 'family': '"Inter","Segoe UI",sans-serif'}
            },
            'bar': {'color': "#2563EB"},
            'bgcolor': "#ffffff",
            'borderwidth': 2,
            'bordercolor': "#dbeafe",
            'steps': [
                {'range': [0, 30], 'color': '#bfdbfe'},
                {'range': [30, 60], 'color': '#fde68a'},
                {'range': [60, 100], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='#f1f5f9'
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
    """Send message to Gemini and get response using agent-specific system prompt."""
    try:
        agent = AGENT_DEFINITIONS[agent_key]
        system_prompt = agent["system_prompt"].format(**context)

        # Build conversation history
        chat = model.start_chat(history=[])

        # Send system prompt
        chat.send_message(system_prompt)

        # Send previous messages for this agent
        for msg in conversation:
            chat.send_message(msg['content'])

        # Send current message
        response = chat.send_message(user_message)
        return response.text

    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."

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
    if 'agent_histories' not in st.session_state:
        st.session_state.agent_histories = {key: [] for key in AGENT_DEFINITIONS}
    if 'active_agent' not in st.session_state:
        st.session_state.active_agent = "coach"
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = "coach"

def reset_app():
    """Reset application state."""
    st.session_state.prediction_made = False
    st.session_state.user_data = {}
    st.session_state.prediction_prob = 0.0
    st.session_state.chat_history = []
    st.session_state.chatbot_initialized = False
    st.session_state.system_prompt = ""
    st.session_state.agent_histories = {key: [] for key in AGENT_DEFINITIONS}
    st.session_state.active_agent = "coach"
    st.session_state.selected_agent = "coach"

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
            color-scheme: light;
        }
        body, .stApp {
            background-color: #f5f7fb !important;
            color: #0f172a !important;
            font-family: "Inter", "Segoe UI", sans-serif !important;
        }
        div[data-testid="stAppViewContainer"] {
            background-color: #f5f7fb !important;
        }
        div[data-testid="stSidebar"] {
            background-color: #eef2ff !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #0f172a !important;
        }
        button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5, #2563eb) !important;
            color: #f8fafc !important;
            border: 1px solid #4338ca !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            box-shadow: 0 16px 36px rgba(79, 70, 229, 0.25) !important;
        }
        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #4338ca, #1d4ed8) !important;
        }
        button[kind="secondary"] {
            background: #ffffff !important;
            color: #1d4ed8 !important;
            border: 1px solid #c7d2fe !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
        }
        button[kind="secondary"]:hover {
            background: #e0e7ff !important;
        }
        form[data-testid="stForm"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 20px !important;
            padding: 2rem 2.5rem !important;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08) !important;
        }
        form[data-testid="stForm"] label {
            font-weight: 600 !important;
            color: #0f172a !important;
        }
        .form-section-header h3 {
            margin-bottom: 0.25rem;
            color: #0f172a;
        }
        .form-section-header p {
            margin-top: 0;
            color: #475569;
        }
        .form-section-divider {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, rgba(148, 163, 184, 0.2), rgba(148, 163, 184, 0));
        }
        .app-hero {
            display: grid;
            grid-template-columns: minmax(0, 2.4fr) minmax(0, 1fr);
            gap: 2.75rem;
            padding: 2.75rem;
            border-radius: 28px;
            border: 1px solid #c7d2fe;
            background: linear-gradient(135deg, #eef2ff, #dbeafe);
            box-shadow: 0 28px 60px rgba(79, 70, 229, 0.22);
            margin-bottom: 1.5rem;
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
            background: rgba(255, 255, 255, 0.45);
            color: #4338ca;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #1e293b;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }
        .hero-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .hero-step {
            background: rgba(255, 255, 255, 0.65);
            color: #1d4ed8;
            font-weight: 600;
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
        }
        .hero-highlight {
            background: #ffffff;
            border-radius: 22px;
            border: 1px solid #e0e7ff;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            box-shadow: 0 22px 45px rgba(15, 23, 42, 0.12);
        }
        .hero-metric-label {
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            color: #475569;
        }
        .hero-metric-value {
            font-size: 3rem;
            font-weight: 700;
            color: #1e293b;
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
            color: #475569;
        }
        .risk-badge-low {
            background: #dcfce7;
            color: #15803d;
        }
        .risk-badge-medium {
            background: #fef9c3;
            color: #b45309;
        }
        .risk-badge-high {
            background: #fee2e2;
            color: #b91c1c;
        }
        .risk-badge-neutral {
            background: #e0f2fe;
            color: #0c4a6e;
        }
        .status-card {
            display: flex;
            gap: 0.75rem;
            background: #ffffff;
            border-radius: 18px;
            border: 1px solid #e2e8f0;
            padding: 1rem 1.2rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            height: 100%;
        }
        .status-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2.5rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #4338ca;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }
        .status-title {
            font-weight: 600;
            color: #0f172a;
        }
        .status-description {
            font-size: 0.9rem;
            color: #475569;
            margin-top: 0.25rem;
        }
        .risk-summary-card {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            background: #ffffff;
            border-radius: 20px;
            border: 1px solid #e0e7ff;
            padding: 1.75rem 2rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.5rem;
        }
        .risk-summary-label {
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            color: #475569;
        }
        .risk-summary-value {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1e293b;
        }
        .insight-item {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            color: #0f172a;
            font-weight: 500;
        }
        div[data-testid="stChatMessage"] {
            background: linear-gradient(135deg, #f8fafc, #eef2ff) !important;
            border: 1px solid #e0e7ff !important;
            border-radius: 18px !important;
            color: #0f172a !important;
            box-shadow: 0 16px 35px rgba(15, 23, 42, 0.08) !important;
        }
        div[data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            border-radius: 16px !important;
            border: 1px solid #cbd5f5 !important;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08) !important;
        }
        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
            color: #0f172a !important;
            border-radius: 12px !important;
        }
        .assistant-tip {
            margin-top: 1rem;
            background: #f8fafc;
            border: 1px dashed #cbd5f5;
            border-radius: 14px;
            padding: 0.85rem 1.1rem;
            color: #334155;
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

    model = load_model(MODEL_PATH)
    diabetic_averages = load_diabetic_averages(AVERAGES_PATH)
    gemini_model = initialize_gemini(api_key, model_name)

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

    # ============ FLOW-BASED UI: Form → Analysis & Visualizations → AI Assistant ============

    if not prediction_made:
        # ============ STEP 1: ASSESSMENT FORM ============
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
                            default_index = options.index(default_value)
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
                if model is None:
                    st.error("Model could not be loaded. Verify the model path and try again.")
                else:
                    clean_data = {k: sanitize_input(v, k) for k, v in user_inputs.items()}
                    feature_order = list(FEATURE_CONFIGS.keys())
                    input_df = pd.DataFrame([clean_data])[feature_order]

                    with st.expander("Model input preview", expanded=False):
                        st.dataframe(input_df)

                    try:
                        probability = float(model.predict_proba(input_df)[0][1] * 100)
                        st.session_state.user_data = clean_data
                        st.session_state.prediction_prob = probability
                        st.session_state.prediction_made = True
                        st.session_state.system_prompt = get_system_prompt(probability, clean_data)
                        st.session_state.chat_history = []
                        st.session_state.chatbot_initialized = False
                        st.rerun()
                    except Exception as error:
                        st.error(f"Prediction failed: {error}")
                        st.info("Check that the model expects these features and is accessible from disk.")
    else:
        # ============ STEP 2 & 3: ANALYSIS + AI ASSISTANT (Side-by-side) ============
        probability = st.session_state.prediction_prob
        risk_level, badge_class, risk_message = classify_risk(probability)

        st.success("✅ Assessment complete! Explore your personalized analysis and chat with healthcare AI agents below.")

        with st.expander("📋 Review your submitted profile", expanded=False):
            summary_df = pd.DataFrame([st.session_state.user_data])
            summary_df.rename(columns=FEATURE_NAMES, inplace=True)
            st.dataframe(summary_df)

        # Create two-column layout: Analysis (left) + Chatbot (right)
        results_col, assistant_col = st.columns([7, 5], gap="large")

        with results_col:
            # ============ ANALYSIS & VISUALIZATIONS ============
            st.subheader("📊 Personalized Risk Summary")
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

            st.markdown("### 📈 How you compare")
            comparison_fig, differences = create_comparison_chart(
                st.session_state.user_data,
                diabetic_averages,
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

            st.markdown("### 💡 Key takeaways")
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

        with assistant_col:
            # ============ AI HEALTH ASSISTANT WITH MULTI-AGENT SUPPORT ============
            st.subheader("🤖 Healthcare AI Agents")

            # Agent selector dropdown
            agent_options = list(AGENT_DEFINITIONS.keys())
            agent_labels = [f"{AGENT_DEFINITIONS[k]['avatar']} {AGENT_DEFINITIONS[k]['name']}" for k in agent_options]

            selected_agent_index = st.selectbox(
                "Select Healthcare Agent",
                options=range(len(agent_options)),
                format_func=lambda i: agent_labels[i],
                index=agent_options.index(st.session_state.selected_agent),
                key="agent_selector"
            )
            st.session_state.selected_agent = agent_options[selected_agent_index]

            context = build_app_context(probability, st.session_state.user_data)

            if gemini_model is None:
                st.caption("⚠️ No Gemini API key detected. Using offline mode with limited responses.")

            # Initialize chatbot automatically with selected agent
            if not st.session_state.chatbot_initialized:
                intro_prompt = (
                    "Share a concise welcome summary highlighting the user's risk score and the first three actions "
                    "they should consider this week."
                )
                if gemini_model is not None:
                    initial_response = send_message_to_gemini(
                        gemini_model,
                        st.session_state.selected_agent,
                        st.session_state.agent_histories.get(st.session_state.selected_agent, []),
                        intro_prompt,
                        context,
                    )
                else:
                    initial_response = generate_fallback_response(
                        probability,
                        st.session_state.user_data,
                        st.session_state.selected_agent,
                    )
                st.session_state.agent_histories[st.session_state.selected_agent].append(
                    {"role": "assistant", "content": initial_response}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": initial_response, "agent": st.session_state.selected_agent}
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
                        with st.chat_message("assistant", avatar=agent_meta.get("avatar", "🤖")):
                            st.markdown(f"**{agent_meta['name']}**\n\n{message['content']}")

            st.markdown(
                """
                <div class="assistant-tip">
                    💬 Try asking:
                    <ul>
                        <li>What lifestyle changes would have the biggest impact?</li>
                        <li>What should I discuss with my doctor?</li>
                        <li>Can you suggest a meal plan?</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            user_message = st.chat_input("Ask about your risk, prevention, or next steps...")
            if user_message:
                # Determine agent: use selected agent by default, but allow keyword routing
                agent_key = determine_agent(user_message, st.session_state.selected_agent)
                st.session_state.active_agent = agent_key

                st.session_state.chat_history.append(
                    {"role": "user", "content": user_message, "agent": agent_key}
                )
                agent_history = st.session_state.agent_histories.setdefault(agent_key, [])
                agent_history.append({"role": "user", "content": user_message})
                history_before_user = agent_history[:-1]

                if gemini_model is not None:
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
