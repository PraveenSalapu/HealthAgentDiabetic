"""
Model loading utilities for the Health AI Chatbot.

This module handles loading and validation of:
- XGBoost model from JSON
- Preprocessor pipeline
- Optimal threshold
- Diabetic population averages
"""

import json
import warnings
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

from config.settings import (
    FEATURE_CONFIGS,
    DEFAULT_DIABETIC_AVERAGES,
)


def load_model_components(
    model_json_path: str, 
    preprocessor_path: str, 
    threshold_path: str
) -> Tuple[Optional[XGBClassifier], Optional[object], Optional[float]]:
    """
    Load XGBoost model (JSON), preprocessor (pkl), and optimal threshold.
    
    Args:
        model_json_path: Path to XGBoost model JSON file
        preprocessor_path: Path to preprocessor pickle file
        threshold_path: Path to threshold JSON file
    
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
        test_data = pd.DataFrame([{
            f: 0 if FEATURE_CONFIGS[f]["type"] == "select" 
            else FEATURE_CONFIGS[f]["default"] 
            for f in test_features
        }])
        
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
    """
    Load average feature values for diabetic population.
    
    Args:
        averages_path: Path to diabetic averages JSON file
    
    Returns:
        dict: Feature name to average value mapping
    """
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
