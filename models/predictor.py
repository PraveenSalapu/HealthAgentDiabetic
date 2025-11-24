"""
Prediction logic for diabetes risk assessment.

This module handles:
- Input validation and sanitization
- Feature preprocessing
- Probability calculation
- Risk classification
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np

from utils.helpers import classify_risk, sanitize_input
from config.settings import FEATURE_CONFIGS


class DiabetesPredictor:
    """Handles diabetes risk prediction using trained XGBoost model."""
    
    def __init__(self, model, preprocessor, threshold: float = 0.5):
        """
        Initialize predictor with model components.
        
        Args:
            model: Trained XGBoost model
            preprocessor: Fitted preprocessing pipeline
            threshold: Optimal classification threshold
        """
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
    
    def validate_and_sanitize_inputs(self, user_data: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and sanitize user inputs.
        
        Args:
            user_data: Raw user input data
        
        Returns:
            dict: Sanitized user data
        """
        sanitized = {}
        for feature, value in user_data.items():
            if feature in FEATURE_CONFIGS:
                sanitized[feature] = sanitize_input(value, feature)
            else:
                sanitized[feature] = value
        return sanitized
    
    def predict(self, user_data: Dict[str, float]) -> Tuple[float, str, str, str]:
        """
        Make diabetes risk prediction.
        
        Args:
            user_data: User health metrics
        
        Returns:
            tuple: (probability, risk_level, badge_class, guidance_message)
        """
        # Sanitize inputs
        clean_data = self.validate_and_sanitize_inputs(user_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([clean_data])
        
        # Ensure correct feature order
        feature_order = list(FEATURE_CONFIGS.keys())
        df = df[feature_order]
        
        # Preprocess
        X_processed = self.preprocessor.transform(df)
        
        # Predict probability
        proba = self.model.predict_proba(X_processed)[0, 1]
        probability = float(proba * 100)
        
        # Classify risk
        risk_level, badge_class, guidance = classify_risk(probability)
        
        return probability, risk_level, badge_class, guidance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the model.
        
        Returns:
            dict: Feature name to importance score mapping
        """
        if hasattr(self.model, 'feature_importances_'):
            feature_names = list(FEATURE_CONFIGS.keys())
            importances = self.model.feature_importances_
            return dict(zip(feature_names, importances))
        return {}
