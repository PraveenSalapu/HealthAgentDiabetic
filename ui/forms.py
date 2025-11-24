"""
Form components for user input.

Note: Form rendering is temporarily handled in main app.
TODO: Extract form components here for better modularity.
"""

import streamlit as st
from typing import Dict
from config.settings import FEATURE_CONFIGS, FEATURE_NAMES, FEATURE_INFO, FORM_SECTIONS


def render_prediction_form() -> Dict[str, float]:
    """
    Render the complete prediction form.
    
    Returns:
        dict: User input data if form submitted, None otherwise
    """
    user_data = {}
    
    with st.form("prediction_form"):
        st.markdown("## 📋 Health Assessment")
        
        for section in FORM_SECTIONS:
            st.markdown(f"### {section['title']}")
            st.markdown(f"*{section['description']}*")
            
            cols = st.columns(2)
            for idx, feature in enumerate(section['features']):
                col = cols[idx % 2]
                config = FEATURE_CONFIGS[feature]
                display_name = FEATURE_NAMES[feature]
                help_text = config.get('help', FEATURE_INFO.get(feature, ''))
                
                with col:
                    if config['type'] == 'select':
                        # Create formatted options
                        options = config['options']
                        labels = config['labels']
                        formatted_options = [f"{labels[i]}" for i in range(len(options))]
                        
                        selected_label = st.selectbox(
                            display_name,
                            options=formatted_options,
                            key=feature,
                            help=help_text
                        )
                        # Map back to numeric value
                        selected_idx = formatted_options.index(selected_label)
                        user_data[feature] = options[selected_idx]
                    else:
                        user_data[feature] = st.number_input(
                            display_name,
                            min_value=config['min'],
                            max_value=config['max'],
                            value=config['default'],
                            step=config['step'],
                            key=feature,
                            help=help_text
                        )
            
            if section != FORM_SECTIONS[-1]:
                st.markdown("---")
        
        submitted = st.form_submit_button("🔍 Assess Risk", type="primary")
        
        if submitted:
            return user_data
    
    return None


__all__ = ['render_prediction_form']
