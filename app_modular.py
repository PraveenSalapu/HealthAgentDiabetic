"""
Health AI Chatbot - Modular Multi-Agent Application

A diabetes risk prediction app with multi-agent chatbot support:
- Gemini Agent: Generic health insights
- RAG Agent: Clinical insights from medical literature

Run with: streamlit run app_modular.py
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Modular imports
from config.settings import (
    MODEL_JSON_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
    AVERAGES_PATH,
    CHAT_MODEL_GEMINI,
    CHAT_MODEL_RAG,
)
from models import load_model_components, load_diabetic_averages, DiabetesPredictor
from agents import AgentManager
from ui.forms import render_prediction_form
from ui.visualizations import (
    create_risk_gauge,
    create_comparison_chart,
    create_contribution_waterfall,
    create_wellness_radar,
    generate_insights,
)
from ui.chat_interface import render_model_selector, render_chat_interface, render_agent_status


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'prediction_prob' not in st.session_state:
        st.session_state.prediction_prob = 0.0
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'agent_manager' not in st.session_state:
        st.session_state.agent_manager = None
    if 'diabetic_averages' not in st.session_state:
        st.session_state.diabetic_averages = {}
    if 'active_chat_model' not in st.session_state:
        st.session_state.active_chat_model = CHAT_MODEL_GEMINI


def load_models():
    """Load ML models and initialize agents."""
    if st.session_state.predictor is None:
        with st.spinner("Loading models..."):
            # Load ML model components
            model, preprocessor, threshold = load_model_components(
                MODEL_JSON_PATH,
                PREPROCESSOR_PATH,
                THRESHOLD_PATH
            )
            
            if model and preprocessor:
                st.session_state.predictor = DiabetesPredictor(model, preprocessor, threshold)
            
            # Load diabetic averages
            st.session_state.diabetic_averages = load_diabetic_averages(AVERAGES_PATH)
    
    # Initialize agent manager
    if st.session_state.agent_manager is None:
        with st.spinner("Initializing AI agents..."):
            st.session_state.agent_manager = AgentManager()


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Health AI Chatbot - Multi-Agent System",
        page_icon="🏥",
        layout="wide",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    load_models()
    
    # Sidebar
    st.sidebar.title("🏥 Health AI Chatbot")
    st.sidebar.markdown("### Multi-Agent System")
    
    # Model selector in sidebar
    if st.session_state.agent_manager:
        selected_model = render_model_selector(st.session_state.active_chat_model)
        
        # Switch agent if model changed
        if selected_model != st.session_state.active_chat_model:
            st.session_state.agent_manager.switch_agent(selected_model)
            st.session_state.active_chat_model = selected_model
            st.rerun()
        
        # Show agent status
        render_agent_status(st.session_state.agent_manager)
    
    # Main content
    st.title("🏥 Diabetes Risk Assessment & AI Health Assistant")
    st.markdown("### Powered by Multi-Agent AI System")
    
    # Check if models loaded
    if st.session_state.predictor is None:
        st.error("⚠️ Models failed to load. Please check model files.")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📋 Assessment", "📊 Results", "💬 AI Assistant"])
    
    with tab1:
        st.markdown("## Health Risk Assessment")
        st.markdown("Complete the form below to assess your diabetes risk.")
        
        # Render prediction form
        user_data = render_prediction_form()
        
        if user_data:
            # Make prediction
            with st.spinner("Analyzing your health data..."):
                probability, risk_level, badge_class, guidance = st.session_state.predictor.predict(user_data)
                
                # Store results
                st.session_state.prediction_made = True
                st.session_state.user_data = user_data
                st.session_state.prediction_prob = probability
                
                # Set prediction context for agents
                if st.session_state.agent_manager:
                    st.session_state.agent_manager.set_prediction_context(probability, user_data)
                
                st.success("✅ Assessment complete! View results in the Results tab.")
                st.rerun()
    
    with tab2:
        if not st.session_state.prediction_made:
            st.info("👈 Complete the assessment in the Assessment tab first.")
        else:
            st.markdown("## Your Risk Assessment Results")
            
            # Display risk gauge
            col1, col2 = st.columns([1, 1])
            with col1:
                fig_gauge = create_risk_gauge(st.session_state.prediction_prob)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.markdown(f"### Risk Level: {risk_level}")
                st.markdown(guidance)
            
            # Visualizations
            st.markdown("---")
            st.markdown("### Detailed Analysis")
            
            fig_comparison, differences = create_comparison_chart(
                st.session_state.user_data,
                st.session_state.diabetic_averages
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                fig_waterfall = create_contribution_waterfall(
                    st.session_state.user_data,
                    st.session_state.diabetic_averages
                )
                st.plotly_chart(fig_waterfall, use_container_width=True)
            
            with col4:
                fig_radar = create_wellness_radar(
                    st.session_state.user_data,
                    st.session_state.diabetic_averages
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Insights
            st.markdown("### 💡 Key Insights")
            insights = generate_insights(
                st.session_state.user_data,
                st.session_state.diabetic_averages,
                differences
            )
            for insight in insights:
                st.markdown(f"- {insight}")
    
    with tab3:
        if not st.session_state.prediction_made:
            st.info("👈 Complete the assessment first to enable the AI assistant.")
        elif not st.session_state.agent_manager:
            st.error("⚠️ AI agents failed to initialize.")
        else:
            # Render chat interface
            render_chat_interface(st.session_state.agent_manager)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.9em;">
        <p>🤖 Multi-Agent AI System: Gemini Agent (Generic Insights) + RAG Agent (Clinical Insights)</p>
        <p>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
