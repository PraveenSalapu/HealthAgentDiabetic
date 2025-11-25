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
import pandas as pd
from config.settings import (
    MODEL_JSON_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
    AVERAGES_PATH,
    CHAT_MODEL_GEMINI,
    FEATURE_NAMES,
    CHAT_MODEL_INFO,
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
from ui.chat_interface import render_model_selector, render_agent_status
from utils.helpers import classify_risk, build_profile_summary


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
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False


def reset_app():
    """Reset application state for new assessment."""
    st.session_state.prediction_made = False
    st.session_state.user_data = {}
    st.session_state.prediction_prob = 0.0
    st.session_state.chatbot_initialized = False

    # Clear agent conversation histories
    if st.session_state.agent_manager:
        st.session_state.agent_manager.clear_conversation_history("all")


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
        initial_sidebar_state="expanded"
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
    
    # Hero Header CSS
    st.markdown("""
    <style>
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
            background: linear-gradient(135deg, #ffffff, #00d9ff);
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
            color: #00d9ff;
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
            color: #cbd5e1;
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
            color: #00d9ff;
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
            border-color: #00d9ff;
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
            color: #94a3b8;
            font-weight: 600;
        }

        .hero-metric-value {
            font-size: 3.5rem;
            font-weight: 800;
            color: #f1f5f9;
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
            color: #cbd5e1;
            line-height: 1.6;
        }

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
            color: #00d9ff;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.2);
        }
    </style>
    """, unsafe_allow_html=True)

    # Check if models loaded
    if st.session_state.predictor is None:
        st.error("⚠️ Models failed to load. Please check model files.")
        return

    # Hero Header
    prediction_made = st.session_state.prediction_made

    if prediction_made:
        probability = st.session_state.prediction_prob
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
                <span class="hero-pill">AI-Powered Multi-Agent System</span>
                <h1>Diabetes Risk Navigator</h1>
                <p class="hero-subtitle">
                    Understand your health indicators, benchmark against diabetic population data,
                    and receive personalized guidance from specialized AI healthcare agents.
                </p>
                <div class="hero-steps">
                    <div class="hero-step">1. Complete Assessment</div>
                    <div class="hero-step">2. Review Analysis</div>
                    <div class="hero-step">3. Chat with AI Agents</div>
                </div>
            </div>
            <div class="hero-highlight">
                <div class="hero-metric-label">Current Risk Estimate</div>
                <div class="hero-metric-value">{}</div>
                <div class="hero-metric-badge {}">{}</div>
                <p class="hero-highlight-note">{}</p>
            </div>
        </div>
        """.format(hero_metric, risk_badge_class, hero_badge_label, risk_message),
        unsafe_allow_html=True,
    )

    # Add reset button if prediction was made
    if st.session_state.prediction_made:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col3:
            if st.button("🔄 Start New Assessment", type="secondary", use_container_width=True):
                reset_app()
                st.rerun()

    st.markdown("")

    # ============ FLOW-BASED UI: Form → Analysis & Visualizations → AI Assistant ============

    if not st.session_state.prediction_made:
        # ============ STEP 1: ASSESSMENT FORM ============
        st.markdown("## 📋 Health Risk Assessment")
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

                # Reset chatbot initialization to trigger auto-welcome
                st.session_state.chatbot_initialized = False

                st.success("✅ Assessment complete! Scroll down to see your results and chat with AI agents.")
                st.rerun()
    else:
        # ============ STEP 2 & 3: ANALYSIS + AI ASSISTANT (Side-by-side) ============
        risk_level, badge_class, guidance = classify_risk(st.session_state.prediction_prob)

        st.success("✅ Assessment complete! Your personalized analysis and AI assistant are ready.")

        with st.expander("Review your submitted profile", expanded=False):
            summary_df = pd.DataFrame([st.session_state.user_data])
            summary_df.rename(columns=FEATURE_NAMES, inplace=True)
            st.dataframe(summary_df)

        # Create two-column layout: Analysis (left) + Chatbot (right)
        results_col, assistant_col = st.columns([7, 5], gap="large")

        with results_col:
            # ============ ANALYSIS & VISUALIZATIONS ============
            st.markdown("## Your Risk Assessment Results")

            # Key Health Metrics Cards (3 columns)
            st.markdown("### Key Health Indicators")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                bmi_value = st.session_state.user_data.get('BMI', 0)
                bmi_delta = bmi_value - st.session_state.diabetic_averages.get('BMI', 28.0)
                st.metric(
                    label="Body Mass Index (BMI)",
                    value=f"{bmi_value:.1f}",
                    delta=f"{bmi_delta:+.1f} vs avg",
                    delta_color="inverse"
                )

            with metric_col2:
                age_value = st.session_state.user_data.get('Age', 0)
                age_labels = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
                             6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
                             11: "70-74", 12: "75-79", 13: "80+"}
                st.metric(
                    label="Age Group",
                    value=age_labels.get(int(age_value), "N/A"),
                    delta=None
                )

            with metric_col3:
                phys_health = st.session_state.user_data.get('PhysHlth', 0)
                st.metric(
                    label="Physical Health (poor days/month)",
                    value=f"{int(phys_health)} days",
                    delta=None
                )

            with metric_col4:
                phys_activity = st.session_state.user_data.get('PhysActivity', 0)
                st.metric(
                    label="Physical Activity",
                    value="Active" if phys_activity == 1 else "Inactive",
                    delta="Good" if phys_activity == 1 else "Needs Improvement",
                    delta_color="normal" if phys_activity == 1 else "inverse"
                )

            # Risk Factors Summary
            st.markdown("### ⚠️ Risk Factors Present")
            risk_factors = []
            if st.session_state.user_data.get('HighBP', 0) == 1:
                risk_factors.append("🔴 High Blood Pressure")
            if st.session_state.user_data.get('HighChol', 0) == 1:
                risk_factors.append("🔴 High Cholesterol")
            if st.session_state.user_data.get('HeartDiseaseorAttack', 0) == 1:
                risk_factors.append("🔴 Heart Disease/Attack History")
            if st.session_state.user_data.get('DiffWalk', 0) == 1:
                risk_factors.append("🟡 Difficulty Walking")
            if st.session_state.user_data.get('PhysActivity', 0) == 0:
                risk_factors.append("🟡 No Physical Activity")
            if st.session_state.user_data.get('BMI', 0) >= 30:
                risk_factors.append("🟡 BMI ≥ 30 (Obese)")
            elif st.session_state.user_data.get('BMI', 0) >= 25:
                risk_factors.append("🟡 BMI ≥ 25 (Overweight)")

            if risk_factors:
                risk_factor_cols = st.columns(min(len(risk_factors), 3))
                for idx, factor in enumerate(risk_factors[:6]):  # Limit to 6
                    with risk_factor_cols[idx % 3]:
                        st.info(factor)
            else:
                st.success("✅ No major risk factors detected!")

            st.markdown("")

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

            # Comparison Chart (full width)
            fig_comparison, differences = create_comparison_chart(
                st.session_state.user_data,
                st.session_state.diabetic_averages
            )
            st.plotly_chart(fig_comparison, use_container_width=True, config={'displayModeBar': False})

            # Two charts side by side with better sizing
            viz_col1, viz_col2 = st.columns(2, gap="large")

            with viz_col1:
                fig_waterfall = create_contribution_waterfall(
                    st.session_state.user_data,
                    st.session_state.diabetic_averages
                )
                st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})

            with viz_col2:
                fig_radar = create_wellness_radar(
                    st.session_state.user_data,
                    st.session_state.diabetic_averages
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

            # Insights
            st.markdown("### 💡 Key Insights")
            insights = generate_insights(
                st.session_state.user_data,
                st.session_state.diabetic_averages,
                differences
            )
            for insight in insights:
                st.markdown(f"- {insight}")

        with assistant_col:
            # ============ AI HEALTH ASSISTANT WITH MULTI-AGENT SUPPORT ============
            if not st.session_state.agent_manager:
                st.error("⚠️ AI agents failed to initialize.")
            else:
                st.markdown("## 🤖 Healthcare AI Agents")

                # Agent selector dropdown (in main area, not just sidebar)
                agent_options = st.session_state.agent_manager.get_available_agents()
                agent_labels = [f"{CHAT_MODEL_INFO[k]['icon']} {CHAT_MODEL_INFO[k]['name']}" for k in agent_options]

                current_index = agent_options.index(st.session_state.active_chat_model)
                selected_agent_index = st.selectbox(
                    "Select Healthcare Agent",
                    options=range(len(agent_options)),
                    format_func=lambda i: agent_labels[i],
                    index=current_index,
                    key="agent_selector_main"
                )
                selected_agent = agent_options[selected_agent_index]

                # Switch agent if changed
                if selected_agent != st.session_state.active_chat_model:
                    st.session_state.agent_manager.switch_agent(selected_agent)
                    st.session_state.active_chat_model = selected_agent
                    # Reset chatbot initialization to trigger new welcome message
                    st.session_state.chatbot_initialized = False
                    st.rerun()

                # Show agent description
                agent_info = st.session_state.agent_manager.get_agent_info(selected_agent)
                st.caption(f"**{agent_info.get('description', '')}**")

                # Build context for agent
                risk_level_current, _, _ = classify_risk(st.session_state.prediction_prob)
                context = {
                    "probability": st.session_state.prediction_prob,
                    "risk_level": risk_level_current,
                    "profile_summary": build_profile_summary(st.session_state.user_data),
                    "user_data": st.session_state.user_data
                }

                # Auto-initialize chatbot with welcome message
                if not st.session_state.chatbot_initialized:
                    # Create agent-specific initial prompts (these are hidden from the user)
                    if selected_agent == CHAT_MODEL_GEMINI:
                        intro_message = f"""Please welcome me and provide a personalized overview of my diabetes risk assessment.

My risk probability is {st.session_state.prediction_prob:.1f}% ({risk_level_current} risk).

Review my health profile and provide:
1. A brief interpretation of my risk level
2. The top 3 most important factors contributing to my risk
3. Three specific actionable steps I should take this week

Keep it conversational, supportive, and personalized to MY specific numbers. Start with a greeting like "I've reviewed your diabetes risk assessment..."."""
                    else:  # RAG Agent
                        intro_message = f"""Welcome! I'd like you to review my diabetes risk profile and provide evidence-based clinical insights.

My Assessment:
- Risk Probability: {st.session_state.prediction_prob:.1f}% ({risk_level_current} risk)
- Health Profile:
{context.get('profile_summary', 'Profile data not available')}

Please provide:
1. A clinical interpretation of my risk level with relevant research context
2. Evidence-based insights about my key risk factors
3. Specific clinical recommendations backed by guidelines or studies
4. What screening tests or consultations I should prioritize

Use your clinical knowledge base to give me research-backed guidance specific to my profile. Start by greeting me and saying you've reviewed my assessment."""

                    with st.spinner("🤖 Analyzing your health profile and preparing personalized insights..."):
                        welcome_response = st.session_state.agent_manager.send_message(intro_message)

                    # Remove the programmatic user message from both histories
                    # We only want to show the AI's welcome response
                    if st.session_state.agent_manager.conversation_histories[selected_agent]:
                        # Remove the last user message (the programmatic intro_message)
                        history = st.session_state.agent_manager.conversation_histories[selected_agent]
                        if len(history) >= 2 and history[-2].get('role') == 'user':
                            # Keep only the assistant's response
                            st.session_state.agent_manager.conversation_histories[selected_agent] = [history[-1]]

                    st.session_state.chatbot_initialized = True
                    st.rerun()

                # Display conversation history
                st.markdown("### 💬 Chat")
                conversation = st.session_state.agent_manager.get_conversation_history()

                # Create scrollable chat container
                chat_container = st.container(height=500)
                with chat_container:
                    if conversation:
                        for msg in conversation:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")

                            if role == "user":
                                with st.chat_message("user"):
                                    st.markdown(content)
                            else:
                                agent_icon = CHAT_MODEL_INFO[st.session_state.active_chat_model]['icon']
                                with st.chat_message("assistant", avatar=agent_icon):
                                    st.markdown(content)

                # Chat input
                st.markdown("""
                <div style="background: rgba(0, 217, 255, 0.1);
                           padding: 12px;
                           margin: 8px 0;
                           border-radius: 8px;
                           border-left: 3px solid #00d9ff;">
                    💡 <strong>Try asking:</strong><br>
                    • What lifestyle changes would have the biggest impact?<br>
                    • Can you explain my risk factors?<br>
                    • What should I discuss with my doctor?
                </div>
                """, unsafe_allow_html=True)

                user_message = st.chat_input("Ask about your health assessment...")
                if user_message:
                    # Send message to agent
                    with st.spinner("Thinking..."):
                        response = st.session_state.agent_manager.send_message(user_message)
                    st.rerun()
    
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
