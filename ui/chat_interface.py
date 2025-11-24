"""
Chat interface components for multi-agent chatbot.

This module provides:
- Model selector UI
- Chat message display
- Chat input interface
"""

import streamlit as st
from typing import Dict, List, Optional
from config.settings import CHAT_MODEL_INFO, CHAT_MODEL_GEMINI, CHAT_MODEL_RAG


def render_model_selector(current_model: str) -> str:
    """
    Render chat model selector UI.
    
    Args:
        current_model: Currently active model type
    
    Returns:
        str: Selected model type
    """
    st.sidebar.markdown("### 🤖 Chat Model")
    
    # Create radio button for model selection
    model_options = {
        CHAT_MODEL_GEMINI: f"{CHAT_MODEL_INFO[CHAT_MODEL_GEMINI]['icon']} {CHAT_MODEL_INFO[CHAT_MODEL_GEMINI]['name']}",
        CHAT_MODEL_RAG: f"{CHAT_MODEL_INFO[CHAT_MODEL_RAG]['icon']} {CHAT_MODEL_INFO[CHAT_MODEL_RAG]['name']}"
    }
    
    selected = st.sidebar.radio(
        "Select chat model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0 if current_model == CHAT_MODEL_GEMINI else 1,
        key="model_selector"
    )
    
    # Show model description and capabilities
    model_info = CHAT_MODEL_INFO[selected]
    st.sidebar.markdown(f"**{model_info['description']}**")
    
    with st.sidebar.expander("ℹ️ Capabilities"):
        for capability in model_info['capabilities']:
            st.markdown(f"• {capability}")
    
    return selected


def render_chat_interface(
    agent_manager,
    show_input: bool = True
):
    """
    Render chat interface with message history and input.
    
    Args:
        agent_manager: AgentManager instance
        show_input: Whether to show chat input
    """
    st.markdown("### 💬 Health Assistant Chat")
    
    # Get active agent info
    active_agent = agent_manager.get_active_agent()
    if active_agent:
        agent_info = agent_manager.get_agent_info(agent_manager.active_agent_type)
        st.markdown(f"**Active:** {agent_info['name']} {CHAT_MODEL_INFO[agent_manager.active_agent_type]['icon']}")
    
    # Display conversation history
    conversation = agent_manager.get_conversation_history()
    
    if conversation:
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                st.markdown(f"""
                <div style="background: rgba(0, 217, 255, 0.1); 
                            border-left: 3px solid #00d9ff; 
                            padding: 12px; 
                            margin: 8px 0; 
                            border-radius: 8px;">
                    <strong>You:</strong><br>{content}
                </div>
                """, unsafe_allow_html=True)
            else:
                agent_icon = CHAT_MODEL_INFO[agent_manager.active_agent_type]['icon']
                st.markdown(f"""
                <div style="background: rgba(124, 58, 237, 0.1); 
                            border-left: 3px solid #7c3aed; 
                            padding: 12px; 
                            margin: 8px 0; 
                            border-radius: 8px;">
                    <strong>{agent_icon} Assistant:</strong><br>{content}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👋 Start a conversation! Ask me about your diabetes risk assessment, lifestyle recommendations, or any health-related questions.")
    
    # Chat input
    if show_input:
        with st.form(key="chat_form", clear_on_submit=True):
            user_message = st.text_area(
                "Your message:",
                placeholder="Ask a question about your health assessment...",
                height=100,
                key="chat_input"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.form_submit_button("Send", type="primary")
            with col2:
                clear = st.form_submit_button("Clear History")
            
            if submit and user_message.strip():
                # Send message to agent
                with st.spinner(f"Thinking..."):
                    response = agent_manager.send_message(user_message)
                st.rerun()
            
            if clear:
                agent_manager.clear_conversation_history()
                st.rerun()


def render_agent_status(agent_manager):
    """
    Render agent status indicators.
    
    Args:
        agent_manager: AgentManager instance
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Agent Status")
    
    for agent_type in agent_manager.get_available_agents():
        is_ready = agent_manager.is_agent_ready(agent_type)
        info = agent_manager.get_agent_info(agent_type)
        icon = CHAT_MODEL_INFO[agent_type]['icon']
        status_icon = "✅" if is_ready else "⚠️"
        
        st.sidebar.markdown(f"{status_icon} {icon} **{info['name']}**")
