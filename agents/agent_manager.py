"""
Agent Manager for orchestrating multi-agent chatbot system.

This module manages:
- Agent selection and switching
- Prediction context management
- Message routing to appropriate agents
- Conversation history per agent
"""

from typing import Dict, List, Optional
from agents.base_agent import BaseAgent
from agents.gemini_agent import GeminiAgent
from agents.rag_agent import RAGAgent
from config.settings import CHAT_MODEL_GEMINI, CHAT_MODEL_RAG
from utils.helpers import build_profile_summary, classify_risk


class AgentManager:
    """Manages multiple chatbot agents and routes conversations."""
    
    def __init__(self):
        """Initialize agent manager."""
        self.agents: Dict[str, BaseAgent] = {}
        self.active_agent_type: str = CHAT_MODEL_GEMINI
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {
            CHAT_MODEL_GEMINI: [],
            CHAT_MODEL_RAG: []
        }
        self.prediction_context: Dict = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all available agents."""
        # Initialize Gemini agent
        gemini_agent = GeminiAgent()
        if gemini_agent.initialize():
            self.agents[CHAT_MODEL_GEMINI] = gemini_agent
            print(f"✅ {gemini_agent.name} initialized")
        else:
            print(f"⚠️ {gemini_agent.name} initialization failed")
            self.agents[CHAT_MODEL_GEMINI] = gemini_agent  # Add anyway for fallback
        
        # Initialize RAG agent
        rag_agent = RAGAgent()
        if rag_agent.initialize():
            self.agents[CHAT_MODEL_RAG] = rag_agent
            print(f"✅ {rag_agent.name} initialized")
        else:
            print(f"⚠️ {rag_agent.name} initialization failed")
            self.agents[CHAT_MODEL_RAG] = rag_agent  # Add anyway for fallback
    
    def set_prediction_context(self, probability: float, user_data: Dict[str, float]):
        """
        Store prediction results for all agents.
        
        This context is automatically passed to agents when generating responses.
        
        Args:
            probability: Diabetes risk probability (0-100)
            user_data: User's health metrics
        """
        risk_level, _, _ = classify_risk(probability)
        profile_summary = build_profile_summary(user_data)
        
        self.prediction_context = {
            "probability": probability,
            "risk_level": risk_level,
            "user_data": user_data,
            "profile_summary": profile_summary
        }
    
    def send_message(self, message: str, agent_type: Optional[str] = None) -> str:
        """
        Route message to agent with prediction context.
        
        Args:
            message: User message
            agent_type: Agent type to use (defaults to active agent)
        
        Returns:
            str: Agent response
        """
        # Use specified agent or active agent
        agent_type = agent_type or self.active_agent_type
        
        # Get agent
        agent = self.agents.get(agent_type)
        if not agent:
            return f"Error: Agent '{agent_type}' not found."
        
        # Get conversation history for this agent
        history = self.conversation_histories.get(agent_type, [])
        
        # Generate response with context
        response = agent.generate_response(
            message=message,
            context=self.prediction_context,
            conversation_history=history
        )
        
        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        self.conversation_histories[agent_type] = history
        
        return response
    
    def switch_agent(self, new_agent_type: str) -> bool:
        """
        Switch active agent while preserving context.
        
        Args:
            new_agent_type: Type of agent to switch to
        
        Returns:
            bool: True if switch successful
        """
        if new_agent_type not in self.agents:
            return False
        
        self.active_agent_type = new_agent_type
        return True
    
    def get_active_agent(self) -> Optional[BaseAgent]:
        """
        Get currently active agent.
        
        Returns:
            BaseAgent: Active agent instance
        """
        return self.agents.get(self.active_agent_type)
    
    def get_agent_info(self, agent_type: str) -> Dict:
        """
        Get information about an agent.
        
        Args:
            agent_type: Agent type
        
        Returns:
            dict: Agent information (name, description, capabilities)
        """
        agent = self.agents.get(agent_type)
        if not agent:
            return {}
        
        return {
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.get_capabilities(),
            "is_ready": agent.is_ready()
        }
    
    def get_conversation_history(self, agent_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get conversation history for an agent.
        
        Args:
            agent_type: Agent type (defaults to active agent)
        
        Returns:
            list: Conversation history
        """
        agent_type = agent_type or self.active_agent_type
        return self.conversation_histories.get(agent_type, [])
    
    def clear_conversation_history(self, agent_type: Optional[str] = None):
        """
        Clear conversation history for an agent.
        
        Args:
            agent_type: Agent type (defaults to active agent, or 'all' for all agents)
        """
        if agent_type == "all":
            for key in self.conversation_histories:
                self.conversation_histories[key] = []
        else:
            agent_type = agent_type or self.active_agent_type
            self.conversation_histories[agent_type] = []
    
    def get_available_agents(self) -> List[str]:
        """
        Get list of available agent types.
        
        Returns:
            list: List of agent type identifiers
        """
        return list(self.agents.keys())
    
    def is_agent_ready(self, agent_type: Optional[str] = None) -> bool:
        """
        Check if an agent is ready to handle requests.
        
        Args:
            agent_type: Agent type (defaults to active agent)
        
        Returns:
            bool: True if agent is ready
        """
        agent_type = agent_type or self.active_agent_type
        agent = self.agents.get(agent_type)
        return agent.is_ready() if agent else False
