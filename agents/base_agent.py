"""
Base agent interface for the multi-agent chatbot system.

This module defines the abstract base class that all agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseAgent(ABC):
    """Abstract base class for all chatbot agents."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the agent with necessary resources.
        
        Args:
            **kwargs: Agent-specific initialization parameters
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def generate_response(
        self, 
        message: str, 
        context: Dict,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response to user message.
        
        Args:
            message: User message
            context: Prediction context (probability, risk_level, user_data, etc.)
            conversation_history: Previous conversation messages
        
        Returns:
            str: Agent response
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of agent capabilities.
        
        Returns:
            list: List of capability descriptions
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if agent is ready to handle requests.
        
        Returns:
            bool: True if agent is initialized and ready
        """
        return self.is_initialized
    
    def reset(self):
        """Reset agent state (optional, can be overridden)."""
        pass
