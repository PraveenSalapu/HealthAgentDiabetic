"""
Gemini Agent for generic health insights.

This agent uses Google's Gemini API to provide:
- General health advice
- Lifestyle recommendations
- Motivational support
- Wellness coaching
"""

from typing import Dict, List, Optional
import google.generativeai as genai

from agents.base_agent import BaseAgent
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, CHAT_MODEL_INFO, CHAT_MODEL_GEMINI


class GeminiAgent(BaseAgent):
    """Agent that provides generic health insights using Gemini API."""
    
    def __init__(self):
        """Initialize Gemini agent."""
        info = CHAT_MODEL_INFO[CHAT_MODEL_GEMINI]
        super().__init__(
            name=info["name"],
            description=info["description"]
        )
        self.model = None
        self.api_key = GEMINI_API_KEY
        self.model_name = GEMINI_MODEL
        self.capabilities = info["capabilities"]
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize Gemini API.
        
        Args:
            **kwargs: Optional api_key and model_name overrides
        
        Returns:
            bool: True if initialization successful
        """
        try:
            api_key = kwargs.get("api_key", self.api_key)
            model_name = kwargs.get("model_name", self.model_name)
            
            if not api_key:
                return False
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Gemini agent: {e}")
            self.is_initialized = False
            return False
    
    def generate_response(
        self, 
        message: str, 
        context: Dict,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response using Gemini API.
        
        Args:
            message: User message
            context: Prediction context with probability, risk_level, profile_summary
            conversation_history: Previous messages
        
        Returns:
            str: Gemini-generated response
        """
        if not self.is_initialized:
            return self._generate_fallback_response(message, context)
        
        try:
            # Build system prompt with context
            system_prompt = self._build_system_prompt(context)
            
            # Start chat with system prompt
            chat = self.model.start_chat(history=[])
            chat.send_message(system_prompt)
            
            # Send conversation history if available
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    if msg.get("role") == "user":
                        chat.send_message(msg["content"])
            
            # Send current message and get response
            response = chat.send_message(message)
            return response.text
        
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            return self._generate_fallback_response(message, context)
    
    def get_capabilities(self) -> List[str]:
        """Return list of Gemini agent capabilities."""
        return self.capabilities
    
    def _build_system_prompt(self, context: Dict) -> str:
        """
        Build system prompt with prediction context.
        
        Args:
            context: Prediction context
        
        Returns:
            str: System prompt
        """
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        profile_summary = context.get("profile_summary", "No data available")
        
        return f"""You are a compassionate health assistant helping users understand their diabetes risk assessment.

CONTEXT:
- Risk probability: {probability:.1f}% ({risk_level} risk)
- User's health metrics:
{profile_summary}

YOUR ROLE:
- Provide empathetic, non-alarmist guidance based on the risk assessment
- Suggest lifestyle modifications (diet, exercise, sleep, stress management)
- Recommend appropriate follow-up actions
- Answer health-related questions in accessible language
- NEVER provide definitive medical diagnoses or prescribe treatments

CRITICAL SAFETY RULES:
1. Always clarify this is a risk estimation tool, not a diagnosis
2. Use language like "the model estimates" or "based on these factors"
3. Include disclaimers about consulting healthcare providers
4. Encourage professional medical consultation for any health concerns
5. Avoid creating alarm - focus on actionable, positive steps
6. If asked about medications or treatments, defer to healthcare providers

CONVERSATION STYLE:
- Warm and supportive, but scientifically accurate
- Use simple language, avoid excessive medical jargon
- Provide specific, actionable suggestions
- Ask clarifying questions when helpful
- Acknowledge emotions and concerns
- Keep responses concise (under 200 words unless asked for detail)

Provide helpful, evidence-based guidance while maintaining appropriate boundaries."""
    
    def _generate_fallback_response(self, message: str, context: Dict) -> str:
        """
        Generate fallback response when API is unavailable.
        
        Args:
            message: User message
            context: Prediction context
        
        Returns:
            str: Fallback response
        """
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        
        return f"""I'm currently unable to connect to the AI service, but I can provide some general guidance.

Based on your {probability:.1f}% risk probability ({risk_level} risk):

**General Recommendations:**
- Maintain a balanced diet rich in vegetables, whole grains, and lean proteins
- Aim for at least 150 minutes of moderate physical activity per week
- Monitor your blood sugar levels regularly if advised by your doctor
- Get adequate sleep (7-9 hours per night)
- Manage stress through relaxation techniques
- Stay hydrated throughout the day

**Next Steps:**
- Schedule a check-up with your healthcare provider
- Discuss your risk assessment results with them
- Ask about appropriate screening tests (A1C, fasting glucose)
- Consider consulting a registered dietitian for personalized nutrition advice

Please note: This is general information only. Always consult with healthcare professionals for personalized medical advice.

You asked: "{message}"

For a more detailed response, please try again when the AI service is available."""
