# Health AI Chatbot - Multi-Agent System

A modular diabetes risk prediction application with multi-agent chatbot support.

## 🎯 Features

### Multi-Agent Chatbot System
- **Gemini Agent**: Generic health insights, lifestyle recommendations, and motivational support
- **RAG Agent**: Clinical insights from medical literature with source citations
- **Seamless Switching**: Toggle between agents in the same UI
- **Context-Aware**: All agents receive prediction context automatically

### Diabetes Risk Assessment
- XGBoost-based prediction model
- Interactive visualizations
- Personalized health insights
- Comparison with diabetic population averages

## 📁 Project Structure

```
Final/
├── app_modular.py              # New modular main application
├── app2.py                     # Legacy application (for reference)
├── config/
│   ├── __init__.py
│   └── settings.py             # All configuration constants
├── models/
│   ├── __init__.py
│   ├── model_loader.py         # ML model loading
│   └── predictor.py            # Prediction logic
├── agents/
│   ├── __init__.py
│   ├── base_agent.py           # Abstract base agent
│   ├── gemini_agent.py         # Gemini API agent
│   ├── rag_agent.py            # RAG-based agent
│   └── agent_manager.py        # Agent orchestration
├── ui/
│   ├── __init__.py
│   ├── chat_interface.py       # Chat UI components
│   ├── forms.py                # Input forms
│   ├── visualizations.py       # Charts and graphs
│   └── styles.py               # CSS styling
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Utility functions
├── data/
│   └── clinical_docs/          # Medical documents for RAG
├── model_output2/              # Trained model files
└── requirements.txt            # Dependencies
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Final
```

### 2. Create virtual environment
```bash
python -m venv .venv
```

### 3. Activate virtual environment
**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🎮 Usage

### Run the modular application
```bash
streamlit run app_modular.py
```

### Run the legacy application
```bash
streamlit run app2.py
```

## 🤖 Multi-Agent System

### Gemini Agent
- **Purpose**: Generic health insights and lifestyle recommendations
- **Capabilities**:
  - General health advice
  - Lifestyle recommendations
  - Motivational support
  - Wellness coaching
- **Technology**: Google Gemini API

### RAG Agent
- **Purpose**: Clinical insights from medical literature
- **Capabilities**:
  - Evidence-based clinical information
  - Medical literature references
  - Research-backed recommendations
  - Source citations
- **Technology**: 
  - ChromaDB for vector storage
  - Sentence Transformers for embeddings
  - Gemini API for generation

### Agent Manager
- Orchestrates multiple agents
- Manages prediction context
- Routes conversations
- Maintains separate conversation histories

## 📊 How It Works

1. **Assessment**: User completes health assessment form
2. **Prediction**: XGBoost model predicts diabetes risk
3. **Context Injection**: Prediction results automatically passed to all agents
4. **Chat**: User can ask questions using either:
   - **Gemini Agent** for general health advice
   - **RAG Agent** for clinical insights with citations
5. **Model Switching**: Seamlessly switch between agents in the same conversation

## 🔧 Configuration

All configuration is centralized in `config/settings.py`:

- Model paths
- Feature configurations
- Agent definitions
- RAG settings (vector store, embeddings, chunk size)
- API keys

## 📚 Adding Clinical Documents

To enhance the RAG agent's knowledge base:

1. Add PDF or TXT files to `data/clinical_docs/`
2. The RAG agent will automatically index them on initialization
3. Documents are chunked and stored in ChromaDB
4. Agent retrieves relevant chunks when answering questions

## 🧪 Testing

### Test modular imports
```bash
python -c "from models import load_model_components; print('✓ Models module')"
python -c "from agents import GeminiAgent, RAGAgent; print('✓ Agents module')"
python -c "from ui import render_chat_interface; print('✓ UI module')"
```

### Test agents individually
```python
from agents import GeminiAgent, RAGAgent

# Test Gemini agent
gemini = GeminiAgent()
gemini.initialize(api_key="your_key")
response = gemini.generate_response("Tell me about diabetes prevention", {})

# Test RAG agent
rag = RAGAgent()
rag.initialize(api_key="your_key")
response = rag.generate_response("What does research say about exercise and diabetes?", {})
```

## 🔄 Migration from app2.py

The new modular architecture maintains backward compatibility:

- All existing functionality preserved
- Visualizations temporarily import from app2.py
- Gradual migration path available
- Both apps can run side-by-side

## 🛠️ Development

### Adding a new agent

1. Create new agent class in `agents/`:
```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def initialize(self, **kwargs):
        # Initialize your agent
        pass
    
    def generate_response(self, message, context, conversation_history):
        # Generate response
        pass
    
    def get_capabilities(self):
        return ["capability1", "capability2"]
```

2. Register in `agents/agent_manager.py`
3. Add configuration to `config/settings.py`
4. Update UI to include new agent option

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📧 Contact

[Your Contact Information]
