"""
RAG Agent for clinical insights from medical literature.

This agent uses Retrieval-Augmented Generation to provide:
- Evidence-based clinical information
- Medical literature references
- Research-backed recommendations
- Source citations
"""

from typing import Dict, List, Optional
import os
from pathlib import Path

import google.generativeai as genai

from agents.base_agent import BaseAgent
from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CHAT_MODEL_INFO,
    CHAT_MODEL_RAG,
    RAG_VECTOR_STORE,
    RAG_EMBEDDING_MODEL,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    CLINICAL_DOCS_PATH,
    CHROMA_PERSIST_DIR,
)


class RAGAgent(BaseAgent):
    """Agent that provides clinical insights using RAG with medical literature."""
    
    def __init__(self):
        """Initialize RAG agent."""
        info = CHAT_MODEL_INFO[CHAT_MODEL_RAG]
        super().__init__(
            name=info["name"],
            description=info["description"]
        )
        self.model = None
        self.vectorstore = None
        self.embeddings = None
        self.api_key = GEMINI_API_KEY
        self.model_name = GEMINI_MODEL
        self.capabilities = info["capabilities"]
        self.docs_loaded = False
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize RAG agent with vector store and embeddings.
        
        Args:
            **kwargs: Optional configuration overrides
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize Gemini for generation
            api_key = kwargs.get("api_key", self.api_key)
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)
            
            # Initialize embeddings and vector store
            self._initialize_vector_store()
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing RAG agent: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_vector_store(self):
        """Initialize vector store with embeddings."""
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import Chroma
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.document_loaders import DirectoryLoader, TextLoader
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=RAG_EMBEDDING_MODEL
            )
            
            # Check if vector store already exists
            persist_dir = Path(CHROMA_PERSIST_DIR)
            if persist_dir.exists() and any(persist_dir.iterdir()):
                # Load existing vector store
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings
                )
                self.docs_loaded = True
                print(f"✅ Loaded existing vector store from {CHROMA_PERSIST_DIR}")
            else:
                # Create new vector store from documents
                docs_path = Path(CLINICAL_DOCS_PATH)
                if docs_path.exists() and any(docs_path.iterdir()):
                    self._load_and_index_documents()
                else:
                    # Create with sample diabetes information
                    self._create_sample_knowledge_base()
            
        except ImportError as e:
            print(f"Warning: RAG dependencies not installed: {e}")
            print("Install with: pip install langchain chromadb sentence-transformers")
            self.vectorstore = None
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vectorstore = None
    
    def _load_and_index_documents(self):
        """Load and index documents from clinical_docs directory."""
        from langchain.document_loaders import DirectoryLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import Chroma
        
        try:
            # Load documents
            loader = DirectoryLoader(
                CLINICAL_DOCS_PATH,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            self.vectorstore.persist()
            self.docs_loaded = True
            print(f"✅ Indexed {len(splits)} document chunks from {len(documents)} documents")
        except Exception as e:
            print(f"Error loading documents: {e}")
            self._create_sample_knowledge_base()
    
    def _create_sample_knowledge_base(self):
        """Create sample knowledge base with diabetes information."""
        from langchain.vectorstores import Chroma
        from langchain.schema import Document
        
        # Sample clinical information about diabetes
        sample_docs = [
            Document(
                page_content="""Diabetes Mellitus Overview:
Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels. 
Type 2 diabetes accounts for 90-95% of all diabetes cases and is strongly associated with 
obesity, physical inactivity, and genetic factors. Key risk factors include BMI >25, 
age >45, family history, and sedentary lifestyle.""",
                metadata={"source": "Clinical Guidelines", "topic": "Overview"}
            ),
            Document(
                page_content="""Diabetes Prevention Strategies:
The Diabetes Prevention Program (DPP) demonstrated that lifestyle interventions can reduce 
diabetes risk by 58%. Key interventions include: 1) Weight loss of 5-7% of body weight, 
2) At least 150 minutes of moderate physical activity per week, 3) Dietary modifications 
emphasizing whole grains, vegetables, and lean proteins, 4) Stress management and adequate sleep.""",
                metadata={"source": "DPP Study", "topic": "Prevention"}
            ),
            Document(
                page_content="""Diabetes Screening Recommendations:
The American Diabetes Association recommends screening for adults with BMI ≥25 and one or 
more risk factors, or all adults ≥45 years. Screening tests include: HbA1c ≥6.5%, fasting 
plasma glucose ≥126 mg/dL, or 2-hour plasma glucose ≥200 mg/dL during OGTT. Prediabetes is 
defined as HbA1c 5.7-6.4% or fasting glucose 100-125 mg/dL.""",
                metadata={"source": "ADA Guidelines", "topic": "Screening"}
            ),
            Document(
                page_content="""Nutrition for Diabetes Prevention:
Evidence-based dietary patterns for diabetes prevention include Mediterranean diet, DASH diet, 
and plant-based diets. Key principles: 1) Limit refined carbohydrates and added sugars, 
2) Increase fiber intake to 25-30g daily, 3) Choose healthy fats (olive oil, nuts, avocado), 
4) Control portion sizes, 5) Limit processed foods and red meat.""",
                metadata={"source": "Nutrition Guidelines", "topic": "Diet"}
            ),
            Document(
                page_content="""Physical Activity and Diabetes:
Regular physical activity improves insulin sensitivity and glucose metabolism. Recommendations: 
1) 150 minutes of moderate-intensity aerobic activity per week, 2) Resistance training 2-3 times 
per week, 3) Reduce sedentary time, 4) Include flexibility and balance exercises. Even small 
amounts of activity (10-minute walks after meals) can improve blood glucose control.""",
                metadata={"source": "Exercise Guidelines", "topic": "Physical Activity"}
            ),
        ]
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=sample_docs,
                embedding=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            self.vectorstore.persist()
            self.docs_loaded = True
            print("✅ Created sample knowledge base with diabetes information")
        except Exception as e:
            print(f"Error creating sample knowledge base: {e}")
            self.vectorstore = None
    
    def generate_response(
        self, 
        message: str, 
        context: Dict,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response using RAG.
        
        Args:
            message: User message
            context: Prediction context
            conversation_history: Previous messages
        
        Returns:
            str: RAG-generated response with citations
        """
        if not self.is_initialized or not self.vectorstore:
            return self._generate_fallback_response(message, context)
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vectorstore.similarity_search(message, k=RAG_TOP_K)
            
            # Build context from retrieved documents
            retrieved_context = self._format_retrieved_docs(relevant_docs)
            
            # Build prompt with retrieved context and prediction data
            prompt = self._build_rag_prompt(message, context, retrieved_context)
            
            # Generate response using Gemini
            if self.model:
                response = self.model.generate_content(prompt)
                return response.text
            else:
                return self._generate_fallback_response(message, context)
        
        except Exception as e:
            print(f"Error generating RAG response: {e}")
            return self._generate_fallback_response(message, context)
    
    def get_capabilities(self) -> List[str]:
        """Return list of RAG agent capabilities."""
        return self.capabilities
    
    def _format_retrieved_docs(self, docs) -> str:
        """Format retrieved documents for prompt."""
        if not docs:
            return "No relevant clinical information found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            formatted.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(formatted)
    
    def _build_rag_prompt(self, message: str, context: Dict, retrieved_context: str) -> str:
        """Build prompt with retrieved context and prediction data."""
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        profile_summary = context.get("profile_summary", "No data available")
        
        return f"""You are a clinical advisor providing evidence-based diabetes information.

PATIENT CONTEXT:
- Risk probability: {probability:.1f}% ({risk_level} risk)
- Health metrics:
{profile_summary}

RELEVANT CLINICAL INFORMATION:
{retrieved_context}

USER QUESTION:
{message}

INSTRUCTIONS:
1. Answer the question using the clinical information provided above
2. Reference specific sources when making recommendations
3. Relate the information to the patient's risk level when appropriate
4. Provide actionable, evidence-based guidance
5. Always include appropriate medical disclaimers
6. If the retrieved information doesn't fully answer the question, acknowledge limitations
7. Keep response concise but comprehensive (200-300 words)

CRITICAL: This is educational information only. Always recommend consulting healthcare providers for personalized medical advice.

Provide your response:"""
    
    def _generate_fallback_response(self, message: str, context: Dict) -> str:
        """Generate fallback response when RAG is unavailable."""
        probability = context.get("probability", 0)
        risk_level = context.get("risk_level", "Unknown")
        
        return f"""I'm currently unable to access the clinical knowledge base, but I can provide general evidence-based information.

Based on your {probability:.1f}% risk probability ({risk_level} risk):

**Evidence-Based Recommendations:**

1. **Lifestyle Modifications** (Diabetes Prevention Program):
   - Target 5-7% weight loss if overweight
   - 150 minutes moderate physical activity weekly
   - Dietary focus on whole grains, vegetables, lean proteins

2. **Screening** (ADA Guidelines):
   - Discuss HbA1c and fasting glucose testing with your doctor
   - Prediabetes range: HbA1c 5.7-6.4%
   - Regular monitoring if risk factors present

3. **Nutrition** (Evidence-Based):
   - Mediterranean or DASH diet patterns
   - Increase fiber to 25-30g daily
   - Limit refined carbohydrates and added sugars

4. **Physical Activity**:
   - Combination of aerobic and resistance training
   - Even 10-minute walks after meals help glucose control
   - Reduce sedentary time

**Your Question:** "{message}"

For detailed, personalized clinical guidance with specific literature references, please consult with your healthcare provider or try again when the knowledge base is available.

**Disclaimer:** This is general educational information based on clinical guidelines. Always consult healthcare professionals for personalized medical advice."""
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add new documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not self.vectorstore:
            print("Vector store not initialized")
            return
        
        try:
            from langchain.schema import Document
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Create Document objects
            docs = [
                Document(page_content=text, metadata=metadata[i] if metadata else {})
                for i, text in enumerate(documents)
            ]
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE,
                chunk_overlap=RAG_CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)
            
            # Add to vector store
            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()
            print(f"✅ Added {len(splits)} document chunks to knowledge base")
        except Exception as e:
            print(f"Error adding documents: {e}")
