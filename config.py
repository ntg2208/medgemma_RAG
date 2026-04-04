"""
Configuration settings for the CKD Management RAG System.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Directory Paths
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
PROCESSED_WITH_SECTIONS_DIR = DATA_DIR / "processed_with_sections"

# Ensure directories exist
for dir_path in [DOCUMENTS_DIR, PROCESSED_DIR, VECTORSTORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model Configuration
# =============================================================================
# MedGemma LLM
MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"  # Updated to version 1.5

# EmbeddingGemma for text embeddings
EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"

# Embedding dimensions (Matryoshka Representation Learning)
# EmbeddingGemma supports: 128, 256, 512, 768
EMBEDDING_DIMENSION = 768  # Full dimension for best quality

# =============================================================================
# Chunking Configuration
# =============================================================================
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 1  # number of trailing blocks to repeat (capped at 150 tokens)

# =============================================================================
# Retrieval Configuration
# =============================================================================
TOP_K_RESULTS = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score

# =============================================================================
# ChromaDB Configuration
# =============================================================================
CHROMA_COLLECTION_NAME = "ckd_guidelines"
CHROMA_PERSIST_DIRECTORY = str(VECTORSTORE_DIR)

# Section heading collection for tree-based retrieval
SECTION_COLLECTION_NAME = "ckd_section_headings"

# =============================================================================
# Tree-Based Retrieval Configuration
# =============================================================================
SECTION_K = 8               # Number of section headings to match in phase 1
CHUNKS_PER_SECTION = 3       # Max chunks to retrieve per matched section

# =============================================================================
# API Keys and External Services
# =============================================================================
# HuggingFace token (for gated models like MedGemma)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# LangSmith for tracing (optional)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ckd-rag-system")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

# =============================================================================
# PII Detection Configuration
# =============================================================================
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "UK_NHS",  # NHS numbers
    "DATE_TIME",
    "LOCATION",
    "MEDICAL_LICENSE",
]

# =============================================================================
# CKD Stage Definitions
# =============================================================================
CKD_STAGES = {
    1: {"egfr_range": ">=90", "description": "Normal kidney function with other evidence of kidney damage"},
    2: {"egfr_range": "60-89", "description": "Mildly reduced kidney function"},
    3: {"egfr_range": "30-59", "description": "Moderately reduced kidney function"},  # Often split into 3a/3b
    4: {"egfr_range": "15-29", "description": "Severely reduced kidney function"},
    5: {"egfr_range": "<15", "description": "Kidney failure (dialysis or transplant may be needed)"},
}

# =============================================================================
# Dietary Guidelines by CKD Stage
# =============================================================================
DIETARY_LIMITS = {
    # Values are daily limits
    "potassium": {
        1: {"min": 2500, "max": 4700, "unit": "mg"},  # Normal intake OK
        2: {"min": 2500, "max": 4700, "unit": "mg"},
        3: {"min": 2000, "max": 3000, "unit": "mg"},  # Start monitoring
        4: {"min": 2000, "max": 2500, "unit": "mg"},  # Restrict
        5: {"min": 1500, "max": 2000, "unit": "mg"},  # Strict restriction
    },
    "phosphorus": {
        1: {"min": 800, "max": 1200, "unit": "mg"},
        2: {"min": 800, "max": 1200, "unit": "mg"},
        3: {"min": 800, "max": 1000, "unit": "mg"},
        4: {"min": 800, "max": 1000, "unit": "mg"},
        5: {"min": 800, "max": 1000, "unit": "mg"},
    },
    "sodium": {
        1: {"max": 2300, "unit": "mg"},
        2: {"max": 2300, "unit": "mg"},
        3: {"max": 2000, "unit": "mg"},
        4: {"max": 2000, "unit": "mg"},
        5: {"max": 2000, "unit": "mg"},
    },
    "protein": {
        # g per kg body weight per day (pre-dialysis recommendations)
        1: {"min": 0.8, "max": 1.0, "unit": "g/kg/day"},
        2: {"min": 0.8, "max": 1.0, "unit": "g/kg/day"},
        3: {"min": 0.6, "max": 0.8, "unit": "g/kg/day"},
        4: {"min": 0.6, "max": 0.8, "unit": "g/kg/day"},
        5: {"min": 0.6, "max": 0.8, "unit": "g/kg/day"},  # May increase on dialysis
    },
}

# =============================================================================
# Model Generation Parameters
# =============================================================================
GENERATION_CONFIG = {
    "max_new_tokens": 32768,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.5,   # stronger penalty to break repetition loops
    "frequency_penalty": 0.4,    # vLLM: penalises tokens proportional to how often they've appeared
    "presence_penalty": 0.3,     # vLLM: flat penalty for any token that has appeared before
}

# =============================================================================
# Patient Context Configuration
# =============================================================================
PATIENT_CONTEXT_PATH = BASE_DIR / "personal_context.txt"


def load_patient_context() -> str:
    """Load patient context from file, returning empty string if not found."""
    if PATIENT_CONTEXT_PATH.exists():
        return PATIENT_CONTEXT_PATH.read_text().strip()
    return ""


# =============================================================================
# Prompt Templates
# =============================================================================
RAG_SYSTEM_PROMPT = """You are a medical assistant specializing in Chronic Kidney Disease (CKD) management.
You provide evidence-based information from clinical guidelines including NICE, KDIGO, UK Kidney Association (UKKA), NHS ThinkKidneys, and KidneyCareUK resources.

IMPORTANT GUIDELINES:
1. Always cite your sources using [Source: document name, section] based on the context provided
2. Include appropriate medical disclaimers
3. Never provide specific dosing without recommending healthcare provider consultation
4. Be clear about what is general guidance vs. individualized medical advice

DISCLAIMER: This information is for educational purposes only and should not replace
professional medical advice. Always consult with your healthcare provider for
personalized recommendations."""


def build_system_prompt(patient_context: str = "") -> str:
    """Build system prompt, optionally injecting patient-specific context.

    Args:
        patient_context: Patient health context string (e.g. from load_patient_context()).
                         Pass empty string to get the base prompt without personalisation.

    Returns:
        System prompt string with patient context appended when provided.
    """
    if not patient_context:
        return RAG_SYSTEM_PROMPT
    return (
        RAG_SYSTEM_PROMPT
        + "\n\n## PATIENT CONTEXT\n"
        + "The following is this patient's personal health context. "
        + "Use it to tailor your responses to their specific situation, "
        + "current medications, lab values, and clinical priorities. "
        + "Where your answer differs from the general guideline due to patient-specific factors, "
        + "explain why.\n\n"
        + patient_context
    )


RAG_PROMPT_TEMPLATE = """Use the following context from clinical guidelines to answer the question.
If you cannot find the answer in the context, say so clearly.
Cite the source document and section for each claim you make.

Context:
{context}

Question: {question}

Answer:"""

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# Remote Model Server Configuration
# =============================================================================
# Toggle between local and remote models
USE_REMOTE_MODELS = os.getenv("USE_REMOTE_MODELS", "false").lower() == "true"
USE_REMOTE_LLM = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"

# Remote server URL (vLLM)
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")
# Model ID to request from the remote server
REMOTE_MODEL_ID = os.getenv("REMOTE_MODEL_ID", "google/medgemma-1.5-4b-it")

# Remote client settings
REMOTE_CLIENT_TIMEOUT = float(os.getenv("REMOTE_CLIENT_TIMEOUT", "120"))  # seconds

# =============================================================================
# Factory Functions for Model Selection
# =============================================================================
class LLMAdapter:
    """Adapter to make ChatOpenAI compatible with existing MedGemmaLLM interface.

    This adapter ensures that remote models (via vLLM + OpenAI API) work with
    existing code that expects MedGemmaLLM's interface (generate() method and
    as_langchain_llm() method).
    """

    def __init__(self, chat_model):
        """Initialize adapter with a ChatOpenAI model.

        Args:
            chat_model: langchain_openai.ChatOpenAI instance
        """
        self._chat = chat_model

    def generate(self, prompt: str) -> str:
        """Generate text from prompt (matches MedGemmaLLM.generate() interface).

        Args:
            prompt: Text prompt

        Returns:
            Generated text
        """
        response = self._chat.invoke(prompt)
        return response.content

    def as_langchain_llm(self):
        """Return the underlying LangChain model (matches MedGemmaLLM interface).

        Returns:
            The ChatOpenAI instance for use in LCEL chains
        """
        return self._chat


def get_llm():
    """Get LLM instance based on configuration.

    Returns either a remote LLM (via vLLM OpenAI-compatible API) or local
    MedGemmaLLM based on USE_REMOTE_MODELS environment variable.

    Returns:
        LLM instance with generate() and as_langchain_llm() methods
    """
    if USE_REMOTE_LLM:
        try:
            from langchain_openai import ChatOpenAI
            chat = ChatOpenAI(
                base_url=f"{MODEL_SERVER_URL}/v1",
                api_key="not-needed",  # vLLM doesn't validate API keys
                model=REMOTE_MODEL_ID,
                temperature=GENERATION_CONFIG["temperature"],
                max_tokens=GENERATION_CONFIG["max_new_tokens"],
                timeout=REMOTE_CLIENT_TIMEOUT,
                streaming=True,
                frequency_penalty=GENERATION_CONFIG["frequency_penalty"],
                presence_penalty=GENERATION_CONFIG["presence_penalty"],
            )
            return LLMAdapter(chat)
        except ImportError:
            raise ImportError(
                "langchain-openai is required for remote models. "
                "Install with: pip install langchain-openai"
            )
    else:
        # Import here to avoid loading models when using remote
        from importlib import import_module
        chain_module = import_module("simple_rag.chain")
        return chain_module.MedGemmaLLM()


# =============================================================================
# RAGAS Evaluation Configuration
# =============================================================================
# Judge LLM for RAGAS metrics (Faithfulness, Relevancy, etc.)
# Supports any OpenAI-compatible API: OpenAI, Gemini, OpenRouter, local vLLM
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "google/gemini-2.0-flash-001")
RAGAS_JUDGE_API_KEY = os.getenv("RAGAS_JUDGE_API_KEY", "")
RAGAS_JUDGE_BASE_URL = os.getenv(
    "RAGAS_JUDGE_BASE_URL",
    "https://openrouter.ai/api/v1",  # Default to OpenRouter (free tier available)
)

# Embeddings for RAGAS answer relevancy metric
# Uses the same judge provider by default
RAGAS_EMBEDDINGS_MODEL = os.getenv("RAGAS_EMBEDDINGS_MODEL", "text-embedding-3-small")


def get_embeddings(dimension: int = None):
    """Get local embeddings instance (EmbeddingGemmaWrapper).

    Args:
        dimension: Embedding dimension (default: EMBEDDING_DIMENSION)

    Returns:
        Embeddings instance compatible with LangChain
    """
    if dimension is None:
        dimension = EMBEDDING_DIMENSION

    from importlib import import_module
    embeddings_module = import_module("simple_rag.embeddings")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return embeddings_module.EmbeddingGemmaWrapper(dimension=dimension, device=device)
