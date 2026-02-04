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

# Ensure directories exist
for dir_path in [DOCUMENTS_DIR, PROCESSED_DIR, VECTORSTORE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model Configuration
# =============================================================================
# MedGemma LLM
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"

# EmbeddingGemma for text embeddings
EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"

# Embedding dimensions (Matryoshka Representation Learning)
# EmbeddingGemma supports: 128, 256, 512, 768
EMBEDDING_DIMENSION = 768  # Full dimension for best quality

# =============================================================================
# Chunking Configuration
# =============================================================================
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 150  # tokens
MIN_CHUNK_SIZE = 100  # minimum chunk size to keep

# =============================================================================
# Retrieval Configuration
# =============================================================================
TOP_K_RESULTS = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score

# =============================================================================
# ChromaDB Configuration
# =============================================================================
CHROMA_COLLECTION_NAME = "ckd_guidelines"
CHROMA_PERSIST_DIRECTORY = str(VECTORSTORE_DIR)

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
    "max_new_tokens": 1024,
    "temperature": 0.3,  # Lower for more factual responses
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# =============================================================================
# Prompt Templates
# =============================================================================
RAG_SYSTEM_PROMPT = """You are a medical assistant specializing in Chronic Kidney Disease (CKD) management.
You provide evidence-based information from NICE guidelines and KidneyCareUK resources.

IMPORTANT GUIDELINES:
1. Always cite your sources using [Source: document name, section/page]
2. Indicate CKD stage relevance when applicable
3. Include appropriate medical disclaimers
4. Never provide specific dosing without recommending healthcare provider consultation
5. Be clear about what is general guidance vs. individualized medical advice

DISCLAIMER: This information is for educational purposes only and should not replace
professional medical advice. Always consult with your healthcare provider for
personalized recommendations."""

RAG_PROMPT_TEMPLATE = """Use the following context from NICE guidelines and KidneyCareUK to answer the question.
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
