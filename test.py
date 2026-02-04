"""
Test file for MedGemma RAG System
=================================
Import all modules for separate testing.
Run specific tests by uncommenting the relevant sections.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================
from config import (
    # Paths
    BASE_DIR,
    DATA_DIR,
    DOCUMENTS_DIR,
    PROCESSED_DIR,
    VECTORSTORE_DIR,
    # Model settings
    MEDGEMMA_MODEL_ID,
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIMENSION,
    # Chunking
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    # Retrieval
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    # ChromaDB
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    # API Keys
    HF_TOKEN,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    # PII
    PII_ENTITIES,
    # CKD
    CKD_STAGES,
    DIETARY_LIMITS,
    # Generation
    GENERATION_CONFIG,
    # Prompts
    RAG_SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    # Logging
    LOG_LEVEL,
    LOG_FORMAT,
)

# =============================================================================
# Data Processing Module
# =============================================================================
from Data import DocumentPreprocessor
from Data.preprocessing import DocumentPreprocessor as PreprocessorDirect

# =============================================================================
# Level 1: Simple RAG
# =============================================================================
from importlib import import_module

# Import from package
try:
    RAG_L1 = import_module("1_Retrieval_Augmented_Generation")
    EmbeddingGemmaWrapper = RAG_L1.EmbeddingGemmaWrapper
    CKDVectorStore = RAG_L1.CKDVectorStore
    CKDRetriever = RAG_L1.CKDRetriever
    SimpleRAGChain = RAG_L1.SimpleRAGChain
    print("[OK] Level 1 RAG modules imported")
except ImportError as e:
    print(f"[WARN] Level 1 RAG import error: {e}")
    EmbeddingGemmaWrapper = None
    CKDVectorStore = None
    CKDRetriever = None
    SimpleRAGChain = None

# =============================================================================
# Level 2: Agentic RAG
# =============================================================================
try:
    RAG_L2 = import_module("2_Agentic_RAG")
    PIIHandler = RAG_L2.PIIHandler
    AgenticRAGGraph = RAG_L2.AgenticRAGGraph
    RAGNodes = RAG_L2.RAGNodes
    print("[OK] Level 2 Agentic RAG modules imported")
except ImportError as e:
    print(f"[WARN] Level 2 Agentic RAG import error: {e}")
    PIIHandler = None
    AgenticRAGGraph = None
    RAGNodes = None

# Level 2: Evaluation submodule
try:
    RAG_L2_Eval = import_module("2_Agentic_RAG.evaluation")
    RAGASEvaluator = RAG_L2_Eval.RAGASEvaluator
    CKDMetrics = RAG_L2_Eval.CKDMetrics
    setup_langsmith = RAG_L2_Eval.setup_langsmith
    print("[OK] Level 2 Evaluation modules imported")
except ImportError as e:
    print(f"[WARN] Level 2 Evaluation import error: {e}")
    RAGASEvaluator = None
    CKDMetrics = None
    setup_langsmith = None

# =============================================================================
# Level 3: Multi-Agent RAG
# =============================================================================
try:
    RAG_L3 = import_module("3_MultiAgent_RAG")
    MultiAgentOrchestrator = RAG_L3.MultiAgentOrchestrator
    print("[OK] Level 3 Multi-Agent RAG orchestrator imported")
except ImportError as e:
    print(f"[WARN] Level 3 Multi-Agent RAG import error: {e}")
    MultiAgentOrchestrator = None

# Level 3: Individual Agents
try:
    RAG_L3_Agents = import_module("3_MultiAgent_RAG.agents")
    RAGAgent = RAG_L3_Agents.RAGAgent
    DietAgent = RAG_L3_Agents.DietAgent
    MedicationAgent = RAG_L3_Agents.MedicationAgent
    LifestyleAgent = RAG_L3_Agents.LifestyleAgent
    print("[OK] Level 3 Specialized agents imported")
except ImportError as e:
    print(f"[WARN] Level 3 Agents import error: {e}")
    RAGAgent = None
    DietAgent = None
    MedicationAgent = None
    LifestyleAgent = None


# =============================================================================
# Test Functions
# =============================================================================

def test_config():
    """Test configuration is loaded correctly."""
    print("\n" + "=" * 60)
    print("CONFIGURATION TEST")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Documents Directory: {DOCUMENTS_DIR}")
    print(f"MedGemma Model: {MEDGEMMA_MODEL_ID}")
    print(f"Embedding Model: {EMBEDDING_MODEL_ID}")
    print(f"Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Top-K Results: {TOP_K_RESULTS}")
    print(f"HF Token Set: {'Yes' if HF_TOKEN else 'No'}")
    print(f"LangSmith Tracing: {LANGSMITH_TRACING}")
    print(f"CKD Stages Defined: {len(CKD_STAGES)}")
    return True


def test_data_preprocessing():
    """Test document preprocessing module."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING TEST")
    print("=" * 60)
    if DocumentPreprocessor:
        preprocessor = DocumentPreprocessor()
        print(f"DocumentPreprocessor initialized: {preprocessor}")
        return True
    else:
        print("DocumentPreprocessor not available")
        return False


def test_level1_rag():
    """Test Level 1 Simple RAG components."""
    print("\n" + "=" * 60)
    print("LEVEL 1: SIMPLE RAG TEST")
    print("=" * 60)

    components = {
        "EmbeddingGemmaWrapper": EmbeddingGemmaWrapper,
        "CKDVectorStore": CKDVectorStore,
        "CKDRetriever": CKDRetriever,
        "SimpleRAGChain": SimpleRAGChain,
    }

    all_ok = True
    for name, component in components.items():
        if component:
            print(f"[OK] {name} available")
        else:
            print(f"[MISSING] {name}")
            all_ok = False

    return all_ok


def test_level2_agentic():
    """Test Level 2 Agentic RAG components."""
    print("\n" + "=" * 60)
    print("LEVEL 2: AGENTIC RAG TEST")
    print("=" * 60)

    components = {
        "PIIHandler": PIIHandler,
        "AgenticRAGGraph": AgenticRAGGraph,
        "RAGNodes": RAGNodes,
        "RAGASEvaluator": RAGASEvaluator,
        "CKDMetrics": CKDMetrics,
        "setup_langsmith": setup_langsmith,
    }

    all_ok = True
    for name, component in components.items():
        if component:
            print(f"[OK] {name} available")
        else:
            print(f"[MISSING] {name}")
            all_ok = False

    return all_ok


def test_level3_multiagent():
    """Test Level 3 Multi-Agent RAG components."""
    print("\n" + "=" * 60)
    print("LEVEL 3: MULTI-AGENT RAG TEST")
    print("=" * 60)

    components = {
        "MultiAgentOrchestrator": MultiAgentOrchestrator,
        "RAGAgent": RAGAgent,
        "DietAgent": DietAgent,
        "MedicationAgent": MedicationAgent,
        "LifestyleAgent": LifestyleAgent,
    }

    all_ok = True
    for name, component in components.items():
        if component:
            print(f"[OK] {name} available")
        else:
            print(f"[MISSING] {name}")
            all_ok = False

    return all_ok


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "=" * 60)
    print("MEDGEMMA RAG SYSTEM - MODULE TEST SUITE")
    print("=" * 60)

    results = {
        "Config": test_config(),
        "Data Preprocessing": test_data_preprocessing(),
        "Level 1 RAG": test_level1_rag(),
        "Level 2 Agentic": test_level2_agentic(),
        "Level 3 Multi-Agent": test_level3_multiagent(),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("MedGemma RAG System - Test Module")
    print("-" * 40)

    # Run all tests by default
    if len(sys.argv) == 1:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    # Run specific test
    test_name = sys.argv[1].lower()
    test_map = {
        "config": test_config,
        "data": test_data_preprocessing,
        "l1": test_level1_rag,
        "level1": test_level1_rag,
        "l2": test_level2_agentic,
        "level2": test_level2_agentic,
        "l3": test_level3_multiagent,
        "level3": test_level3_multiagent,
        "all": run_all_tests,
    }

    if test_name in test_map:
        test_map[test_name]()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_map.keys())}")
        sys.exit(1)
