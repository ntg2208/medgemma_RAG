# CKD Management RAG System

A 3-tier Retrieval-Augmented Generation (RAG) system for Chronic Kidney Disease (CKD) management, built for the **Kaggle MedGemma Impact Challenge**.

## Overview

This project provides an AI-powered assistant for CKD patients and healthcare providers, offering evidence-based information from:
- **NICE NG203 Guidelines** - CKD assessment and management
- **KidneyCareUK Resources** - Dietary and patient information
- **UK Kidney Association** - Clinical guidance

## Features

### Three Levels of RAG

| Level | Description | Key Features |
|-------|-------------|--------------|
| **Level 1: Simple RAG** | Basic retrieval and generation | Source citations, CKD stage filtering |
| **Level 2: Agentic RAG** | LangGraph workflow orchestration | PII detection, query routing, RAGAS evaluation |
| **Level 3: Multi-Agent** | Specialized domain agents | Diet, Medication, Lifestyle, Knowledge retrieval |

### Specialized Agents (Level 3)

- **RAG Agent**: General knowledge retrieval from guidelines
- **Diet Agent**: Personalized dietary recommendations (potassium, phosphorus, sodium, protein)
- **Medication Agent**: Kidney-safe medication guidance and drug interaction warnings
- **Lifestyle Agent**: Exercise, blood pressure, smoking, stress management guidance

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | MedGemma 1.5 4B (`google/medgemma-4b-it`) |
| Embeddings | EmbeddingGemma 300M (`google/embeddinggemma-300m`), MedEmbeddings (`google/medembeddings-300m`) |
| Vector Store | ChromaDB, Wavier |
| Framework | LangChain + LangGraph, Google ADK |
| PII Detection | Microsoft Presidio |
| Evaluation | RAGAS + Custom Metrics + LLM-as-a-Judge |
| Tracing | LangSmith |
| UI | Gradio |

## Project Structure

```
medgemma_RAG/
├── Data/
│   ├── documents/           # PDF source documents
│   ├── processed/           # Chunked documents
│   └── preprocessing.py     # PDF extraction and chunking
│
├── 1_Retrieval_Augmented_Generation/
│   ├── embeddings.py        # EmbeddingGemma wrapper
│   ├── vectorstore.py       # ChromaDB operations
│   ├── retriever.py         # Document retrieval
│   └── chain.py             # Simple RAG chain
│
├── 2_Agentic_RAG/
│   ├── pii_handler.py       # Presidio PII detection
│   ├── nodes.py             # LangGraph node functions
│   ├── graph.py             # Workflow definition
│   └── evaluation/          # RAGAS + custom metrics
│
├── 3_MultiAgent_RAG/
│   ├── orchestrator.py      # Query routing
│   └── agents/              # Specialized agents
│
├── app.py                   # Gradio UI
├── config.py                # Configuration
└── requirements.txt         # Dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or Apple Silicon
- HuggingFace account with access to MedGemma

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd medgemma_RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm  # For PII detection
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your HuggingFace token
   ```

5. **Add source documents**
   - Place PDF files in `Data/documents/`
   - Download NICE NG203 guidelines and KidneyCareUK resources

6. **Process documents**
   ```bash
   python -c "from Data.preprocessing import preprocess_documents; preprocess_documents()"
   ```

## Usage

### Launch the Gradio UI

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Programmatic Usage

```python
# Level 1: Simple RAG
from 1_Retrieval_Augmented_Generation.embeddings import EmbeddingGemmaWrapper
from 1_Retrieval_Augmented_Generation.vectorstore import CKDVectorStore
from 1_Retrieval_Augmented_Generation.chain import SimpleRAGChain, MedGemmaLLM
from 1_Retrieval_Augmented_Generation.retriever import CKDRetriever

embeddings = EmbeddingGemmaWrapper()
vectorstore = CKDVectorStore(embeddings)
llm = MedGemmaLLM()
retriever = CKDRetriever(vectorstore=vectorstore)
chain = SimpleRAGChain(retriever=retriever, llm=llm)

response = chain.invoke("What are the dietary restrictions for CKD stage 3?")
print(response.answer)

# Level 3: Multi-Agent
from 3_MultiAgent_RAG.orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator(retriever=retriever, llm=llm)
response = orchestrator.process(
    query="What foods should I avoid and is ibuprofen safe?",
    ckd_stage=3,
    weight_kg=70
)
print(f"Answer: {response.answer}")
print(f"Agents used: {response.agents_used}")
```

## Configuration

Key settings in `config.py`:

| Setting | Description | Default |
|---------|-------------|---------|
| `EMBEDDING_DIMENSION` | Embedding vector size | 768 |
| `CHUNK_SIZE` | Document chunk size (tokens) | 800 |
| `TOP_K_RESULTS` | Documents to retrieve | 5 |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | 0.7 |

## Evaluation

The system includes comprehensive evaluation:

### RAGAS Metrics
- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does it address the question?
- **Context Precision**: Are retrieved docs relevant?
- **Context Recall**: Are all relevant docs retrieved?

### Custom CKD Metrics
- Citation accuracy
- CKD stage appropriateness
- Medical disclaimer presence
- Actionability score

## Medical Disclaimer

**This tool is for educational purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Competition

Built for the [Kaggle MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

- **Deadline**: February 24, 2026
- **Prize Pool**: $100,000
- **Requirements**: 3-minute video, technical overview, reproducible code

## License

This project is for competition submission purposes. See LICENSE file for details.

## Acknowledgments

- Google Health AI for MedGemma
- NICE for clinical guidelines
- KidneyCareUK for patient resources
- LangChain and LangGraph teams
