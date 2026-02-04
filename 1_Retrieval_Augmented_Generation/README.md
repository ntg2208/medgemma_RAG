# Level 1: Simple Retrieval-Augmented Generation

Basic RAG pipeline for CKD question answering with document retrieval and source citations.

## Architecture

```
User Query → EmbeddingGemma → Vector Search → Context Retrieval → MedGemma → Response
```

## Components

### embeddings.py - EmbeddingGemma Wrapper

LangChain-compatible wrapper for Google's EmbeddingGemma model.

**Features:**
- Matryoshka Representation Learning (MRL) support
- Flexible dimensions: 128, 256, 512, 768
- Batch processing
- Optional caching

```python
from embeddings import EmbeddingGemmaWrapper

embeddings = EmbeddingGemmaWrapper(dimension=768)
vector = embeddings.embed_query("What is CKD stage 3?")
```

### vectorstore.py - ChromaDB Vector Store

Persistent vector storage with metadata filtering.

**Features:**
- Document insertion with metadata
- Similarity search with scores
- CKD stage filtering
- Document type filtering

```python
from vectorstore import CKDVectorStore

store = CKDVectorStore(embeddings)
store.add_documents(documents)
results = store.search("potassium restrictions", k=5)
```

### retriever.py - Document Retriever

Enhanced retrieval with medical term expansion.

**Features:**
- Query expansion for medical terms
- CKD stage-aware retrieval
- Configurable thresholds
- Hybrid retrieval option

```python
from retriever import CKDRetriever

retriever = CKDRetriever(vectorstore=store, ckd_stage=3)
docs = retriever.invoke("dietary restrictions")
```

### chain.py - RAG Chain

Complete RAG pipeline with MedGemma.

**Features:**
- 4-bit quantization support
- Custom medical prompts
- Source citation formatting
- Streaming support

```python
from chain import SimpleRAGChain, MedGemmaLLM

llm = MedGemmaLLM(load_in_4bit=True)
chain = SimpleRAGChain(retriever=retriever, llm=llm)

response = chain.invoke("What foods should I avoid with CKD?")
print(response.answer)
print(response.source_documents)
```

## Prompt Template

```
You are a medical assistant specializing in Chronic Kidney Disease (CKD) management.
Use the following context from NICE guidelines and KidneyCareUK to answer the question.
Always cite your sources and indicate the CKD stage relevance when applicable.

Context: {context}

Question: {question}

Answer:
```

## Usage

```python
# Full pipeline
from embeddings import EmbeddingGemmaWrapper
from vectorstore import CKDVectorStore
from retriever import CKDRetriever
from chain import SimpleRAGChain, MedGemmaLLM

# Initialize
embeddings = EmbeddingGemmaWrapper(dimension=768)
store = CKDVectorStore(embeddings)
llm = MedGemmaLLM()
retriever = CKDRetriever(vectorstore=store)
chain = SimpleRAGChain(retriever=retriever, llm=llm)

# Query
response = chain.invoke("What is the target blood pressure for CKD patients?")
```

## Configuration

See `config.py` for settings:
- `EMBEDDING_MODEL_ID`: EmbeddingGemma model
- `MEDGEMMA_MODEL_ID`: MedGemma model
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Chunking parameters
- `TOP_K_RESULTS`: Number of documents to retrieve
