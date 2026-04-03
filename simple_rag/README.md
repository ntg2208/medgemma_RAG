# Level 1: Simple Retrieval-Augmented Generation

Basic RAG pipeline for CKD question answering with document retrieval and source citations.

## Architecture

```
User Query → EmbeddingGemma → Vector Search → Context Retrieval → MedGemma → Response
                                    ↑
                              TreeRetriever
                         (section routing + chunk retrieval)
```

## Components

### embeddings.py - EmbeddingGemma Wrapper

LangChain-compatible wrapper for Google's EmbeddingGemma model.

- Matryoshka Representation Learning (MRL): dimensions 128, 256, 512, 768
- Batch processing with configurable batch size
- Optional in-memory caching (`CachedEmbeddingGemma`)
- Auto-detection: CUDA → MPS → CPU

```python
from simple_rag.embeddings import EmbeddingGemmaWrapper

embeddings = EmbeddingGemmaWrapper(dimension=768)
vector = embeddings.embed_query("What is chronic kidney disease?")
```

### vectorstore.py - ChromaDB Vector Store

Persistent vector storage with metadata filtering.

- Document insertion with metadata (source, section, chunk_id)
- Similarity search with relevance scores
- Score threshold filtering
- Collection statistics and management

```python
from simple_rag.vectorstore import CKDVectorStore

store = CKDVectorStore(embeddings)
store.add_documents(documents)
results = store.search("potassium restrictions", k=5)
```

### retriever.py - Document Retrievers

Two retriever implementations:

**CKDRetriever** (default for Agentic/Multi-Agent layers):
- Medical term expansion (CKD→chronic kidney disease, etc.)
- Configurable similarity thresholds
- LangChain `BaseRetriever` interface

**HybridRetriever** (experimental):
- Reciprocal rank fusion of semantic search results
- Configurable semantic vs keyword weights

```python
from simple_rag.retriever import CKDRetriever, create_retriever

retriever = CKDRetriever(vectorstore=store, k=5, score_threshold=0.3)
docs = retriever.invoke("dietary restrictions for CKD stage 3")

# Or use factory function:
retriever = create_retriever(store, use_tree=True, embedding_function=embeddings)
```

### tree_retriever.py - Tree-Based Section Routing

Two-phase retrieval that uses document structure for precision:

1. **Section routing**: Find relevant sections via heading similarity (small collection)
2. **Parent expansion**: Include parent sections for hierarchical context
3. **Chunk retrieval**: Fetch chunks only from matched sections
4. **Re-ranking**: Score-based deduplication and top-k selection

Falls back to flat similarity search when tree routing finds no matches.

```python
from simple_rag.tree_retriever import TreeRetriever

retriever = TreeRetriever(
    vectorstore=store,
    embedding_function=embeddings,
    k=5,
    section_k=8,
    chunks_per_section=3,
)
docs = retriever.invoke("ESA dosing for hemodialysis")
```

### chain.py - RAG Chain

Complete RAG pipeline with MedGemma.

- Local inference (MedGemma 4B, optional 4-bit quantization)
- Remote inference (vLLM via `config.get_llm()`)
- Patient context personalization (`personal_context.txt`)
- LangChain LCEL pipeline with streaming support
- Source citation formatting

```python
from simple_rag.chain import SimpleRAGChain
from config import get_llm

llm = get_llm()
chain = SimpleRAGChain(retriever=retriever, llm=llm)

response = chain.invoke("What foods should I avoid with CKD?")
print(response.answer)
print(response.source_documents)

# Streaming
for chunk in chain.stream("What is eGFR?"):
    print(chunk, end="", flush=True)
```

## Full Pipeline Example

```python
from config import get_llm, get_embeddings
from simple_rag.vectorstore import CKDVectorStore
from simple_rag.tree_retriever import TreeRetriever
from simple_rag.chain import SimpleRAGChain

# Initialize
embeddings = get_embeddings()
store = CKDVectorStore(embeddings)
llm = get_llm()

# Tree-based retriever (recommended)
retriever = TreeRetriever(vectorstore=store, embedding_function=embeddings)
chain = SimpleRAGChain(retriever=retriever, llm=llm)

# Query
response = chain.invoke("What is the target blood pressure for CKD patients?")
```

## Configuration

See `config.py` for settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `EMBEDDING_MODEL_ID` | EmbeddingGemma model | `google/embeddinggemma-300m` |
| `EMBEDDING_DIMENSION` | Vector dimension (MRL) | 768 |
| `MEDGEMMA_MODEL_ID` | MedGemma model | `google/medgemma-1.5-4b-it` |
| `CHUNK_SIZE` | Tokens per chunk | 800 |
| `CHUNK_OVERLAP` | Trailing blocks for overlap | 1 |
| `TOP_K_RESULTS` | Documents to retrieve | 5 |
| `SIMILARITY_THRESHOLD` | Minimum similarity | 0.3 |
| `SECTION_K` | Section headings to match | 8 |
| `CHUNKS_PER_SECTION` | Chunks per matched section | 3 |
