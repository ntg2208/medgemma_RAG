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

Four retriever implementations, selectable through `create_retriever(...)`:

**CKDRetriever** (default for Agentic/Multi-Agent layers):
- Medical term expansion (CKD→chronic kidney disease, etc.)
- Word-boundary matching to avoid substring false positives
- Configurable similarity thresholds
- LangChain `BaseRetriever` interface

**HybridRetriever** (experimental):
- Reciprocal rank fusion of semantic search results
- Configurable semantic vs keyword weights

**RaptorRetriever** (see `raptor_retriever.py`): flat top-k search over a
separate ChromaDB collection containing both leaf chunks and LLM-generated
multi-layer cluster summaries.

**ContextualRetriever** (see `contextual_retriever.py`): hybrid semantic + BM25
retrieval over chunks that have been prepended with an LLM-generated
situating context.

```python
from simple_rag.retriever import CKDRetriever, create_retriever

retriever = CKDRetriever(vectorstore=store, k=5, score_threshold=0.3)
docs = retriever.invoke("dietary restrictions for CKD stage 3")

# Or use the factory function — pick exactly one strategy flag:
retriever = create_retriever(store, use_tree=True, embedding_function=embeddings)
retriever = create_retriever(store, use_raptor=True, embedding_function=embeddings)
retriever = create_retriever(store, use_contextual=True, embedding_function=embeddings)
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

### raptor_builder.py / raptor_retriever.py - RAPTOR

Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al.,
ICLR 2024). Build time:

1. Embed leaf chunks.
2. UMAP-reduce embeddings, then GMM-cluster with soft (multi-cluster) assignment
   and BIC-selected cluster count.
3. Summarise each cluster with the LLM → a new layer of nodes.
4. Recurse until `RAPTOR_MAX_DEPTH` or the layer collapses to a single node.

Query time: collapsed retrieval — a single flat top-k search over all layers
(leaves + summaries) in the `ckd_raptor` collection. High-level queries tend to
match summary nodes; specific queries tend to match leaves.

```python
from simple_rag.raptor_builder import RaptorBuilder
from simple_rag.raptor_retriever import create_raptor_retriever

# Build (one-off)
builder = RaptorBuilder(embedding_function=embeddings, llm=llm)
tree = builder.build(chunks)

# Query
retriever = create_raptor_retriever(embedding_function=embeddings, k=5)
docs = retriever.invoke("What drives CKD progression?")
```

Build via CLI: `uv run python scripts/build_raptor_index.py [--max-depth 3]`.

Visualise the tree (Pyvis HTML, optionally highlight a query's top-k hits):

```bash
uv run python scripts/visualize_raptor.py --output raptor_tree.html
uv run python scripts/visualize_raptor.py --query "potassium limits CKD"
```

### contextual_builder.py / contextual_retriever.py - Contextual RAG

Implements Anthropic's Contextual Retrieval technique. Build time:

1. For each chunk, the LLM generates a short paragraph explaining where the
   chunk sits inside the full document.
2. The context is prepended to the chunk body before embedding and BM25
   tokenisation.

Query time: hybrid retrieval with reciprocal rank fusion (RRF) combining
semantic similarity over contextualised embeddings and BM25 keyword matching
(`CONTEXTUAL_SEMANTIC_WEIGHT`, `CONTEXTUAL_BM25_WEIGHT`).

```python
from simple_rag.contextual_builder import ContextualBuilder
from simple_rag.contextual_retriever import create_contextual_retriever

builder = ContextualBuilder(embedding_function=embeddings, llm=llm)
ctx_chunks = builder.build(chunks, document_texts)  # {source_name: full_text}

retriever = create_contextual_retriever(embedding_function=embeddings, k=5)
docs = retriever.invoke("high potassium foods")
```

Build via CLI: `uv run python scripts/build_contextual_index.py`.

> **Cost note.** The context prompt includes the *entire source document* for
> every chunk (see `CONTEXT_PROMPT` in `contextual_builder.py`). Anthropic's
> original technique relies on prompt caching; a local vLLM run has no such
> cache, so expect build time to scale as O(chunks × doc_tokens).

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
| `RAPTOR_COLLECTION_NAME` | ChromaDB collection for RAPTOR nodes | `ckd_raptor` |
| `RAPTOR_MAX_DEPTH` | Maximum tree depth | 3 |
| `RAPTOR_CLUSTER_DIM` | UMAP target dimensions | 10 |
| `RAPTOR_MIN_CLUSTER_PROB` | Min GMM probability for soft assignment | 0.1 |
| `CONTEXTUAL_COLLECTION_NAME` | ChromaDB collection for contextualised chunks | `ckd_contextual` |
| `CONTEXTUAL_BM25_PATH` | BM25 index persistence file | `Data/vectorstore/bm25_contextual.json` |
| `CONTEXTUAL_SEMANTIC_WEIGHT` | RRF weight for semantic hits | 0.7 |
| `CONTEXTUAL_BM25_WEIGHT` | RRF weight for BM25 hits | 0.3 |
