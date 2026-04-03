# RAPTOR + Contextual RAG Design Spec

**Date:** 2026-04-03

**Goal:** Add two new retrieval strategies — RAPTOR (recursive summarization tree) and Contextual RAG (Anthropic-style chunk enrichment) — as separate, independent modules alongside the existing CKDRetriever and TreeRetriever.

## Background

### Current Retrieval Strategies

| Strategy | How it works | Strength | Weakness |
|----------|-------------|----------|----------|
| **CKDRetriever** | Flat similarity search + medical term expansion | Fast, simple | No structure awareness |
| **TreeRetriever** | Section-route-then-chunk using document headings | Precise for structured docs | No cross-document themes, depends on heading quality |
| **HybridRetriever** | RRF over semantic results | Better than flat | Still no structure awareness |

### What's Missing

1. **Cross-document thematic retrieval** — "What do all guidelines say about potassium?" routes per-section, missing cross-document themes. RAPTOR's summaries capture these.
2. **Chunk context loss** — A chunk like "Limit intake to 2000mg/day" loses context about *what substance* and *which CKD stage*. Contextual RAG fixes this by prepending LLM-generated context.

### Corpus Stats

- 24 documents, 923 chunks total
- Well-structured clinical guidelines with numbered headings
- Metadata per chunk: `source`, `title`, `chunk_id`, `section`, `section_path`, `section_numbering`, `doc_name`

## Design

### Principle: Independent Modules

RAPTOR and Contextual RAG are **independent** of each other and of the existing retrievers. Each is:
- A separate module under `simple_rag/`
- Pluggable via the existing `create_retriever()` factory
- Tested independently
- Built at index time (one-time cost), zero additional query-time cost

### 1. RAPTOR Retriever

**Paper:** Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)

**Core idea:** Build a tree of summaries bottom-up by clustering chunks, summarizing each cluster with an LLM, then repeating recursively. At query time, search all levels (collapsed retrieval).

#### Architecture

```
Existing chunks (Layer 0)
        ↓
  UMAP + GMM soft clustering
        ↓
  LLM summarizes each cluster → Layer 1 summary nodes
        ↓
  Re-embed summaries → cluster again
        ↓
  Layer 2 summary nodes
        ↓
  ... until single cluster or max depth
        ↓
  All nodes (leaves + summaries) indexed in one ChromaDB collection
        ↓
  Query: flat top-k over all levels (collapsed retrieval)
```

#### Key Decisions

- **Collapsed retrieval** (not tree traversal) — simpler, and the paper found it slightly better on benchmarks.
- **Soft GMM clustering** — chunks can belong to multiple clusters (~probability threshold 0.1). This captures cross-topic information.
- **UMAP** for dimensionality reduction before GMM (768 → 10 dimensions, `n_neighbors=10`, `min_dist=0.0`, `metric='cosine'`).
- **BIC** to auto-select cluster count (range: 1 to ceil(sqrt(n_nodes))).
- **Max depth**: 3 levels. With 923 chunks, expect ~150-200 L1 summaries, ~30-50 L2, ~5-10 L3.
- **Separate ChromaDB collection**: `ckd_raptor` — keeps RAPTOR index isolated from the main `ckd_guidelines` collection.
- **LLM for summarization**: Use the project's existing `get_llm()` (MedGemma via vLLM or local). Prompt: concise summary preserving key medical details.

#### Files

| File | Purpose |
|------|---------|
| `simple_rag/raptor_builder.py` | Index-time: cluster + summarize + build tree |
| `simple_rag/raptor_retriever.py` | Query-time: collapsed retrieval over RAPTOR collection |
| `scripts/build_raptor_index.py` | CLI script to run the RAPTOR indexing pipeline |
| `tests/test_raptor_builder.py` | Unit tests for clustering and tree building |
| `tests/test_raptor_retriever.py` | Unit tests for retrieval |

#### `raptor_builder.py` — Public Interface

```python
class RaptorBuilder:
    def __init__(
        self,
        embedding_function: Embeddings,
        llm: Any,  # LLM with generate() method
        max_depth: int = 3,
        cluster_dim: int = 10,
        min_cluster_probability: float = 0.1,
        collection_name: str = "ckd_raptor",
    ): ...

    def build(self, chunks: list[Document]) -> RaptorTree:
        """Build the full RAPTOR tree from leaf chunks."""
        ...

    def index(self, tree: RaptorTree) -> None:
        """Index all nodes (leaves + summaries) into ChromaDB."""
        ...
```

```python
@dataclass
class RaptorNode:
    text: str
    embedding: list[float]  # cached embedding
    layer: int              # 0=leaf, 1+=summary
    children: list[str]     # IDs of child nodes
    metadata: dict          # source, section, etc. (leaves) or cluster info (summaries)

@dataclass
class RaptorTree:
    nodes: dict[str, RaptorNode]  # node_id -> node
    depth: int
```

#### `raptor_retriever.py` — Public Interface

```python
class RaptorRetriever(BaseRetriever):
    """Collapsed retrieval over RAPTOR tree (all levels)."""

    vectorstore: Any       # Chroma collection for RAPTOR
    embedding_function: Any
    k: int = 5
    score_threshold: float = 0.3

    def _get_relevant_documents(self, query, *, run_manager) -> list[Document]:
        """Flat top-k similarity search over all RAPTOR layers."""
        ...
```

#### Summarization Prompt

```
You are summarizing sections of clinical guidelines about Chronic Kidney Disease.
Write a concise summary of the following text passages, preserving:
- Specific recommendations and their evidence grades
- Numerical thresholds (eGFR values, dosages, lab ranges)
- Which CKD stages the guidance applies to
- Source document names

Text:
{concatenated_cluster_texts}

Summary:
```

### 2. Contextual RAG

**Source:** Anthropic blog, "Introducing Contextual Retrieval" (September 2024)

**Core idea:** Before embedding each chunk, use an LLM to generate a short context paragraph (50-100 tokens) that explains where the chunk sits in the document. Prepend this context to the chunk, then embed and index.

#### Architecture

```
For each (document, chunk) pair:
    LLM sees full document + chunk
        ↓
    Generates: "This chunk is from [section] of [document], discussing [topic]..."
        ↓
    Prepend context to chunk text
        ↓
    Embed contextualized chunk
        ↓
    Index in ChromaDB (separate collection: ckd_contextual)

At query time:
    Contextual Embeddings (semantic) → top-N candidates
    + Contextual BM25 (keyword) → top-N candidates
        ↓
    Merge + deduplicate
        ↓
    Rerank (optional, via cross-encoder or LLM)
        ↓
    Return top-K
```

#### Key Decisions

- **Separate ChromaDB collection**: `ckd_contextual` — keeps contextual index isolated.
- **BM25 index**: In-memory via `rank_bm25` library on the contextualized chunks. Serialized to disk as JSON for persistence.
- **Reranking**: Optional. Start without it; add as a future enhancement if needed.
- **LLM for context generation**: Use `get_llm()`. Each call sends the full document markdown + the specific chunk.
- **Cost estimate**: 923 chunks. Each call sends full doc (~2-5K tokens) + chunk (~800 tokens) + response (~100 tokens). With prompt caching across chunks from the same document, this is affordable.
- **Context stored in metadata**: The generated context is stored in the chunk's metadata field `contextual_context` so it can be inspected/debugged separately from the chunk content.

#### Files

| File | Purpose |
|------|---------|
| `simple_rag/contextual_builder.py` | Index-time: generate context per chunk, build BM25 index |
| `simple_rag/contextual_retriever.py` | Query-time: hybrid semantic + BM25 retrieval |
| `scripts/build_contextual_index.py` | CLI script to run the contextual enrichment pipeline |
| `tests/test_contextual_builder.py` | Unit tests for context generation |
| `tests/test_contextual_retriever.py` | Unit tests for hybrid retrieval |

#### `contextual_builder.py` — Public Interface

```python
class ContextualBuilder:
    def __init__(
        self,
        embedding_function: Embeddings,
        llm: Any,
        collection_name: str = "ckd_contextual",
        bm25_path: str = "Data/vectorstore/bm25_contextual.json",
    ): ...

    def generate_context(self, document_text: str, chunk: Document) -> str:
        """Generate contextual description for a single chunk."""
        ...

    def build(self, documents_dir: Path, chunks: list[Document]) -> None:
        """Generate context for all chunks and index them."""
        ...
```

#### `contextual_retriever.py` — Public Interface

```python
class ContextualRetriever(BaseRetriever):
    """Hybrid semantic + BM25 retrieval over contextualized chunks."""

    vectorstore: Any          # Chroma collection for contextual chunks
    embedding_function: Any
    bm25_index: Any           # BM25Okapi instance
    k: int = 5
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    score_threshold: float = 0.3

    def _get_relevant_documents(self, query, *, run_manager) -> list[Document]:
        """
        1. Semantic search over contextualized embeddings → top-N
        2. BM25 search over contextualized text → top-N
        3. Reciprocal rank fusion
        4. Return top-K
        """
        ...
```

#### Context Generation Prompt

```
<document>
{full_document_markdown}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

### 3. Integration with Existing System

#### Updated `create_retriever()` Factory

```python
def create_retriever(
    vectorstore,
    k=TOP_K_RESULTS,
    use_hybrid=False,
    use_tree=False,
    use_raptor=False,
    use_contextual=False,
    embedding_function=None,
) -> BaseRetriever:
```

New flags: `use_raptor` and `use_contextual`. Each creates the corresponding retriever pointed at its own ChromaDB collection.

#### Config Additions (`config.py`)

```python
# RAPTOR Configuration
RAPTOR_COLLECTION_NAME = "ckd_raptor"
RAPTOR_MAX_DEPTH = 3
RAPTOR_CLUSTER_DIM = 10
RAPTOR_MIN_CLUSTER_PROB = 0.1

# Contextual RAG Configuration
CONTEXTUAL_COLLECTION_NAME = "ckd_contextual"
CONTEXTUAL_BM25_PATH = str(VECTORSTORE_DIR / "bm25_contextual.json")
CONTEXTUAL_SEMANTIC_WEIGHT = 0.7
CONTEXTUAL_BM25_WEIGHT = 0.3
```

#### CLI (`main.py`) — New Retriever Flags

```bash
uv run python main.py simple --retriever tree       # existing (default)
uv run python main.py simple --retriever flat        # existing CKDRetriever
uv run python main.py simple --retriever raptor      # new
uv run python main.py simple --retriever contextual  # new
```

### 4. Dependencies

| Package | Purpose | Already Installed? |
|---------|---------|-------------------|
| `umap-learn` | Dimensionality reduction for RAPTOR clustering | No |
| `rank-bm25` | BM25 keyword search for Contextual RAG | No |
| `scikit-learn` | GMM clustering (GaussianMixture) | Yes (1.8.0) |
| `numpy` | Array operations | Yes (2.4.2) |

Only two new dependencies: `umap-learn` and `rank-bm25`.

### 5. What This Does NOT Include

- **Reranking** — Can be added later as a separate enhancement (cross-encoder or LLM-based).
- **Combining RAPTOR + Contextual** — They are independent strategies. A combined retriever could be built later.
- **UI changes** — The Gradio app already routes through the retriever; changing the retriever is a config change.
- **Evaluation** — The existing RAGAS evaluation pipeline will be used to compare all retriever strategies. No new evaluation code needed.

## Testing Strategy

- **Unit tests with mocked LLM**: Test clustering logic, tree building, context generation, and retrieval without calling real LLMs.
- **Integration test with small corpus**: 5-10 test chunks, verify end-to-end indexing and retrieval.
- **No tests that require GPU or real model inference** — all LLM calls mocked in tests.
