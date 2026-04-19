# `main.py` — Terminal Demo Guide

## What it Does

`main.py` is a colored terminal chat that lets you talk to all 3 RAG levels with full step-by-step logging. You can see exactly what each level does as it processes your query.

---

## Usage

```bash
uv run python main.py <level> [options]
```

| Argument | Values | Default |
|----------|--------|---------|
| `level` (required) | `simple`, `agentic`, `multi` | — |
| `--retriever` / `-r` | `flat`, `tree`, `raptor`, `contextual` | `tree` |
| `--show-context` / `-c` | flag | off |
| `--verbose` / `-v` | flag | off |

---

## Common Commands

```bash
# Level 1 — most basic, quickest to start
uv run python main.py simple --retriever flat

# Level 1 — show the retrieved chunks before the answer
uv run python main.py simple --retriever flat --show-context

# Level 2 — agentic with PII detection + query routing
uv run python main.py agentic --retriever flat

# Level 3 — multi-agent (diet/medication/lifestyle/RAG agents)
uv run python main.py multi --retriever flat

# Debug all logs
uv run python main.py simple -v
```

---

## Startup Sequence (what happens before you type anything)

```
CKD Management RAG — Terminal Chat
──────────────────────────────────────

  HH:MM:SS info  initializing components (retriever=flat)...
  HH:MM:SS embeddings  loading embedding model...    <- EmbeddingGemma 300M
  HH:MM:SS vectorstore loading ChromaDB...           <- opens Data/vectorstore/
  HH:MM:SS info  vectorstore has N documents          <- or auto-loads from Data/processed/
  HH:MM:SS retriever   initializing flat retriever...
  HH:MM:SS llm          loading LLM...               <- MedGemma 1.5 4B (local or remote)
  HH:MM:SS info  ready!
```

**If vectorstore is empty** it will auto-load JSON chunk files from `Data/processed/` and index them into ChromaDB. This takes a few minutes the first time.

---

## Level 1: Simple RAG (`simple`)

What it does per query:

1. Retrieves top-k documents from ChromaDB
2. Logs each retrieved doc (source file + section)
3. Optionally prints the chunk text if `--show-context`
4. Streams the MedGemma response token-by-token

**Thinking block handling** — MedGemma 1.5 produces `<unused94>thought\n...chain of thought...<unused95>` tokens. The streaming loop in `chat_simple()` (`main.py:211-270`) detects these tags, prints the thinking block in grey, then switches to printing the final answer normally. Three streaming states:
- `pre` → looking for the opening `<unused94>` tag
- `think` → inside the thinking block, print grey
- `answer` → after `<unused95>`, print normal

**Sample output:**
```
You: What potassium foods should a CKD stage 3 patient avoid?

  12:34:01 retriever    searching for top-k documents...
  12:34:01 info         retrieved 5 documents
  12:34:01   doc 1      kdigo_2024_chapters.pdf > 3.2 Potassium Management
  12:34:01   doc 2      nhs_ckd_diet.pdf > Dietary Advice
  12:34:01 llm           generating response (streaming)...

  ────────────────────────────
  Thinking:
  The user is asking about potassium restrictions for stage 3...
  ────────────────────────────
  Assistant: For CKD stage 3, you should limit high-potassium foods including...
```

---

## Level 2: Agentic RAG (`agentic`)

See `docs/agentic_rag_notes.md` for the full architecture. In the terminal you see each LangGraph node fire in sequence:

```
  12:34:01 pii_check         pii_check
  12:34:01 analyze_query     analyze_query
  12:34:01 info              intent -> retrieval
  12:34:01 retrieve_documents retrieve_documents
  12:34:01 info              retrieved 5 documents
  ...
  12:34:05 generate_response  generate_response
  12:34:06 evaluate          evaluate_response```

**Special input:** prefix with `stage:N` to pass a CKD stage:
```
You: stage:3 What foods should I avoid?
```

**Note:** The agentic mode does NOT stream token-by-token. It waits for each node to finish, then prints the full final response at the end. The node-by-node logs are what you see while it runs.

---

## Level 3: Multi-Agent (`multi`)

Shows the routing decision before agents are called:

```
  12:34:01 router   classifying query...
  12:34:01 info     route -> diet (confidence=90%)
  12:34:01 info     secondary agents: rag
  12:34:01 info     reasoning: query about food restrictions
  12:34:01 agents   processing query...
  12:34:06 info     agents used: diet, rag
  12:34:06 info     completed in 5.2s```

---

## `--show-context` Flag (Level 1 only)

Prints each retrieved chunk before the answer. For each chunk shows:
- Source PDF filename
- Section heading
- RAPTOR layer (if using raptor retriever)
- Contextual context snippet (if using contextual retriever)
- First 150 characters of chunk content

Useful for debugging retrieval quality without running full RAGAS.

---

## Initialization Details (`init_components`)

`main.py:102-150` — runs before any chat loop:

1. `get_embeddings()` — creates `EmbeddingGemmaWrapper` (from `config.py`, local or remote TEI)
2. `CKDVectorStore(embeddings)` — opens ChromaDB at `Data/vectorstore/`
3. If collection is empty → scans `Data/processed/*_chunks.json` and indexes all chunks
4. `create_retriever(...)` — builds the selected retriever type
5. `get_llm()` — loads MedGemma (local `transformers` 4-bit, or remote vLLM via `MODEL_SERVER_URL`)

The `import_package()` helper (`main.py:61-69`) handles loading the `simple_rag` package via `importlib` rather than a normal import — this was needed when directories had numeric prefixes.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `vectorstore is empty` on startup | `Data/processed/` has no `*_chunks.json` files | Run the data pipeline: `uv run python Data/export_chunks.py` |
| `initialization failed` | MedGemma can't load | Check `HF_TOKEN` in `.env`; or set `USE_REMOTE_MODELS=true` |
| Response cuts off mid-sentence | Streaming fallback triggered | Check the `log_warn: streaming failed` message; usually a transient vLLM issue |
| All queries route to `out_of_scope` in agentic | Query lacks CKD keywords | Include words like `kidney`, `ckd`, `egfr`, `potassium`, etc. |
| PII redaction fires unexpectedly | 10-digit numbers matching NHS pattern | Known false positive for plain `1234567890` numbers (0.6 confidence threshold) |
