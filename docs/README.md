# MedGemma RAG — Documentation Index

Complete documentation for the CKD RAG system. The top-level
[`README.md`](../README.md) is the project overview; this folder contains
deeper guides.

---

## Start here

- **[Project README](../README.md)** — overview, install, quick-start, latest eval numbers.
- **[Architecture](architecture.md)** — system overview, all 3 RAG levels, deployment topology, Mermaid diagrams.
- **[Usage](usage.md)** — terminal demo (`main.py`) and Gradio UI walkthrough.

## Per-tier deep dives

- **[Simple RAG (Level 1)](modules/simple-rag.md)** — retrievers (flat / tree / RAPTOR / contextual), embeddings, vector store, RAG chain.
- **[Agentic RAG (Level 2)](modules/agentic-rag.md)** — LangGraph state machine, PII handling, intent routing, RAGAS hooks.
- **[Multi-Agent RAG (Level 3)](modules/multi-agent-rag.md)** — orchestrator, Diet / Medication / Lifestyle / RAG agents.

## Data & evaluation

- **[Data Pipeline](data-pipeline.md)** — PDF → OCR → cleaning → section split → block-aware chunking → ChromaDB.
- **[Evaluation Guide](evaluation.md)** — RAGAS concepts, run instructions, latest results, interpretation.

## Deployment

- **[Deployment overview](deployment/README.md)**
  - [EC2 workflow](deployment/ec2-workflow.md) — start / stop / sync session pattern.
  - [GPU spot strategy](deployment/gpu-spot-strategy.md) — instance choice and availability.
  - [Remote model server](deployment/remote-model-server.md) — vLLM + TEI on EC2.
  - [S3 model cache setup](deployment/s3-model-cache-setup.md) — pre-warming model weights.

## Working notes

- **[TODO](TODO.md)** — Fix / Update / Upgrade list discovered during the doc-vs-code audit on 2026-04-25.

---

## Where things live in the codebase

| Code path | Doc that covers it |
|-----------|--------------------|
| `simple_rag/` | [`modules/simple-rag.md`](modules/simple-rag.md) |
| `agentic_rag/` | [`modules/agentic-rag.md`](modules/agentic-rag.md) |
| `multi_agent_rag/` | [`modules/multi-agent-rag.md`](modules/multi-agent-rag.md) |
| `Data/`, `scripts/build_*.py`, `scripts/ocr-*.sh` | [`data-pipeline.md`](data-pipeline.md) |
| `eval/`, `agentic_rag/evaluation/` | [`evaluation.md`](evaluation.md) |
| `infrastructure/`, `scripts/start.sh`, `scripts/stop.sh`, `scripts/sync.sh`, `scripts/startup.sh` | [`deployment/`](deployment/) |
| `app.py`, `main.py` | [`usage.md`](usage.md) |
