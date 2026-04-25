# Data Pipeline: PDF to Vector Store

## Overview

```
PDF (22 clinical guidelines)
 |
 |  Docling v2.72.0 (OCR + table recognition)
 v
processed_ocr/          *.md + *.json per document
 |
 |  split_sections.py   (LLM classification + heuristic fallback)
 v
processed_with_sections/
 |  {Title}/
 |    ├── main_text.md   clinical content only
 |    ├── references.md  bibliography, citations, end matter
 |    └── metadata.json  title, source, summary, processing stats
 |
 |  Manual cleaning      (OCR artifact fixes, described below)
 |
 |  preprocessing.py     (block-aware chunking)
 v
LangChain Documents     in-memory, with section metadata
 |
 |  EmbeddingGemma 300M  (768-dim vectors)
 v
ChromaDB                Data/vectorstore/, collection "ckd_guidelines"
```

---

## Stage 1: OCR with Docling

**Tool:** IBM Docling v2.72.0
**Scripts:** `scripts/ocr-process.sh` (batch), `scripts/ocr-single.py` (single file)
**Input:** `Data/documents/` (22 PDFs, ~28MB)
**Output:** `Data/processed_ocr/` (MD + JSON per document, ~35MB)

Docling settings:
- `do_ocr = True` for scanned pages
- `do_table_structure = True` with `TableFormerMode.ACCURATE`
- Exports both markdown (human-readable) and JSON (structured document tree with hierarchy, pictures, tables)

---

## Stage 2: Section Splitting

**Script:** `Data/split_sections.py` (486 lines)
**Input:** `Data/processed_ocr/*.md`
**Output:** `Data/processed_with_sections/{Title}/`

### Classification (Hybrid Approach)

1. **LLM-based (primary):** MedGemma classifies each `## heading` into:
   - `front_matter` -- authors, TOC, conflicts of interest, acknowledgments
   - `main_content` -- clinical guidelines, recommendations, appendices
   - `references` -- bibliography, citation lists
   - `end_matter` -- abbreviations, copyright

2. **Heuristic fallback:** Regex patterns detect boundaries when LLM is unavailable:
   - Front matter: "Contents", "Table of Contents", "Conflicts of Interest", "Acknowledgements"
   - References: "References", "Bibliography", "Further reading"
   - End matter: "Abbreviations", "Finding More Information" (only if >=70% through doc)

### Output per document

| File | Content |
|------|---------|
| `main_text.md` | Clinical content only (stripped of front matter, references, page noise) |
| `references.md` | All references, inline citations, end matter |
| `metadata.json` | Title, source file, summary, front matter text, processing stats |

### Review Status

Tracked in `Data/processed_with_sections/review_summary.md`. Key flags:
- **Pass** -- reviewed and verified
- **Check** -- needs manual review (e.g. KDIGO IgAN appendix order, UKKA Commentary trailing references)
- **Delete** -- superseded document (KDIGO 2024 replaced by 2025)
- Potential duplicates flagged by matching line counts

---

## Stage 3: Manual Cleaning

OCR artifacts fixed after section splitting:

### `- o ` sub-bullet replacement

Docling sometimes renders sub-bullets as `- o text` instead of `  - text`. Fixed in:

| File (processed_ocr/) | Occurrences |
|------------------------|-------------|
| Commentary-on-the-NICE-Guideline-on-RRT-and-conservative-management.md | 6 |
| Commentary-on-the-NICE-Guideline-on-RRT-and-conservative-management_cleaned.md | 6 |
| KDIGO_2025_Anemia_Guideline_Draft.md | 6 |
| Clinical Practice Guideline Exercise and Lifestyle in Chronic Kidney Disease.md | 8 |
| UKKA_Hyperkalaemia_Management_Guideline.md | 2 (table cell) |

In `processed_with_sections/`, only the Hyperkalaemia table cell needed fixing (other files were already clean after section splitting).

### Reference extraction (UKKA Commentary NICE Hypertension)

References 1-17 were embedded in the main file with OCR page-number noise between refs 7 and 8. Moved to a separate `_references.md` file in `processed_ocr/`.

In `processed_with_sections/`, references were already in a separate `references.md`.

### Other OCR artifacts present but not yet fixed

- `/uniFB01` and `/uniFB02` unicode escapes for ligatures (fi, fl) throughout KDIGO files
- `/C15` bullet markers in KDIGO IgAN methods section
- Orphaned page numbers between sections (e.g. standalone `1`, `2`, ... `30` lines)
- Temperature notation `2 o C` (degrees) in Hyperkalaemia tables
- Broken words from OCR: `o n` (on), `o utcomes` (outcomes) in Notts CKD guidelines

---

## Stage 4: Block-Aware Chunking

**Script:** `Data/preprocessing.py` (371 lines), class `DocumentPreprocessor`
**Input:** `Data/processed_with_sections/{Title}/main_text.md`

### Block Parsing

Markdown is split into atomic blocks at blank lines:

| Block Type | Detection | Rule |
|------------|-----------|------|
| Heading | Lines starting with `##` | Always a single-line block |
| Table | Contiguous lines with `\|` | Grouped together, never split |
| List | Lines matching `^\d+\.` or `^- ` or `^* ` | Grouped with continuations and indentation |
| Paragraph | Everything else | Contiguous non-blank lines |

Each block tracks its nearest parent heading for section metadata.

### Chunk Packing (Greedy Bin Packing)

- **Chunk size:** `CHUNK_SIZE = 2000` tokens (configurable in `config.py`)
- **Token estimation:** `len(text) // 4` (~4 chars per token)
- **Overlap:** `CHUNK_OVERLAP = 200` trailing blocks from previous chunk, capped at 150 tokens via `overlap_token_cap` in `Data/preprocessing.py:153` (so the cap, not the block count, is the binding constraint in practice). The comment in `config.py:43` is currently stale — see [`docs/TODO.md`](TODO.md).
- **Atomic rule:** Blocks are never split across chunks
- **Heading binding:** Heading + following block are paired to prevent orphaned headings

### Chunk Metadata

Each chunk carries:
- `source` -- original PDF filename
- `title` -- document title
- `section` -- nearest heading context
- `chunk_id` -- position in document
- `total_chunks` -- total for this document

---

## Stage 5: Embedding and Storage

**Embeddings:** EmbeddingGemma 300M (`google/embeddinggemma-300m`)
- 768-dimensional vectors (full Matryoshka dimension)
- Device auto-detection: CUDA -> MPS -> CPU

**Vector Store:** ChromaDB (persistent)
- Location: `Data/vectorstore/`
- Collection: `ckd_guidelines`
- Document IDs: `{source_filename}_{chunk_id}`
- Batch ingestion: 100 documents per batch

---

## Stage 6: Retrieval

**CKDRetriever** (primary):
- Semantic similarity search with score threshold (default `SIMILARITY_THRESHOLD = 0.3`)
- Medical synonym expansion (e.g. "ckd" -> ["chronic kidney disease", "renal disease"])

**HybridRetriever** (alternative):
- Reciprocal rank fusion: semantic + keyword search

For the full retriever inventory (flat / tree / RAPTOR / contextual) see
[`docs/modules/simple-rag.md`](modules/simple-rag.md).

---

## Configuration (`config.py`)

```python
CHUNK_SIZE = 2000         # tokens
CHUNK_OVERLAP = 200       # trailing blocks (capped at 150 tokens)
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3
CHROMA_COLLECTION_NAME = "ckd_guidelines"
CHROMA_PERSIST_DIRECTORY = "Data/vectorstore"
```

---

## Source Documents

22 clinical guidelines covering CKD management:
- KDIGO guidelines (CKD, Anemia, IgAN/IgAV)
- NICE guidelines (CKD assessment, Hypertension)
- UK Kidney Association clinical practice guidelines
- NHS nutritional and dietary guidelines
- Specialist guidelines (Exercise, Haemodialysis, Vascular Access)
