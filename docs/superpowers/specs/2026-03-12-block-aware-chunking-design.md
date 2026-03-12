# Block-Aware Markdown Chunking

## Problem

The current chunking pipeline in `Data/preprocessing.py` extracts text from PDFs with PyMuPDF, then splits with `RecursiveCharacterTextSplitter`. This:
- Loses document structure (paragraphs, lists, tables split mid-way)
- Uses unreliable heuristic section/page detection
- Includes CKD stage detection that is no longer needed

Documents have already been cleaned and exported as markdown with section structure in `Data/processed_with_sections/`. The chunking pipeline should read from these instead.

## Input

Each document lives in `Data/processed_with_sections/{doc_name}/`:
- `main_text.md` — markdown with headings, paragraphs, lists, tables
- `metadata.json` — title, source_file, summary, front_matter, processing stats

Only `main_text.md` is chunked. `references.md` and front_matter are excluded.

## Design

### Block Parser

Reads `main_text.md` line by line and groups lines into typed blocks:

- **Heading**: line starting with `#`. Always a single line.
- **Table**: contiguous lines where the content lines start with `|`. Includes the `|---|` separator.
- **List**: contiguous lines starting with `- `, `* `, or `\d[\d.]*\. ` (covers both `1. ` and dotted numbering like `1.1.1 `), plus any continuation lines (indented lines following a list item that don't start a new block type).
- **Paragraph**: contiguous non-blank lines that aren't heading/table/list.

Blank lines are block separators (not blocks themselves).

Each block is a `NamedTuple("Block", type, text, heading_context)` where `heading_context` is the most recent heading text seen before this block, defaulting to `None` if no heading has been encountered yet. A heading block's own `heading_context` is itself (it updates the tracker immediately).

### Greedy Packer

Walks the block list and packs blocks into chunks:

1. Start a new chunk with an empty accumulator.
2. For each block, estimate its token count (`len(text) // 4`).
3. If adding this block would exceed `CHUNK_SIZE` tokens and the accumulator is non-empty:
   - Emit the accumulated blocks as a chunk.
   - Start a new accumulator. Copy trailing blocks from the previous chunk up to `CHUNK_OVERLAP` blocks, but not exceeding ~150 tokens total. This caps overlap to avoid repeating large blocks.
4. Add the block to the accumulator.
5. If a single block exceeds `CHUNK_SIZE`, emit it as its own chunk (never split a block — accept oversized chunks to preserve table/list integrity).
6. After all blocks, emit any remaining accumulated blocks.

**Heading rule:** A heading block is always packed together with the next non-heading block. Never emit a chunk that ends with a heading alone, and never carry a heading as the sole overlap block.

### Chunk Assembly

Each chunk joins its block texts with `\n\n`. Metadata per chunk:
- `title` — from `metadata.json`
- `source_file` — from `metadata.json`
- `section` — the `heading_context` of the first block in the chunk
- `chunk_id` — 0-indexed sequential
- `total_chunks` — set after all chunks created

### Token Estimation

Use `len(text) // 4` as rough char-to-token conversion (same as current code). `CHUNK_SIZE` in config stays at 800 tokens.

### Overlap

`CHUNK_OVERLAP` changes meaning from character overlap to **block count overlap with a token cap**. Default: 1 block, capped at 150 tokens. This means the last block(s) of chunk N are repeated at the start of chunk N+1, but only if they fit within the token cap. This prevents a single large table or paragraph from being duplicated as overlap.

## File Changes

### `config.py`
- Add `PROCESSED_WITH_SECTIONS_DIR = DATA_DIR / "processed_with_sections"`
- Remove `MIN_CHUNK_SIZE` (blocks are atomic, no minimum filtering needed)
- Change `CHUNK_OVERLAP` default to `1` (block count)

### `Data/preprocessing.py`
Rewrite the `DocumentPreprocessor` class:

**Remove:**
- `extract_text_from_pdf` — no longer reading PDFs
- `clean_text` — markdown is already cleaned
- `extract_ckd_stages` — removed per requirement
- `CKD_STAGE_PATTERNS` — removed
- `NOISE_PATTERNS` — removed
- `detect_section` — replaced by heading tracking in block parser
- `classify_document_type` — use metadata.json instead

**Add:**
- `parse_blocks(text: str) -> list[Block]` — the block parser
- `pack_chunks(blocks: list[Block]) -> list[Document]` — the greedy packer

**Keep (updated defaults):**
- `process_directory(input_dir) -> list[Document]` — iterates document dirs instead of PDFs. Default `input_dir` changes to `PROCESSED_WITH_SECTIONS_DIR`. Drop `output_dir` param (no longer writing intermediate files).
- `preprocess_documents(input_dir, chunk_size, chunk_overlap) -> list[Document]` — convenience function, updated defaults.
- `get_document_stats(documents) -> dict` — remove `ckd_stage_coverage` from stats

### `Data/export_chunks.py`
- Update `prepare_chunk_data` to drop `ckd_stages` and `page_number` fields
- Keep `section` field (now populated from heading tracking)
- Drop `document_type` from per-chunk metadata (it's a document-level property, can stay in export_metadata)

## Output

Same as current: `list[Document]` where each Document has:
- `page_content`: the chunk text (joined blocks with `\n\n`)
- `metadata`: `{title, source, section, chunk_id, total_chunks}`

Note: metadata key is `source` (not `source_file`) to match existing downstream code in vectorstore and export.

Export JSON structure stays the same shape, minus `ckd_stages` and `page_number` per chunk.

## Tests

- Block parser: test each block type detection (heading, table, list with dotted numbering, paragraph)
- Greedy packer: test normal packing, oversized single block, heading-sticks-to-next-block rule
- Overlap: test token-capped block overlap
- Integration: process one real document directory end-to-end
