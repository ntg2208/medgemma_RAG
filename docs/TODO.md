# TODO — Update / Upgrade / Fix

Tracking list of follow-ups discovered during the doc-vs-code audit on
2026-04-25 before the ship snapshot. Grouped by intent: **Fix** (incorrect /
inconsistent), **Update** (refresh / re-run), **Upgrade** (deliberate
improvement for a future iteration).

Items marked `(done 2026-04-25)` were resolved during the same audit /
restructure session and are kept here as a paper trail.

---

## Fix — doc/code mismatches

These were factually wrong claims; most are now corrected.

- [x] **CHUNK_SIZE / CHUNK_OVERLAP stale across docs.** *(done 2026-04-25)*
      Real values: `CHUNK_SIZE = 2000`, `CHUNK_OVERLAP = 200` (`config.py:42-43`).
      Fixed in `README.md`, `docs/modules/simple-rag.md`, `docs/data-pipeline.md`.
- [x] **`SIMILARITY_THRESHOLD` stale across docs.** *(done 2026-04-25)*
      Real value `0.3` now consistent across `docs/architecture.md` and
      `docs/data-pipeline.md`.
- [x] **PDF count.** *(done 2026-04-25)* Actual = 22; updated in `README.md`
      and `docs/data-pipeline.md`.
- [x] **Test count phrasing.** *(done 2026-04-25)* Removed the "(50+ tests)"
      figure from the project tree; replaced with "Pytest test suite".
- [x] **Agentic eval `intent_accuracy` historical bug.** *(done 2026-04-25)*
      Fixed in `eval/run_agentic_eval.py:162` (case-insensitive comparison).
      Documented under "Historical note on intent_accuracy" in
      [`docs/evaluation.md`](evaluation.md).
- [x] **Doc cross-references after the restructure.** *(done 2026-04-25)*
      `docs/usage.md` and `README.md` updated; the merged `docs/evaluation.md`
      replaces the old `docs/evaluation_guide.md` and `docs/ragas_eval_notes.md`.
- [ ] **`config.py:43` comment is wrong.** `CHUNK_OVERLAP = 200` is documented
      in code as "characters of overlap between consecutive chunks" but
      `Data/preprocessing.py:152` uses it as `chunk_overlap` = *number of
      trailing blocks* (capped at 150 tokens via `overlap_token_cap`). Either
      fix the comment to "trailing blocks (capped at 150 tokens)" or change
      the value back to a sensible block count (it was `1` historically).
      The current value is effectively no-op past the 150-token cap.

---

## Update — refresh stale numbers / re-run

Things to do once before the next public update.

- [ ] **Re-run `eval/run_agentic_eval.py`** at the new `temperature=0.3`
      (already in `scripts/startup.sh`) and replace the "Latest run" tables
      in [`README.md`](../README.md) and [`docs/evaluation.md`](evaluation.md).
      Expect higher faithfulness.
- [ ] **Multi-agent RAGAS is empty in the latest run** because the
      orchestrator does not surface per-agent contexts into the harness.
      Fix the context propagation in `eval/run_agentic_eval.py`
      (the `for agent_name, agent_resp in response_obj.individual_responses.items()`
      loop at line ~346 only catches `contexts` / `retrieved_contexts`
      attributes — verify each agent response actually exposes one).
- [ ] **Choose the published retriever.** Eval currently hard-wires
      `CKDRetriever` (flat) at `eval/run_agentic_eval.py:101` and `:262`.
      Either:
      - Switch to `TreeRetriever` to match the "default for Simple RAG"
        narrative in `README.md`, **or**
      - Add a `--retriever {flat,tree,raptor,contextual}` CLI flag so the
        same script can compare retriever strategies under the agentic /
        multi-agent path.
- [ ] **Add ground-truth references** to more entries in
      `eval/test_queries_agentic.json`. Several queries lack a `reference`
      field, which collapses RAGAS context_recall to 0.0 and drags the
      aggregate down (current 0.256 is mostly unscored, not low recall).
- [ ] **Verify pytest test count.** Run `uv run pytest --collect-only -q`
      and add the actual function count back to `README.md` if you want a
      concrete figure.
- [ ] **Re-run retriever evals across all four retrievers.** The committed
      results currently only cover `CKDRetriever` (basic):
      - `tests/eval_retriever.py` supports `--retriever {basic,tree,hybrid}`
        but the published confusion matrix is basic-only. Re-run for tree
        and hybrid (and add raptor / contextual support if missing) and
        publish a comparison table in `docs/evaluation.md`.
      - `eval/run_retriever_comparison.py` was last run with `--retriever flat`
        (2026-04-16). Re-run with `--retriever both` (and extend to RAPTOR /
        Contextual) to populate the side-by-side RAGAS table.

---

## Upgrade — planned improvements (next iteration)

Deliberate work that doesn't block shipping but should land in the next round.

### Routing & classification

- [ ] **Intent classifier misses two cases** (see [`docs/evaluation.md`](evaluation.md)
      → "Known limitations"):
  - `agentic_direct_02` ("What does eGFR stand for?") routes to RETRIEVAL
    — the keyword classifier in `agentic_rag/nodes.py` treats `eGFR` as a
    retrieval trigger. Add a definitional-question short-circuit.
  - `clarification_01` ("What should I do about my levels?") routes to
    OUT_OF_SCOPE — no incomplete-context detector exists. Add a "vague
    referent" pattern (`"my X"` with no concrete subject).
- [ ] **Multi-agent over-fans-out to RAG.** 9 of 17 queries triggered the
      `multi` route with the RAG agent included, dragging routing precision
      to 0.598. Tune the secondary threshold in
      `multi_agent_rag/orchestrator.py:171` (currently
      `0.3 * primary_score`) — likely too permissive.
- [ ] **Replace keyword-based intent classification with an LLM call** for
      ambiguous queries (keep keyword fast-path for clear hits to preserve
      latency).

### Generation quality

- [ ] **Faithfulness 0.448** is below the 0.50 "Needs Work" threshold.
      Beyond temperature, consider:
  - Tightening `RAG_PROMPT_TEMPLATE` in `config.py:220` to require explicit
    grounding ("only state facts present in the context").
  - Adding a post-generation faithfulness check that re-asks MedGemma to
    identify unsupported claims.
- [ ] **Citation score 0.59** — the regex citation detector in
      `agentic_rag/evaluation/custom_metrics.py` only matches a few formats.
      Either expand the regex or change the prompt to require a fixed
      citation format.

### Evaluation infrastructure

- [ ] **Streaming RAGAS results.** Currently the script blocks per query
      until RAGAS finishes scoring. Could write per-query JSONL incrementally
      so a crash mid-run preserves partial results.
- [ ] **Add a CKD-stage-stratified report.** Aggregate metrics across stages
      hide per-stage failures (e.g. stage-5 dialysis advice vs stage-1
      lifestyle).
- [ ] **Add a confusion-matrix PNG export** for the agent routing decisions
      (`expected_agents` vs `actual_agents`), similar to the existing
      `tests/retriever_confusion_matrix.png`.

### Docs / housekeeping

- [ ] **Personal context file.** `personal_context_.txt` (note trailing
      underscore) is referenced in `simple_rag/chain.py` via
      `load_patient_context()` but is gitignored as `personal_context*.txt`.
      Either add a `personal_context.example.txt` to the repo or document
      the schema in [`README.md`](../README.md).
- [ ] **Architecture diagram regen.** If you ever export Mermaid diagrams
      to SVG/PNG and check them in, regenerate after the
      `SIMILARITY_THRESHOLD` 0.7→0.3 fix in `docs/architecture.md`.
