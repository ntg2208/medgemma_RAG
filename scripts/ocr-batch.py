#!/usr/bin/env python3
"""Batch-process all PDFs with PaddleOCR-VL.

Key optimisation: loads the model ONCE and processes all PDFs sequentially,
avoiding the ~30-60s model-load overhead per file that ocr-process.sh incurs.

Optionally spawns N worker processes, each with its own model instance,
splitting the PDF list. Useful when GPU VRAM allows multiple model copies
(e.g. g6e.xlarge with 48GB L40S).

Usage:
    # Sequential (single model, all PDFs):
    python scripts/ocr-batch.py

    # Two parallel workers (needs ~2x VRAM):
    python scripts/ocr-batch.py --workers 2

    # Custom dirs:
    python scripts/ocr-batch.py -i Data/documents -o Data/processed_ocr

    # Force reprocess (ignore existing outputs):
    python scripts/ocr-batch.py --force
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def extract_results(merged):
    """Extract markdown, json from merged results."""
    md_parts = []
    json_parts = []
    for res in merged:
        md_data = res.markdown
        if isinstance(md_data, dict):
            md_parts.append(md_data.get("markdown_texts", str(md_data)))
        else:
            md_parts.append(str(md_data))

        json_data = res.json
        if isinstance(json_data, dict):
            json_parts.append(json_data)
        elif isinstance(json_data, str):
            try:
                json_parts.append(json.loads(json_data))
            except json.JSONDecodeError:
                json_parts.append({"raw": json_data})
        else:
            json_parts.append({"raw": str(json_data)})
    return md_parts, json_parts


def write_outputs(output_dir, stem, pdf_name, md_parts, json_parts):
    """Write md, json, txt outputs."""
    full_md = "\n\n".join(md_parts)

    # Extract title from first heading
    title = stem
    for line in full_md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s.lstrip("#").strip()
            break

    # Write markdown
    (output_dir / f"{stem}.md").write_text(full_md, encoding="utf-8")

    # Write JSON
    json_out = {"title": title, "source_file": pdf_name, "pages": json_parts}
    (output_dir / f"{stem}.json").write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write plain text
    text_lines = []
    for line in full_md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        if s and all(c in "-| " for c in s):
            continue
        text_lines.append(s)
    (output_dir / f"{stem}.txt").write_text("\n".join(text_lines), encoding="utf-8")

    return len(full_md)


def worker_fn(args):
    """Worker: load model once, process assigned PDFs."""
    pdf_paths, output_dir, device, worker_id, force = args
    from paddleocr import PaddleOCRVL

    kwargs = {"vl_rec_max_concurrency": 4}
    if device:
        kwargs["device"] = device

    print(f"[W{worker_id}] Loading PaddleOCR-VL model...", flush=True)
    pipeline = PaddleOCRVL(**kwargs)
    print(f"[W{worker_id}] Model loaded. {len(pdf_paths)} PDFs to process.", flush=True)

    results = []
    for i, pdf in enumerate(pdf_paths, 1):
        stem = pdf.stem
        md_path = output_dir / f"{stem}.md"

        if not force and md_path.exists():
            print(f"[W{worker_id}] [{i}/{len(pdf_paths)}] SKIP: {stem}", flush=True)
            results.append({"file": pdf.name, "skipped": True})
            continue

        print(f"[W{worker_id}] [{i}/{len(pdf_paths)}] {pdf.name}", flush=True)
        t0 = time.time()
        try:
            page_results = list(pipeline.predict(str(pdf)))
            merged = pipeline.restructure_pages(
                page_results,
                merge_tables=True,
                relevel_titles=True,
                concatenate_pages=True,
            )
            md_parts, json_parts = extract_results(merged)
            char_count = write_outputs(output_dir, stem, pdf.name, md_parts, json_parts)
            elapsed = time.time() - t0
            pages = len(page_results)
            pps = pages / elapsed if elapsed > 0 else 0
            print(f"[W{worker_id}]   {pages} pages, {char_count} chars, "
                  f"{elapsed:.1f}s ({pps:.1f} p/s)", flush=True)
            results.append({
                "file": pdf.name, "pages": pages,
                "time": round(elapsed, 1), "pages_per_sec": round(pps, 1),
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[W{worker_id}]   ERROR after {elapsed:.0f}s: {e}", flush=True)
            results.append({"file": pdf.name, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process PDFs with PaddleOCR-VL (single model load)",
    )
    parser.add_argument(
        "-i", "--input-dir", default="Data/documents",
        help="Directory with input PDFs (default: Data/documents)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="Data/processed_ocr",
        help="Output directory (default: Data/processed_ocr)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: gpu, gpu:0, cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers, each loads own model (default: 1)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess PDFs even if output exists",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return 1

    total_start = time.time()
    print("=" * 60)
    print("PaddleOCR-VL Batch Processing")
    print("=" * 60)
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"PDFs:    {len(pdf_files)}")
    print(f"Workers: {args.workers}")
    print(f"Device:  {args.device or 'auto'}")
    print(f"Force:   {args.force}")
    print("=" * 60)
    print(flush=True)

    if args.workers <= 1:
        all_results = worker_fn(
            (pdf_files, output_dir, args.device, 0, args.force)
        )
    else:
        # Round-robin split PDFs across workers
        chunks = [[] for _ in range(args.workers)]
        for i, pdf in enumerate(pdf_files):
            chunks[i % args.workers].append(pdf)
        chunks = [c for c in chunks if c]

        worker_args = [
            (chunk, output_dir, args.device, wid, args.force)
            for wid, chunk in enumerate(chunks)
        ]

        print(f"Spawning {len(chunks)} worker processes...\n", flush=True)
        with mp.Pool(len(chunks)) as pool:
            nested = pool.map(worker_fn, worker_args)
        all_results = [r for batch in nested for r in batch]

    # Summary
    total_elapsed = time.time() - total_start
    processed = [r for r in all_results if "error" not in r and not r.get("skipped")]
    failed = [r for r in all_results if "error" in r]
    skipped = [r for r in all_results if r.get("skipped")]
    total_pages = sum(r.get("pages", 0) for r in processed)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time:    {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Processed:     {len(processed)}")
    print(f"Skipped:       {len(skipped)}")
    print(f"Failed:        {len(failed)}")
    print(f"Total pages:   {total_pages}")
    if total_elapsed > 0 and total_pages > 0:
        print(f"Overall speed: {total_pages/total_elapsed:.1f} pages/sec")
    if failed:
        print(f"\nFailed files:")
        for r in failed:
            print(f"  - {r['file']}: {r['error']}")

    # Write summary JSON
    summary = {
        "total_time_sec": round(total_elapsed, 1),
        "total_pages": total_pages,
        "pages_per_sec": round(total_pages / total_elapsed, 1) if total_elapsed > 0 else 0,
        "workers": args.workers,
        "results": all_results,
    }
    summary_path = output_dir / "_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
