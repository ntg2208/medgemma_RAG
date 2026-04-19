"""
Build retriever evaluation ground truth by searching actual chunk content.

For each evaluation query, searches all chunks for the query's keywords
and reports which sources contain meaningful matches. This produces
data-driven ground truth rather than filename-guessed assignments.

Usage:
    uv run python tests/build_eval_ground_truth.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "Data" / "processed"
OUTPUT_PATH = Path(__file__).parent / "retriever_eval_dataset.json"


def load_all_chunks() -> list[dict]:
    """Load every chunk from all JSON files."""
    chunks = []
    for f in sorted(PROCESSED_DIR.glob("*_chunks.json")):
        with open(f) as fp:
            data = json.load(fp)
        for c in data.get("chunks", []):
            chunks.append({
                "source": c["metadata"].get("source", ""),
                "section": c["metadata"].get("section", ""),
                "content": c["content"],
                "doc_name": c["metadata"].get("doc_name", ""),
            })
    return chunks


def search_chunks(
    chunks: list[dict],
    must_have: list[str],
    any_of: list[list[str]] | None = None,
    min_matches: int = 2,
    min_density: float = 0.0,
) -> dict[str, int]:
    """Search chunks for keyword groups with density filtering.

    A source is only relevant if it has enough matching chunks (min_matches)
    AND a high enough ratio of matching chunks to total chunks (min_density).
    This filters out incidental mentions in large documents.

    Args:
        chunks: All chunks.
        must_have: All of these terms must appear in a chunk (case-insensitive).
        any_of: List of synonym groups — at least one term from each group
                must appear. E.g. [["anaemia", "anemia"], ["esa", "erythropoietin"]]
        min_matches: Minimum number of matching chunks per source.
        min_density: Minimum ratio of matching chunks to total chunks per source.
                     E.g. 0.05 means at least 5% of a document's chunks must match.

    Returns:
        Dict of source -> match_count for sources meeting both thresholds.
    """
    source_counts: dict[str, int] = defaultdict(int)
    source_totals: dict[str, int] = defaultdict(int)

    # Count totals per source
    for chunk in chunks:
        source_totals[chunk["source"]] += 1

    for chunk in chunks:
        text = chunk["content"].lower()

        # Check must_have terms
        if not all(term.lower() in text for term in must_have):
            continue

        # Check any_of groups
        if any_of:
            if not all(
                any(syn.lower() in text for syn in group)
                for group in any_of
            ):
                continue

        source_counts[chunk["source"]] += 1

    # Filter by min_matches AND min_density
    results = {}
    for src, count in source_counts.items():
        total = source_totals.get(src, 1)
        density = count / total
        if count >= min_matches and density >= min_density:
            results[src] = count
    return results


# ── Query definitions ────────────────────────────────────────────────────────
# Each query defines:
#   id, query (keywords), topic,
#   must_have (all required), any_of (synonym groups, at least one from each),
#   min_matches (minimum chunk hits per source)

QUERIES = [
    {
        "id": "diet_potassium_restriction",
        "query": "potassium restriction CKD stage 3",
        "topic": "diet",
        "must_have": ["potassium"],
        "any_of": [
            ["restrict", "limit", "intake", "dietary", "diet", "avoid", "reduce"],
            ["ckd", "kidney", "renal", "stage"],
        ],
        "min_matches": 3,
        "min_density": 0.10,
    },
    {
        "id": "diet_phosphorus_hd",
        "query": "phosphorus intake haemodialysis daily limit",
        "topic": "diet",
        "must_have": [],
        "any_of": [
            ["phosphorus", "phosphate"],
            ["diet", "intake", "limit", "restrict", "food", "nutrition"],
            ["haemodialysis", "hemodialysis", "dialysis", "ckd", "kidney"],
        ],
        "min_matches": 3,
        "min_density": 0.08,
    },
    {
        "id": "diet_protein_predialysis",
        "query": "protein intake pre-dialysis CKD recommendation",
        "topic": "diet",
        "must_have": [],
        "any_of": [
            ["protein intake", "dietary protein", "g/kg", "protein restriction", "protein requirement"],
            ["ckd", "kidney", "renal", "dialysis"],
        ],
        "min_matches": 2,
        "min_density": 0.05,
    },
    {
        "id": "diet_fluid_haemodialysis",
        "query": "fluid allowance haemodialysis management",
        "topic": "diet",
        "must_have": ["fluid"],
        "any_of": [
            ["haemodialysis", "hemodialysis", "dialysis"],
            ["allowance", "restrict", "intake", "limit", "balance", "ml", "litre"],
        ],
        "min_matches": 3,
        "min_density": 0.06,
    },
    {
        "id": "diet_sodium_limit",
        "query": "sodium salt daily limit CKD",
        "topic": "diet",
        "must_have": [],
        "any_of": [
            ["sodium", "salt"],
            ["diet", "intake", "limit", "restrict", "reduce", "food"],
            ["ckd", "kidney", "renal"],
        ],
        "min_matches": 3,
        "min_density": 0.08,
    },
    {
        "id": "diet_peritoneal_nutrition",
        "query": "peritoneal dialysis nutrition recommendations",
        "topic": "diet",
        "must_have": ["peritoneal"],
        "any_of": [["nutrition", "diet", "intake", "energy", "protein", "calori"]],
        "min_matches": 3,
        "min_density": 0.08,
    },
    {
        "id": "diet_high_potassium_foods",
        "query": "high potassium foods avoid kidney",
        "topic": "diet",
        "must_have": ["potassium"],
        "any_of": [["banana", "orange", "potato", "tomato", "fruit", "vegetable", "food"]],
        "min_matches": 1,
        "min_density": 0.05,
    },
    {
        "id": "anaemia_esa_initiation",
        "query": "ESA erythropoietin initiation CKD anaemia",
        "topic": "anaemia",
        "must_have": [],
        "any_of": [
            ["esa", "erythropoietin", "erythropoiesis stimulating", "epoetin", "darbepoetin"],
            ["anaemia", "anemia", "haemoglobin", "hemoglobin"],
        ],
        "min_matches": 5,
        "min_density": 0.05,
    },
    {
        "id": "anaemia_iron_ferritin_tsat",
        "query": "ferritin transferrin saturation iron target CKD",
        "topic": "anaemia",
        "must_have": ["ferritin"],
        "any_of": [
            ["iron", "transferrin", "tsat"],
            ["anaemia", "anemia", "ckd", "renal"],
        ],
        "min_matches": 3,
        "min_density": 0.03,
    },
    {
        "id": "anaemia_haemoglobin_target",
        "query": "haemoglobin target range anaemia CKD",
        "topic": "anaemia",
        "must_have": [],
        "any_of": [
            ["haemoglobin", "hemoglobin"],
            ["target", "range", "g/l", "g/dl"],
            ["anaemia", "anemia"],
        ],
        "min_matches": 3,
        "min_density": 0.04,
    },
    {
        "id": "anaemia_hif_phi",
        "query": "HIF-PHI roxadustat daprodustat anaemia CKD",
        "topic": "anaemia",
        "must_have": [],
        "any_of": [["hif-ph", "roxadustat", "daprodustat", "vadadustat", "molidustat", "prolyl hydroxylase"]],
        "min_matches": 2,
        "min_density": 0.02,
    },
    {
        "id": "bp_target_dialysis",
        "query": "blood pressure target dialysis adults",
        "topic": "blood_pressure",
        "must_have": ["blood pressure"],
        "any_of": [
            ["target", "mmhg", "systolic", "diastolic"],
            ["dialysis", "haemodialysis", "hemodialysis"],
        ],
        "min_matches": 3,
        "min_density": 0.08,
    },
    {
        "id": "bp_children_dialysis",
        "query": "blood pressure children young people dialysis",
        "topic": "blood_pressure",
        "must_have": ["blood pressure"],
        "any_of": [
            ["child", "children", "paediatric", "pediatric", "young people"],
            ["dialysis", "haemodialysis"],
        ],
        "min_matches": 3,
        "min_density": 0.06,
    },
    {
        "id": "bp_hypertension_nice_ng136",
        "query": "NICE NG136 hypertension CKD commentary",
        "topic": "blood_pressure",
        "must_have": ["ng136"],
        "any_of": [["hypertension", "blood pressure"]],
        "min_matches": 2,
        "min_density": 0.10,
    },
    {
        "id": "hyperkalaemia_acute_treatment",
        "query": "acute hyperkalaemia emergency treatment protocol",
        "topic": "hyperkalaemia",
        "must_have": [],
        "any_of": [
            ["hyperkalaemia", "hyperkalemia"],
            ["treatment", "management", "emergency", "acute", "protocol"],
        ],
        "min_matches": 5,
        "min_density": 0.10,
    },
    {
        "id": "hyperkalaemia_insulin_glucose",
        "query": "insulin glucose infusion hyperkalaemia",
        "topic": "hyperkalaemia",
        "must_have": ["insulin"],
        "any_of": [
            ["glucose", "dextrose"],
            ["hyperkalaemia", "hyperkalemia"],
        ],
        "min_matches": 2,
        "min_density": 0.02,
    },
    {
        "id": "hyperkalaemia_calcium_gluconate",
        "query": "calcium gluconate IV hyperkalaemia cardiac protection",
        "topic": "hyperkalaemia",
        "must_have": ["calcium gluconate"],
        "any_of": [["hyperkalaemia", "hyperkalemia"]],
        "min_matches": 2,
        "min_density": 0.02,
    },
    {
        "id": "vascular_access_fistula",
        "query": "arteriovenous fistula AVF creation haemodialysis",
        "topic": "vascular_access",
        "must_have": [],
        "any_of": [["fistula", "avf", "arteriovenous"]],
        "min_matches": 4,
        "min_density": 0.10,
    },
    {
        "id": "vascular_access_catheter",
        "query": "central venous catheter dialysis access",
        "topic": "vascular_access",
        "must_have": ["catheter"],
        "any_of": [["dialysis", "haemodialysis", "hemodialysis", "vascular access"]],
        "min_matches": 3,
        "min_density": 0.05,
    },
    {
        "id": "hd_adequacy_kt_v",
        "query": "haemodialysis adequacy Kt/V URR target",
        "topic": "haemodialysis",
        "must_have": ["kt/v"],
        "any_of": [["haemodialysis", "hemodialysis", "dialysis"]],
        "min_matches": 2,
        "min_density": 0.03,
    },
    {
        "id": "exercise_ckd_recommendation",
        "query": "exercise physical activity CKD recommendation",
        "topic": "exercise",
        "must_have": [],
        "any_of": [
            ["exercise", "physical activity"],
            ["ckd", "kidney", "renal"],
        ],
        "min_matches": 5,
        "min_density": 0.10,
    },
    {
        "id": "exercise_intradialytic",
        "query": "intradialytic exercise haemodialysis safety",
        "topic": "exercise",
        "must_have": [],
        "any_of": [
            ["intradialytic", "during dialysis", "during haemodialysis"],
            ["exercise", "physical activity"],
        ],
        "min_matches": 2,
        "min_density": 0.03,
    },
    {
        "id": "smoking_ckd_progression",
        "query": "smoking cessation CKD progression lifestyle",
        "topic": "lifestyle",
        "must_have": ["smoking"],
        "any_of": [["ckd", "kidney", "renal", "progression"]],
        "min_matches": 2,
        "min_density": 0.03,
    },
    {
        "id": "rrt_conservative_management",
        "query": "renal replacement therapy conservative management stage 5",
        "topic": "rrt",
        "must_have": ["conservative"],
        "any_of": [
            ["renal replacement", "rrt", "dialysis", "stage 5", "kidney failure"],
        ],
        "min_matches": 3,
        "min_density": 0.08,
    },
    {
        "id": "ckd_staging_egfr_albuminuria",
        "query": "CKD staging eGFR albuminuria classification",
        "topic": "ckd_general",
        "must_have": ["egfr"],
        "any_of": [
            ["stage", "staging", "classification", "category", "g1", "g2", "g3"],
            ["ckd", "kidney disease"],
        ],
        "min_matches": 3,
        "min_density": 0.05,
    },
    {
        "id": "ckd_referral_nephrology",
        "query": "CKD referral nephrologist criteria",
        "topic": "ckd_general",
        "must_have": ["referral"],
        "any_of": [["nephrolog", "specialist", "secondary care"]],
        "min_matches": 2,
        "min_density": 0.04,
    },
    {
        "id": "igan_treatment_budesonide",
        "query": "IgA nephropathy treatment immunosuppression budesonide",
        "topic": "igan",
        "must_have": [],
        "any_of": [
            ["iga nephropathy", "igan"],
            ["treatment", "budesonide", "nefecon", "immunosuppress", "corticosteroid", "glucocorticoid"],
        ],
        "min_matches": 5,
        "min_density": 0.08,
    },
    {
        "id": "igav_children_hsp",
        "query": "IgA vasculitis Henoch Schonlein purpura children",
        "topic": "igan",
        "must_have": [],
        "any_of": [
            ["iga vasculitis", "igav", "henoch", "schonlein"],
            ["child", "children", "paediatric", "pediatric", "young"],
        ],
        "min_matches": 3,
        "min_density": 0.08,
    },
    {
        "id": "aki_nutrition",
        "query": "acute kidney injury nutrition feeding",
        "topic": "aki",
        "must_have": [],
        "any_of": [
            ["acute kidney injury", " aki "],
            ["nutrition", "feed", "diet", "energy"],
        ],
        "min_matches": 2,
        "min_density": 0.08,
    },
    {
        "id": "nsaid_nephrotoxic",
        "query": "NSAID ibuprofen nephrotoxic CKD avoid",
        "topic": "medication",
        "must_have": [],
        "any_of": [
            ["nsaid", "ibuprofen", "naproxen", "non-steroidal"],
            ["avoid", "nephrotoxic", "contraindic", "caution"],
        ],
        "min_matches": 2,
        "min_density": 0.03,
    },
    {
        "id": "ace_arb_proteinuria",
        "query": "ACE inhibitor ARB proteinuria CKD",
        "topic": "medication",
        "must_have": [],
        "any_of": [
            ["ace inhibitor", "angiotensin", "ramipril", "enalapril", "losartan"],
            ["proteinuria", "albuminuria"],
        ],
        "min_matches": 2,
        "min_density": 0.04,
    },
    {
        "id": "phosphate_binder",
        "query": "phosphate binder CKD calcium sevelamer",
        "topic": "medication",
        "must_have": ["phosphate binder"],
        "any_of": [],
        "min_matches": 1,
        "min_density": 0.02,
    },
]


def main():
    print("Loading all chunks ...")
    chunks = load_all_chunks()
    print(f"  {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} sources\n")

    dataset_entries = []

    # Build source totals for density display
    source_totals: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        source_totals[chunk["source"]] += 1

    for q in QUERIES:
        matches = search_chunks(
            chunks,
            must_have=q["must_have"],
            any_of=q.get("any_of"),
            min_matches=q.get("min_matches", 2),
            min_density=q.get("min_density", 0.0),
        )

        # Sort by match count descending
        sorted_sources = sorted(matches.items(), key=lambda x: -x[1])

        print(f"--- {q['id']} ---")
        print(f"  Query: {q['query']}")
        print(f"  Filters: must_have={q['must_have']} min_matches={q.get('min_matches',2)} min_density={q.get('min_density',0.0)}")
        print(f"  Matched sources ({len(sorted_sources)}):")
        for src, count in sorted_sources:
            total = source_totals.get(src, 1)
            density = count / total
            print(f"    [{count:3d}/{total:3d} = {density:.0%}] {src}")
        print()

        dataset_entries.append({
            "id": q["id"],
            "query": q["query"],
            "topic": q["topic"],
            "relevant_sources": [src for src, _ in sorted_sources],
        })

    # Write dataset
    dataset = {
        "description": "Retriever evaluation dataset for the CKD RAG system. Ground truth built by keyword search of actual chunk content.",
        "version": "3.0",
        "entries": dataset_entries,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset written to {OUTPUT_PATH}")
    print(f"  {len(dataset_entries)} queries")
    avg_sources = sum(len(e['relevant_sources']) for e in dataset_entries) / len(dataset_entries)
    print(f"  Avg relevant sources per query: {avg_sources:.1f}")


if __name__ == "__main__":
    main()
