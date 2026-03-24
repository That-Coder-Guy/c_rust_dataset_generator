"""
Phase 4 — Storage and Export
Reads validated snippets from SQLite, builds a search index,
and exports to JSONL, CSV, and HuggingFace datasets format.
Outputs:
  output/dataset.jsonl
  output/dataset.csv
  output/dataset_hf/          ← HuggingFace-compatible parquet + dataset_info.json
  output/index.db             ← FTS5 full-text search index
  output/export_summary.json
"""

import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH        = Path("output/snippets.db")
INDEX_DB_PATH  = Path("output/index.db")
JSONL_PATH     = Path("output/dataset.jsonl")
CSV_PATH       = Path("output/dataset.csv")
HF_DIR         = Path("output/dataset_hf")
SUMMARY_PATH   = Path("output/export_summary.json")

JSONL_FIELDS = [
    "id", "category", "difficulty", "description",
    "c_code", "rust_code", "generated_at",
]

CSV_FIELDS = JSONL_FIELDS  # same columns

HF_SCHEMA_FIELDS = {
    "id":           "string",
    "category":     "string",
    "difficulty":   "string",
    "description":  "string",
    "c_code":       "string",
    "rust_code":    "string",
    "generated_at": "string",
}

# ── Source DB ─────────────────────────────────────────────────────────────────

def get_validated(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT {', '.join(JSONL_FIELDS)} FROM raw_snippets WHERE status='validated' ORDER BY category, difficulty, id"
    ).fetchall()


# ── Export 1: JSONL ───────────────────────────────────────────────────────────

def export_jsonl(rows: list[sqlite3.Row], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")
    print(f"✓ JSONL  → {path}  ({len(rows)} rows)")


# ── Export 2: CSV ─────────────────────────────────────────────────────────────

def export_csv(rows: list[sqlite3.Row], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})
    print(f"✓ CSV    → {path}  ({len(rows)} rows)")


# ── Export 3: HuggingFace datasets format ─────────────────────────────────────

def export_huggingface(rows: list[sqlite3.Row], out_dir: Path):
    """
    Writes a HuggingFace-compatible dataset:
      dataset_hf/
        train.jsonl         ← 90% of data
        test.jsonl          ← 10% of data
        dataset_info.json   ← schema + stats
        README.md           ← dataset card
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train/test split (90/10, deterministic by row order)
    split_idx = int(len(rows) * 0.9)
    train_rows = rows[:split_idx]
    test_rows  = rows[split_idx:]

    for split_name, split_rows in [("train", train_rows), ("test", test_rows)]:
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(dict(row)) + "\n")

    # Count by category
    cat_counts: dict[str, int] = {}
    for row in rows:
        cat_counts[row["category"]] = cat_counts.get(row["category"], 0) + 1

    dataset_info = {
        "dataset_name": "c_rust_snippet_pairs",
        "description": "5,000 paired C and Rust code snippets with semantic equivalence validation.",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(rows),
        "splits": {
            "train": len(train_rows),
            "test":  len(test_rows),
        },
        "features": HF_SCHEMA_FIELDS,
        "categories": cat_counts,
        "license": "MIT",
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    readme = f"""# C / Rust Snippet Pairs Dataset

{dataset_info['description']}

## Stats
- Total pairs: **{len(rows):,}**
- Train: {len(train_rows):,} | Test: {len(test_rows):,}
- Categories: {', '.join(cat_counts.keys())}

## Fields
| Field | Type | Description |
|---|---|---|
| id | string | Unique identifier |
| category | string | Snippet category |
| difficulty | string | beginner / intermediate / advanced |
| description | string | One-sentence summary |
| c_code | string | C snippet |
| rust_code | string | Equivalent Rust snippet |
| generated_at | string | ISO timestamp |

## Usage
```python
from datasets import load_dataset
ds = load_dataset("json", data_files={{"train": "train.jsonl", "test": "test.jsonl"}})
```
"""
    (out_dir / "README.md").write_text(readme)
    print(f"✓ HuggingFace → {out_dir}/  (train={len(train_rows)}, test={len(test_rows)})")


# ── Export 4: FTS5 search index ───────────────────────────────────────────────

def build_search_index(rows: list[sqlite3.Row], index_path: Path):
    """
    Build a SQLite FTS5 full-text search index over descriptions and code.
    Allows queries like: SELECT * FROM snippets_fts WHERE snippets_fts MATCH 'linked list';
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        index_path.unlink()

    conn = sqlite3.connect(index_path)
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS snippets_fts USING fts5(
            id UNINDEXED,
            category,
            difficulty,
            description,
            c_code,
            rust_code,
            content='',
            tokenize='porter ascii'
        );
    """)

    conn.executemany(
        "INSERT INTO snippets_fts(id, category, difficulty, description, c_code, rust_code) VALUES (?,?,?,?,?,?)",
        [
            (row["id"], row["category"], row["difficulty"],
             row["description"], row["c_code"], row["rust_code"])
            for row in rows
        ],
    )
    conn.commit()
    conn.close()
    print(f"✓ FTS5 index → {index_path}  ({len(rows)} entries)")


# ── Export summary ────────────────────────────────────────────────────────────

def write_summary(rows: list[sqlite3.Row], paths: dict):
    cat_diff: dict[str, dict[str, int]] = {}
    for row in rows:
        cat_diff.setdefault(row["category"], {})
        cat_diff[row["category"]][row["difficulty"]] = \
            cat_diff[row["category"]].get(row["difficulty"], 0) + 1

    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_validated_pairs": len(rows),
        "output_files": {k: str(v) for k, v in paths.items()},
        "by_category_and_difficulty": cat_diff,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"✓ Summary → {SUMMARY_PATH}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Run phases 1–3 first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = get_validated(conn)
    conn.close()

    if not rows:
        print("No validated snippets found. Run phase 3 first.")
        return

    print(f"\nExporting {len(rows):,} validated pairs...\n")

    export_jsonl(rows, JSONL_PATH)
    export_csv(rows, CSV_PATH)
    export_huggingface(rows, HF_DIR)
    build_search_index(rows, INDEX_DB_PATH)

    summary = write_summary(rows, {
        "jsonl":       JSONL_PATH,
        "csv":         CSV_PATH,
        "huggingface": HF_DIR,
        "fts_index":   INDEX_DB_PATH,
    })

    print(f"\n── Export complete ──")
    print(f"  Total pairs exported : {summary['total_validated_pairs']:,}")
    print(f"\n── By category ──")
    for cat, diffs in summary["by_category_and_difficulty"].items():
        total = sum(diffs.values())
        detail = "  ".join(f"{d}:{n}" for d, n in sorted(diffs.items()))
        print(f"  {cat:<30} {total:>4}   ({detail})")


if __name__ == "__main__":
    main()
