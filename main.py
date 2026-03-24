"""
main.py - C/Rust Dataset Generation Orchestrator (llama-cpp-python edition)
Runs all four phases end-to-end with progress tracking, gap analysis,
and automatic re-generation of rejected snippets.

Usage:
    python main.py                     # full run
    python main.py --phase 2           # run only phase 2
    python main.py --phase 3,4         # run phases 3 and 4
    python main.py --resume            # skip already-completed work
    python main.py --dry-run           # show plan without running inference
    python main.py --target 100        # override total count (for testing)
    python main.py --llm codellama:13b --embed nomic-embed-text

Requirements:
    pip install llama-cpp-python numpy json-repair
    # Models should be downloaded as GGUF files to ~/c_rust_dataset/models/
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -- Paths ---------------------------------------------------------------------

OUTPUT_DIR    = Path("output")
TAXONOMY_PATH = OUTPUT_DIR / "taxonomy.json"
DB_PATH       = OUTPUT_DIR / "snippets.db"
STATE_PATH    = OUTPUT_DIR / "run_state.json"
MAIN_LOG      = OUTPUT_DIR / "main.log"

# -- Defaults ------------------------------------------------------------------

import os as _os
from pathlib import Path as _Path
DEFAULT_LLM_MODEL_PATH   = _os.environ.get("LLM_MODEL_PATH",   str(_MODELS / "qwen2.5-coder-32b.gguf"))
DEFAULT_EMBED_MODEL_PATH = _os.environ.get("EMBED_MODEL_PATH", str(_MODELS / "nomic-embed-text.gguf"))


# -- Logging -------------------------------------------------------------------

def setup_logging(level: str = "INFO"):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    # Force UTF-8 on stdout so Windows cp1252 consoles don't error on unicode.
    # reconfigure() is available on Python 3.7+ and is the cleanest cross-platform fix.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt))

    file_handler = logging.FileHandler(MAIN_LOG, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(level=getattr(logging, level), handlers=[stream_handler, file_handler])


log = logging.getLogger("main")


# -- State ---------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "started_at": None,
        "phases_completed": [],
        "total_validated": 0,
        "target": 5000,
        "iterations": 0,
    }


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


# -- Pre-flight ----------------------------------------------------------------

def preflight_checks(llm_model_path: str, embed_model_path: str) -> bool:
    ok = True

    # Python packages
    for pkg in ["llama_cpp", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            log.error(f"Missing package: {pkg}  ->  pip install {pkg.replace('_', '-')}")
            ok = False

    if not ok:
        return False

    # Check model files exist
    from pathlib import Path as _Path
    for label, path in [("LLM", llm_model_path), ("Embed", embed_model_path)]:
        if _Path(path).exists():
            size_gb = _Path(path).stat().st_size / 1e9
            log.info(f"  {label} model: {path} ({size_gb:.1f} GB)")
        else:
            log.error(f"  {label} model not found: {path}")
            ok = False

    # Compile tools
    import shutil
    for tool in ["gcc", "rustc"]:
        found = shutil.which(tool) is not None
        log.info(f"  {tool}: {'found' if found else 'NOT FOUND (compile checks will be skipped)'}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    return ok


# -- Gap analysis --------------------------------------------------------------

def get_gap_analysis(taxonomy: dict) -> dict:
    # Default: nothing validated yet
    empty = {
        cat: {"have": 0, "target": info["target"], "gap": info["target"]}
        for cat, info in taxonomy["categories"].items()
    }

    if not DB_PATH.exists():
        return empty

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT category, COUNT(*) as n FROM raw_snippets "
            "WHERE status='validated' GROUP BY category"
        ).fetchall()
    except sqlite3.OperationalError:
        # raw_snippets table does not exist yet (phase 3 has not run)
        conn.close()
        return empty
    conn.close()

    validated = {r[0]: r[1] for r in rows}
    return {
        cat: {
            "have":   validated.get(cat, 0),
            "target": info["target"],
            "gap":    max(0, info["target"] - validated.get(cat, 0)),
        }
        for cat, info in taxonomy["categories"].items()
    }


def print_progress(taxonomy: dict) -> tuple[int, int]:
    gaps = get_gap_analysis(taxonomy)
    total_have = sum(g["have"]   for g in gaps.values())
    total_need = sum(g["target"] for g in gaps.values())
    pct = total_have / total_need * 100 if total_need else 0

    print(f"\n{'-'*62}")
    print(f"  Progress: {total_have:,} / {total_need:,}  ({pct:.1f}%)")
    print(f"{'-'*62}")
    print(f"  {'Category':<30} {'Have':>6} {'Target':>7} {'Gap':>6}  {'Bar'}")
    print(f"  {'-'*30} {'-'*6} {'-'*7} {'-'*6}  {'-'*20}")
    for cat, g in gaps.items():
        filled = int(g["have"] / g["target"] * 20) if g["target"] else 0
        bar = ("█" * filled).ljust(20)
        print(f"  {cat:<30} {g['have']:>6} {g['target']:>7} {g['gap']:>6}  {bar}")
    print(f"{'-'*62}\n")
    return total_have, total_need


# -- Phase runners (inject model config at import time) ------------------------

def run_phase1(target_override: int | None = None):
    log.info("--- Phase 1: Building taxonomy ---")
    import phase1_taxonomy as p1
    if target_override:
        scale = target_override / sum(c["target"] for c in p1.CATEGORIES.values())
        for cat in p1.CATEGORIES.values():
            cat["target"] = max(1, int(cat["target"] * scale))
        log.info(f"  Targets scaled to ~{target_override} total")
    p1.main()
    log.info("Phase 1 complete [OK]")


def run_phase2(llm_model: str, embed_model: str):
    log.info("--- Phase 2: Generating unique ideas ---")
    import phase2_ideas as p2
    p2.main()
    log.info("Phase 2 complete [OK]")


def run_phase3(llm_model: str):
    log.info("--- Phase 3: Generating code from ideas ---")
    import phase3_generation as p3
    p3.main()
    log.info("Phase 3 complete [OK]")


def run_phase4(llm_model: str, embed_model: str):
    log.info("--- Phase 4: Validating and deduplicating ---")
    import phase4_validation as p4
    p4.main()
    log.info("Phase 4 complete [OK]")


def run_phase5():
    log.info("--- Phase 5: Exporting dataset ---")
    import phase5_export as p5
    p5.main()
    log.info("Phase 5 complete [OK]")


# -- Main ----------------------------------------------------------------------

def main():
    # Change to the directory containing main.py so all relative paths
    # in every phase file resolve correctly regardless of where the script
    # is invoked from (e.g. sbatch from home dir, or python from elsewhere).
    import os
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser(
        description="Generate 5,000 paired C/Rust code snippets using llama-cpp-python."
    )
    parser.add_argument("--phase",     type=str,  default=None,
                        help="Comma-separated phases to run, e.g. '2' or '3,4'")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip already-completed phases")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Show plan without running inference")
    parser.add_argument("--target",    type=int,  default=None,
                        help="Override total pair count (useful for testing, e.g. 100)")
    parser.add_argument("--max-iter",  type=int,  default=5,
                        help="Max re-generation iterations to fill gaps (default: 5)")
    parser.add_argument("--llm",   type=str, default=DEFAULT_LLM_MODEL_PATH,
                        help="Path to LLM GGUF file")
    parser.add_argument("--embed", type=str, default=DEFAULT_EMBED_MODEL_PATH,
                        help="Path to embedding GGUF file")
    parser.add_argument("--log-level", type=str,  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    print("""
+======================================================+
|      C / Rust Snippet Dataset Generator              |
|      Backend: llama-cpp-python (HPC/local)           |
+======================================================+
""")
    log.info(f"LLM model   : {args.llm}")
    log.info(f"Embed model : {args.embed}")

    # Set env vars so llm_backend.py picks up the right model paths
    import os as _os
    _os.environ["LLM_MODEL_PATH"]   = args.llm
    _os.environ["EMBED_MODEL_PATH"] = args.embed

    if not preflight_checks(args.llm, args.embed):
        sys.exit(1)

    state = load_state()
    state["started_at"] = state["started_at"] or datetime.now(timezone.utc).isoformat()
    state["target"]     = args.target or state.get("target", 5000)
    save_state(state)

    # Which phases to run
    if args.phase:
        phases_to_run = {int(p.strip()) for p in args.phase.split(",")}
    else:
        phases_to_run = {1, 2, 3, 4, 5}

    if args.resume:
        phases_to_run -= set(state["phases_completed"])
        log.info(f"Resuming - already completed: {state['phases_completed']}")

    log.info(f"Phases to run: {sorted(phases_to_run)}")

    if args.dry_run:
        print("\n[DRY RUN] Would run phases:", sorted(phases_to_run))
        if TAXONOMY_PATH.exists():
            taxonomy = json.loads(TAXONOMY_PATH.read_text())
            print_progress(taxonomy)
        print("[DRY RUN] No inference calls made.\n")
        return

    # -- Phase 1 ------------------------------------------------------------
    if 1 in phases_to_run:
        t0 = time.time()
        run_phase1(target_override=args.target)
        state["phases_completed"].append(1)
        state["phase1_duration_s"] = round(time.time() - t0, 1)
        save_state(state)

    if not TAXONOMY_PATH.exists():
        log.error("taxonomy.json not found - run phase 1 first.")
        sys.exit(1)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())

    # -- Phase 2: generate all unique ideas first -----------------------
    if 2 in phases_to_run:
        t0 = time.time()
        run_phase2(args.llm, args.embed)
        if 2 not in state["phases_completed"]:
            state["phases_completed"].append(2)
        state["phase2_duration_s"] = round(time.time() - t0, 1)
        save_state(state)

    # -- Phases 3 + 4: code gen -> validation, iterative for gaps ------
    if 3 in phases_to_run or 4 in phases_to_run:
        for iteration in range(1, args.max_iter + 1):
            state["iterations"] = iteration
            save_state(state)

            log.info(f"\n{'='*60}")
            log.info(f"  Iteration {iteration}/{args.max_iter}")
            log.info(f"{'='*60}")

            total_have, total_need = print_progress(taxonomy)
            if total_have >= total_need:
                log.info(f"  Target of {total_need:,} already met!")
                break

            # Replace any ideas the LLM completely failed to implement
            if DB_PATH.exists():
                import sqlite3 as _sq
                _c = _sq.connect(DB_PATH)
                failed_ideas = _c.execute(
                    "SELECT COUNT(*) FROM ideas WHERE status='failed'"
                ).fetchone()[0]
                _c.close()
                if failed_ideas > 0:
                    log.info(f"  {failed_ideas} failed ideas -- running phase 2 to replace")
                    run_phase2(args.llm, args.embed)

            if 3 in phases_to_run:
                t0 = time.time()
                run_phase3(args.llm)
                state[f"phase3_iter{iteration}_s"] = round(time.time() - t0, 1)
                save_state(state)

            if 4 in phases_to_run:
                t0 = time.time()
                run_phase4(args.llm, args.embed)
                state[f"phase4_iter{iteration}_s"] = round(time.time() - t0, 1)
                save_state(state)

            total_have, total_need = print_progress(taxonomy)
            state["total_validated"] = total_have
            save_state(state)

            if total_have >= total_need:
                log.info(f"  [OK] Target reached after iteration {iteration}!")
                break

            log.info(f"  {total_need - total_have:,} still needed - next iteration...")
        else:
            log.warning(f"Max iterations ({args.max_iter}) reached. Dataset may be incomplete.")

        for p in [3, 4]:
            if p not in state["phases_completed"]:
                state["phases_completed"].append(p)
        save_state(state)

    # -- Final summary -------------------------------------------------------
    print("""
+======================================================+
|  Run complete                                        |
+======================================================+
""")
    total_have, total_need = print_progress(taxonomy)
    state["total_validated"] = total_have
    state["finished_at"]     = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"  Output directory : {OUTPUT_DIR.resolve()}")
    print(f"  State file       : {STATE_PATH}")
    print(f"  Main log         : {MAIN_LOG}")

    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"  Database         : {DB_PATH}  ({size_mb:.1f} MB)")

    for label, path in [("Dataset JSONL", OUTPUT_DIR / "dataset.jsonl"),
                        ("HuggingFace dir", OUTPUT_DIR / "dataset_hf")]:
        if path.exists():
            print(f"  {label:<17}: {path}" + ("/" if path.is_dir() else ""))

    print()
    if total_have >= total_need:
        print(f"  [OK] SUCCESS - {total_have:,} / {total_need:,} pairs generated and validated.")
    else:
        pct = total_have / total_need * 100
        print(f"  [!!] PARTIAL  - {total_have:,} / {total_need:,} ({pct:.1f}%).")
        print("    Re-run with --resume to continue filling gaps.")
    print()


if __name__ == "__main__":
    main()
