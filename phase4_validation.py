"""
Phase 3 -- Validation and Deduplication (Ollama)
Runs four checks on every raw snippet:
  1. Embedding-based deduplication  (Ollama embed model)
  2. C compilation                  (gcc, with iterative LLM correction)
  3. Rust compilation               (rustc, with iterative LLM correction)
  4. Semantic equivalence judge     (Ollama LLM)

When a snippet fails a compile check, the compiler error is fed back to the
LLM which attempts to fix the code. This repeats up to MAX_FIX_ATTEMPTS times
before the snippet is finally rejected.

Outputs:
  Updates status column in output/snippets.db
  output/embeddings.db        -- vector store (cosine similarity via numpy)
  output/validation_report.json

Requirements:
  pip install llama-cpp-python numpy
  Models configured via llm_backend.py
"""

import json
import logging
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from chat_logger import llm_chat_logged, embed_logged, code_embed_logged

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH          = Path("output/snippets.db")
EMBED_DB_PATH    = Path("output/embeddings.db")
REPORT_PATH      = Path("output/validation_report.json")

# Qwen3-Embedding-8B is used for code deduplication -- threshold raised
COSINE_THRESHOLD = 0.95

# How many times to ask the LLM to fix a compile error before giving up.
# Each attempt sends the compiler error + current code back to the model.
MAX_FIX_ATTEMPTS = 3

# Context/output limits for correction calls -- code + error fits in 4096 easily
FIX_NUM_CTX      = 8192
FIX_NUM_PREDICT  = 4096

# -c = compile only, no linking -- avoids all linker errors (WinMain, undefined refs etc.)
# We only need to know if the code is valid C, not if it links into an executable.
GCC_FLAGS   = ["-Wall", "-Wextra", "-Werror", "-std=c11", "-x", "c", "-c", "-"]
# --emit=metadata skips code generation and linking -- only checks syntax/types
RUSTC_FLAGS = ["--edition", "2021", "--crate-type", "lib", "--emit=metadata", "-"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/phase4.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Database ──────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for col, typedef in [
        ("c_compile_ok",      "INTEGER"),
        ("rust_compile_ok",   "INTEGER"),
        ("is_duplicate",      "INTEGER DEFAULT 0"),
        ("is_equivalent",     "INTEGER"),
        ("rejection_reason",  "TEXT"),
        ("validated_at",      "TEXT"),
        ("c_fix_attempts",    "INTEGER DEFAULT 0"),
        ("rust_fix_attempts", "INTEGER DEFAULT 0"),
        # Store the (possibly corrected) code that finally passed or was rejected
        ("final_c_code",      "TEXT"),
        ("final_rust_code",   "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE raw_snippets ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def get_embed_db() -> sqlite3.Connection:
    conn = sqlite3.connect(EMBED_DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            snippet_id  TEXT PRIMARY KEY,
            vector_blob BLOB NOT NULL
        );
    """)
    conn.commit()
    return conn


# ── Check 1: Embedding deduplication ─────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    vec  = code_embed_logged(text[:4096])
    arr  = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def load_existing_embeddings(econn: sqlite3.Connection) -> list[tuple[str, np.ndarray]]:
    rows = econn.execute("SELECT snippet_id, vector_blob FROM embeddings").fetchall()
    return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]


def check_duplicate_embedding(
    snippet_id: str,
    c_code: str,
    rust_code: str,
    existing: list[tuple[str, np.ndarray]],
    econn: sqlite3.Connection,
) -> tuple[bool, str]:
    combined = f"C:\n{c_code}\n\nRust:\n{rust_code}"
    vec = get_embedding(combined)

    for prev_id, prev_vec in existing:
        if cosine_similarity(vec, prev_vec) >= COSINE_THRESHOLD:
            return True, prev_id

    econn.execute(
        "INSERT OR REPLACE INTO embeddings (snippet_id, vector_blob) VALUES (?,?)",
        (snippet_id, vec.tobytes()),
    )
    econn.commit()
    existing.append((snippet_id, vec))
    return False, ""


# ── Check 2: C compilation with iterative LLM correction ─────────────────────

C_FIX_PROMPT = """\
The following C code failed to compile. Fix it so it compiles cleanly with:
gcc -Wall -Wextra -Werror -std=c11

Compiler error:
{error}

Current code:
{code}

Rules:
- Return ONLY the corrected C code, no explanation, no markdown, no code fences
- Preserve the original intent and logic exactly
- Include all necessary #include headers
- Do NOT add a main() function under any circumstances -- the code is compiled as a library
- Do NOT add a WinMain() function
- Only fix the actual compile errors shown, do not restructure the code
"""


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def _run_gcc(c_code: str) -> tuple[bool, str]:
    """Run gcc and return (success, error_text)."""
    import tempfile, os
    preamble = ""
    if "#include" not in c_code:
        preamble = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
    # Output to a .o object file -- we are compile-only, no linking needed
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["gcc"] + GCC_FLAGS + ["-o", tmp_path],
            input=preamble + c_code,
            capture_output=True, text=True, timeout=15,
        )
        return (True, "") if result.returncode == 0 else (False, result.stderr[:600])
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _llm_fix_c(code: str, error: str, context_id: str = "", attempt: int = 1) -> str:
    """Ask the LLM to fix a C compile error. Returns the corrected code."""
    prompt = C_FIX_PROMPT.format(error=error, code=code)
    fixed = llm_chat_logged(
        phase="4_fix_c",
        context_id=context_id,
        prompt=prompt,
        temperature=0.2,
        max_tokens=FIX_NUM_PREDICT,
        note=f"fix attempt {attempt}/{MAX_FIX_ATTEMPTS}",
    )
    fixed = fixed.strip()
    # Strip any accidental markdown fences
    if "```" in fixed:
        parts = fixed.split("```")
        candidates = [p.lstrip("c").lstrip("cpp").strip() for p in parts[1::2]]
        fixed = max(candidates, key=len) if candidates else fixed
    return fixed.strip()


def check_c_compiles_with_correction(c_code: str, context_id: str = "") -> tuple[bool, str, str, int]:
    """
    Try to compile C code, asking the LLM to fix errors up to MAX_FIX_ATTEMPTS times.
    Returns (success, final_error, final_code, fix_attempts_used).
    """
    if not _tool_available("gcc"):
        return True, "", c_code, 0

    ok, err = _run_gcc(c_code)
    if ok:
        return True, "", c_code, 0

    # Print the initial compile error in full before attempting fixes
    log.info("    C compile error:")
    for line in err.strip().splitlines():
        log.info(f"      {line}")

    # Attempt iterative correction
    current_code = c_code
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        log.info(f"    C fix attempt {attempt}/{MAX_FIX_ATTEMPTS} ...")
        try:
            fixed_code = _llm_fix_c(current_code, err, context_id=context_id, attempt=attempt)
        except Exception as e:
            log.warning(f"    LLM fix call failed: {e}")
            break

        ok, err = _run_gcc(fixed_code)
        current_code = fixed_code
        if ok:
            log.info(f"    C fixed after {attempt} attempt(s)")
            return True, "", current_code, attempt

        # Print the new error so progress (or lack of it) is visible
        log.info(f"    C still failing after fix attempt {attempt}:")
        for line in err.strip().splitlines():
            log.info(f"      {line}")

    log.info("    C correction failed -- snippet will be rejected")
    return False, err[:300], current_code, MAX_FIX_ATTEMPTS


# ── Check 3: Rust compilation with iterative LLM correction ──────────────────

RUST_FIX_PROMPT = (
    "The following Rust code failed to compile. Fix it so it compiles cleanly with:\n"
    "rustc --edition 2021 --crate-type lib\n\n"
    "Compiler error:\n{error}\n\n"
    "{explanation}"
    "Current code:\n{code}\n\n"
    "Rules:\n"
    "- Return ONLY the corrected Rust code, no explanation, no markdown, no code fences\n"
    "- Preserve the original intent and logic exactly\n"
    "- Include all necessary use statements\n"
    "- Do NOT add a main() function under any circumstances -- compiled as a library\n"
    "- Do NOT add a WinMain() function\n"
    "- Only fix the actual compile errors shown, do not restructure the code\n"
    "- Add #[allow(dead_code, unused_variables, unused_imports)] at the top if needed\n"
    "- If a method is not found on &mut str, convert to String first\n"
)


def _rustc_explain(error_text: str) -> str:
    """
    Extract error codes like E0599 from rustc output and run rustc --explain
    on each. Returns combined explanation text, or empty string if none found.
    """
    codes = list(dict.fromkeys(re.findall(r"E\d{4}", error_text)))
    if not codes:
        return ""
    explanations = []
    for code in codes[:3]:
        try:
            result = subprocess.run(
                ["rustc", "--explain", code],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                explanations.append(
                    f"rustc --explain {code}:\n{result.stdout.strip()[:400]}"
                )
        except Exception:
            pass
    return "\n\n".join(explanations)


def _run_rustc(rust_code: str) -> tuple[bool, str]:
    """Run rustc and return (success, error_text)."""
    import tempfile, os
    preamble = ""
    if "fn main" not in rust_code and "mod " not in rust_code:
        preamble = "#[allow(dead_code, unused_variables, unused_imports)]\n"
    # Use a real temp file for output -- /dev/null does not exist on Windows
    with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["rustc"] + RUSTC_FLAGS + ["-o", tmp_path],
            input=preamble + rust_code,
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            return True, ""
        errors = [l for l in result.stderr.splitlines() if "error" in l.lower()]
        return False, "\n".join(errors[:8])
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _llm_fix_rust(code: str, error: str, context_id: str = "", attempt: int = 1) -> str:
    """Ask the LLM to fix a Rust compile error. Returns the corrected code."""
    explanation = _rustc_explain(error)
    explanation_block = f"Rustc explanation:\n{explanation}\n\n" if explanation else ""
    prompt = RUST_FIX_PROMPT.format(error=error, explanation=explanation_block, code=code)
    fixed = llm_chat_logged(
        phase="4_fix_rust",
        context_id=context_id,
        prompt=prompt,
        temperature=0.2,
        max_tokens=FIX_NUM_PREDICT,
        note=f"fix attempt {attempt}/{MAX_FIX_ATTEMPTS}",
    )
    fixed = fixed.strip()
    if "```" in fixed:
        parts = fixed.split("```")
        candidates = [p.lstrip("rust").strip() for p in parts[1::2]]
        fixed = max(candidates, key=len) if candidates else fixed
    return fixed.strip()


def check_rust_compiles_with_correction(rust_code: str, context_id: str = "") -> tuple[bool, str, str, int]:
    """
    Try to compile Rust code, asking the LLM to fix errors up to MAX_FIX_ATTEMPTS times.
    Returns (success, final_error, final_code, fix_attempts_used).
    """
    if not _tool_available("rustc"):
        return True, "", rust_code, 0

    ok, err = _run_rustc(rust_code)
    if ok:
        return True, "", rust_code, 0

    # Print the initial compile error in full before attempting fixes
    log.info("    Rust compile error:")
    for line in err.strip().splitlines():
        log.info(f"      {line}")

    current_code = rust_code
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        log.info(f"    Rust fix attempt {attempt}/{MAX_FIX_ATTEMPTS} ...")
        try:
            fixed_code = _llm_fix_rust(current_code, err, context_id=context_id, attempt=attempt)
        except Exception as e:
            log.warning(f"    LLM fix call failed: {e}")
            break

        ok, err = _run_rustc(fixed_code)
        current_code = fixed_code
        if ok:
            log.info(f"    Rust fixed after {attempt} attempt(s)")
            return True, "", current_code, attempt

        # Print the new error so progress (or lack of it) is visible
        log.info(f"    Rust still failing after fix attempt {attempt}:")
        for line in err.strip().splitlines():
            log.info(f"      {line}")

    log.info("    Rust correction failed -- snippet will be rejected")
    return False, err[:300], current_code, MAX_FIX_ATTEMPTS


# ── Check 4: Semantic equivalence (Ollama LLM judge) ─────────────────────────

EQUIVALENCE_PROMPT = """\
You are a strict code review judge.

Determine whether the C snippet and the Rust snippet below are SEMANTICALLY \
EQUIVALENT -- meaning they implement the same algorithm and would produce \
identical results for all valid inputs.

C:
{c_code}

Rust:
{rust_code}

Reply with exactly one word on the first line:
  EQUIVALENT
  NOT_EQUIVALENT
  UNCERTAIN

Then on the next line, a single sentence explanation (max 20 words).
No other output.
"""


def check_semantic_equivalence(c_code: str, rust_code: str, context_id: str = "") -> tuple[bool, str]:
    try:
        prompt = EQUIVALENCE_PROMPT.format(
            c_code=c_code[:2000],
            rust_code=rust_code[:2000],
        )
        text = llm_chat_logged(
            phase="4_equivalence",
            context_id=context_id,
            prompt=prompt,
            temperature=0.0,
            max_tokens=128,
        )
        text = text.strip()
        lines   = text.splitlines()
        verdict = lines[0].strip().upper()
        note    = lines[1].strip() if len(lines) > 1 else ""

        if "EQUIVALENT" in verdict and "NOT" not in verdict:
            return True, note
        elif "NOT_EQUIVALENT" in verdict:
            return False, note
        else:
            return True, f"UNCERTAIN: {note}"
    except Exception as e:
        log.warning(f"  Equivalence check error: {e}")
        return True, "check skipped (error)"


# ── Lint helpers (non-blocking) ───────────────────────────────────────────────

def cppcheck_lite(c_code: str) -> list[str]:
    warnings = []
    if re.search(r"\bgets\s*\(", c_code):
        warnings.append("uses gets() -- buffer overflow risk")
    if re.search(r"\bstrcpy\s*\(", c_code) and "strncpy" not in c_code:
        warnings.append("uses strcpy() without bounds")
    if re.search(r"\bsprintf\s*\(", c_code) and "snprintf" not in c_code:
        warnings.append("uses sprintf() -- prefer snprintf")
    if re.search(r"\bmalloc\s*\(", c_code) and "free(" not in c_code:
        warnings.append("malloc() without free() -- possible leak")
    return warnings


def clippy_lite(rust_code: str) -> list[str]:
    warnings = []
    if rust_code.count("unsafe") > 0:
        warnings.append(f"contains {rust_code.count('unsafe')} unsafe block(s)")
    for pat in [".unwrap()", "panic!("]:
        if pat in rust_code:
            warnings.append(f"uses {pat}")
    return warnings


# ── Main validation loop ──────────────────────────────────────────────────────

def validate_all(conn: sqlite3.Connection, econn: sqlite3.Connection) -> dict:
    pending = conn.execute(
        "SELECT * FROM raw_snippets WHERE status='raw' ORDER BY rowid"
    ).fetchall()

    if not pending:
        log.info("No raw snippets to validate.")
        return {
            "total": 0, "validated": 0, "duplicate": 0,
            "c_compile_fail": 0, "rust_compile_fail": 0, "not_equivalent": 0,
            "c_fixes_applied": 0, "rust_fixes_applied": 0,
        }

    log.info(f"Validating {len(pending)} raw snippets")
    from llm_backend import LLM_MODEL_PATH, CODE_EMBED_MODEL_PATH
    log.info(f"LLM model        : {LLM_MODEL_PATH}")
    log.info(f"Code embed model : {CODE_EMBED_MODEL_PATH}")
    log.info(f"Max fix attempts : {MAX_FIX_ATTEMPTS} per language")

    existing_embeddings = load_existing_embeddings(econn)
    log.info(f"Loaded {len(existing_embeddings)} existing embeddings")

    stats = dict(
        total=len(pending), validated=0, duplicate=0,
        c_compile_fail=0, rust_compile_fail=0, not_equivalent=0,
        c_fixes_applied=0, rust_fixes_applied=0,
    )

    for i, row in enumerate(pending):
        sid = row["id"]
        log.info(f"[{i+1}/{len(pending)}] {sid}")

        reason        = None
        c_ok          = None
        rust_ok       = None
        is_equiv      = None
        c_fix_count   = 0
        rust_fix_count = 0

        # Use the original code as starting point; may be replaced by fixed version
        c_code    = row["c_code"]
        rust_code = row["rust_code"]

        # 1. Embedding deduplication
        is_dup, matched_id = check_duplicate_embedding(
            sid, c_code, rust_code, existing_embeddings, econn
        )
        if is_dup:
            reason = f"duplicate of {matched_id}"
            stats["duplicate"] += 1
        else:
            # 2. C compile (with correction loop)
            c_ok, c_err, c_code, c_fix_count = check_c_compiles_with_correction(c_code, context_id=sid)
            if c_fix_count > 0:
                stats["c_fixes_applied"] += c_fix_count
            if not c_ok:
                reason = f"c_compile_fail: {c_err[:120]}"
                stats["c_compile_fail"] += 1
            else:
                # 3. Rust compile (with correction loop)
                rust_ok, rust_err, rust_code, rust_fix_count = check_rust_compiles_with_correction(rust_code, context_id=sid)
                if rust_fix_count > 0:
                    stats["rust_fixes_applied"] += rust_fix_count
                if not rust_ok:
                    reason = f"rust_compile_fail: {rust_err[:120]}"
                    stats["rust_compile_fail"] += 1
                else:
                    # 4. Semantic equivalence (use final corrected code)
                    is_equiv, note = check_semantic_equivalence(c_code, rust_code, context_id=sid)
                    if not is_equiv:
                        reason = f"not_equivalent: {note}"
                        stats["not_equivalent"] += 1

        lint = "; ".join(cppcheck_lite(c_code) + clippy_lite(rust_code)) or None
        new_status = "rejected" if reason else "validated"
        if new_status == "validated":
            stats["validated"] += 1

        conn.execute(
            """UPDATE raw_snippets SET
                status=?, c_compile_ok=?, rust_compile_ok=?,
                is_duplicate=?, is_equivalent=?,
                rejection_reason=?, validated_at=datetime('now'),
                c_fix_attempts=?, rust_fix_attempts=?,
                final_c_code=?, final_rust_code=?
               WHERE id=?""",
            (
                new_status,
                1 if c_ok else (0 if c_ok is False else None),
                1 if rust_ok else (0 if rust_ok is False else None),
                1 if is_dup else 0,
                1 if is_equiv else (0 if is_equiv is False else None),
                reason or lint,
                c_fix_count,
                rust_fix_count,
                c_code,      # store final (possibly corrected) code
                rust_code,
                sid,
            ),
        )
        conn.commit()

        icon = "[OK]" if new_status == "validated" else "[X]"
        fix_note = ""
        if c_fix_count or rust_fix_count:
            fix_note = f" (fixes: C={c_fix_count} Rust={rust_fix_count})"
        log.info(f"  {icon} {new_status}{fix_note}" + (f" -- {reason}" if reason else ""))

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conn  = get_db()
    econn = get_embed_db()

    stats = validate_all(conn, econn)

    print("\n-- Validation summary --")
    for k, v in stats.items():
        print(f"  {k:<25} {v}")

    total = max(stats["total"], 1)
    report = {
        "stats":      stats,
        "pass_rate":  round(stats["validated"] / total * 100, 1),
        "fix_rate": {
            "c_fixes_applied":    stats["c_fixes_applied"],
            "rust_fixes_applied": stats["rust_fixes_applied"],
        },
        "rejection_breakdown": {
            "duplicate":         stats["duplicate"],
            "c_compile_fail":    stats["c_compile_fail"],
            "rust_compile_fail": stats["rust_compile_fail"],
            "not_equivalent":    stats["not_equivalent"],
        },
        "category_breakdown": {},
    }

    for r in conn.execute(
        "SELECT category, status, COUNT(*) as n FROM raw_snippets GROUP BY category, status"
    ).fetchall():
        report["category_breakdown"].setdefault(r["category"], {})[r["status"]] = r["n"]

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info(f"Report written to {REPORT_PATH}")

    print("\n-- Validated pairs by category --")
    for r in conn.execute(
        "SELECT category, COUNT(*) as n FROM raw_snippets WHERE status='validated' GROUP BY category"
    ).fetchall():
        print(f"  {r['category']:<30} {r['n']:>4}")

    conn.close()
    econn.close()


if __name__ == "__main__":
    main()
