"""
Phase 2b -- Code Generation from Ideas
Reads ideas from the `ideas` table (written by phase 2a) and generates
a C/Rust code pair for each one.

If the LLM cannot implement an idea after MAX_RETRIES attempts, the idea is
marked 'failed' so main.py can request a replacement idea from phase 2a.

Outputs:
  output/snippets.db  (raw_snippets table, ideas.status updated)
  output/generation_log.jsonl

Requirements:
  pip install ollama json-repair
  Models configured via llm_backend.py
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

from chat_logger import llm_chat_logged

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH       = Path("output/snippets.db")
LOG_PATH      = Path("output/generation_log.jsonl")

# Model paths configured in llm_backend.py
BATCH_SIZE    = 1    # pairs per API call -- keep at 1 for small models
MAX_RETRIES   = 3
RETRY_DELAY   = 4
REQUEST_DELAY = 0.2
NUM_CTX       = 8192
NUM_PREDICT   = 4096

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/phase3.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Database ──────────────────────────────────────────────────────────────────

SNIPPETS_DDL = """
CREATE TABLE IF NOT EXISTS raw_snippets (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    description     TEXT,
    c_code          TEXT NOT NULL,
    rust_code       TEXT NOT NULL,
    idea_id         TEXT,
    generated_at    TEXT DEFAULT (datetime('now')),
    status          TEXT DEFAULT 'raw'
);
CREATE INDEX IF NOT EXISTS idx_category ON raw_snippets(category);
CREATE INDEX IF NOT EXISTS idx_status   ON raw_snippets(status);
"""


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SNIPPETS_DDL)
    conn.commit()
    return conn


def get_pending_ideas(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all ideas that don't yet have generated code."""
    return conn.execute(
        "SELECT * FROM ideas WHERE status='pending' ORDER BY rowid"
    ).fetchall()


def mark_idea(conn: sqlite3.Connection, idea_id: str, status: str):
    conn.execute("UPDATE ideas SET status=? WHERE id=?", (status, idea_id))
    conn.commit()


# ── Prompt ────────────────────────────────────────────────────────────────────

DIFFICULTY_INSTRUCTIONS = {
    "beginner":     "Short (10-25 lines), single concept, clear and simple.",
    "intermediate": "25-60 lines, multiple related concepts, realistic error checking.",
    "advanced":     "40-100 lines, non-trivial patterns, may use unsafe or advanced idioms.",
}

CODE_PROMPT = """\
Generate a paired C and Rust code snippet that implements the following:

  Concept   : {description}
  Category  : {category_name}
  Difficulty: {difficulty} -- {difficulty_instruction}

Rules:
- The C snippet must compile cleanly with: gcc -Wall -Werror -std=c11
- The Rust snippet must compile cleanly with: cargo check (edition 2021)
- The pair must be semantically equivalent (same logic, same observable behaviour)
- Include all necessary #include headers or use statements
- Do NOT wrap in main() unless the concept specifically requires program entry
- Do NOT add a WinMain() function

Respond with a single JSON object ONLY. No markdown, no code fences, no explanation.
The object must have exactly these keys:
  "id"          : "{idea_id}"
  "category"    : "{category_slug}"
  "difficulty"  : "{difficulty}"
  "description" : "{description}"
  "c_code"      : the complete C snippet as a string
  "rust_code"   : the complete Rust snippet as a string
"""


def render_prompt(idea: sqlite3.Row) -> str:
    return CODE_PROMPT.format(
        description=idea["description"],
        category_name=idea["category"].replace("_", " ").title(),
        category_slug=idea["category"],
        difficulty=idea["difficulty"],
        difficulty_instruction=DIFFICULTY_INSTRUCTIONS[idea["difficulty"]],
        idea_id=idea["id"],
    )


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        candidates = [p.lstrip("json").strip() for p in parts[1::2]]
        text = max(candidates, key=len) if candidates else text
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()


def _parse_object(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if HAS_JSON_REPAIR:
        try:
            result = json.loads(repair_json(raw))
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    raise ValueError("Could not parse a valid JSON object from model response")


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, context_id: str = "") -> dict:
    raw_text = llm_chat_logged(
        phase="3_generation",
        context_id=context_id,
        prompt=prompt,
        temperature=0.7,
        max_tokens=NUM_PREDICT,
    )
    raw = _extract_json_object(raw_text)
    obj = _parse_object(raw)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")

    required = {"id", "category", "difficulty", "description", "c_code", "rust_code"}
    missing  = required - obj.keys()
    if missing:
        raise ValueError(f"Response missing keys: {missing}")

    return obj


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_code_for_idea(conn: sqlite3.Connection, idea: sqlite3.Row) -> bool:
    """
    Generate C/Rust code for one idea. Returns True if successfully inserted.
    Marks the idea as 'code_generated' or 'failed' in the ideas table.
    """
    prompt     = render_prompt(idea)
    snippet    = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snippet = call_ollama(prompt, context_id=idea["id"])
            break
        except Exception as e:
            last_error = str(e)
            log.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    if snippet is None:
        log.warning(f"  Code generation failed for idea {idea['id']}: {last_error}")
        mark_idea(conn, idea["id"], "failed")
        return False

    # Use the idea's id as the snippet id for traceability
    sid = idea["id"]

    try:
        conn.execute(
            "INSERT OR IGNORE INTO raw_snippets "
            "(id, category, difficulty, description, c_code, rust_code, idea_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (sid, snippet["category"], snippet["difficulty"],
             snippet.get("description", idea["description"]),
             snippet["c_code"], snippet["rust_code"], idea["id"]),
        )
        inserted = conn.execute("SELECT changes()").fetchone()[0]
    except sqlite3.Error as e:
        log.warning(f"  DB insert error: {e}")
        mark_idea(conn, idea["id"], "failed")
        return False

    if inserted:
        mark_idea(conn, idea["id"], "code_generated")
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "idea_id":    idea["id"],
                "category":   idea["category"],
                "difficulty": idea["difficulty"],
                "description": idea["description"],
            }) + "\n")
        return True
    else:
        # Snippet ID already exists (duplicate idea slipped through)
        mark_idea(conn, idea["id"], "code_generated")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("output").mkdir(exist_ok=True)
    LOG_PATH.touch()

    conn = get_db()

    pending = get_pending_ideas(conn)
    total   = conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]
    done    = total - len(pending)

    from llm_backend import LLM_MODEL_PATH
    log.info(f"LLM model      : {LLM_MODEL_PATH}")
    from llm_backend import N_CTX
    log.info(f"Context window : {N_CTX} tokens")
    log.info(f"Total ideas    : {total}")
    log.info(f"Already coded  : {done}")
    log.info(f"Pending        : {len(pending)}")

    succeeded  = 0
    failed_ids = []

    for i, idea in enumerate(pending):
        log.info(f"[{i+1}/{len(pending)}] {idea['id']}")
        log.info(f"  Concept: {idea['description']}")

        ok = generate_code_for_idea(conn, idea)

        if ok:
            succeeded += 1
            log.info(f"  [+] Code generated  (total coded: {done + succeeded}/{total})")
        else:
            failed_ids.append(idea["id"])
            log.warning(f"  [!] Failed -- idea will be flagged for replacement")

        time.sleep(REQUEST_DELAY)

    log.info(f"Phase 2b complete.")
    log.info(f"  Succeeded : {succeeded}")
    log.info(f"  Failed    : {len(failed_ids)}")

    if failed_ids:
        log.info(f"  Failed idea IDs: {failed_ids[:10]}" +
                 (" ..." if len(failed_ids) > 10 else ""))

    print("\n-- Code generation summary --")
    for r in conn.execute(
        "SELECT category, difficulty, COUNT(*) as n FROM raw_snippets "
        "GROUP BY category, difficulty ORDER BY category, difficulty"
    ).fetchall():
        print(f"  {r['category']:<30} {r['difficulty']:<14} {r['n']:>4}")

    conn.close()


if __name__ == "__main__":
    main()
