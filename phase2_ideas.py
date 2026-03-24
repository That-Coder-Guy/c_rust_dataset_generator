"""
Phase 2a -- Unique Idea Generation
Generates 5,000 unique snippet concept descriptions BEFORE any code is written.
Uniqueness is enforced via cosine similarity on the embedding model so that
every idea in the dataset is semantically distinct.

Each idea names a concept that suits C and also admits a clear Rust
implementation (idiomatic, mostly safe). Ideas are stored in the `ideas` table and consumed
by phase 2b which generates the actual code.

Outputs:
  output/snippets.db   (ideas table)
  output/idea_embeddings.db

Requirements:
  pip install ollama numpy
  Models configured via llm_backend.py
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
from chat_logger import llm_chat_logged, embed_logged

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH          = Path("output/snippets.db")
IDEA_EMBED_PATH  = Path("output/idea_embeddings.db")
TAXONOMY_PATH    = Path("output/taxonomy.json")

# Model paths are configured in llm_backend.py

# Cosine similarity above this = duplicate idea, generate a new one instead
IDEA_SIMILARITY_THRESHOLD = 0.92

MAX_RETRIES      = 5    # attempts to get a unique idea before skipping a slot
RETRY_DELAY      = 2
REQUEST_DELAY    = 0.1
NUM_CTX          = 4096  # ideas are short, small context is fine
NUM_PREDICT      = 256   # one sentence needs very few tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/phase2.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Database ──────────────────────────────────────────────────────────────────

IDEAS_DDL = """
CREATE TABLE IF NOT EXISTS ideas (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    description     TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',  -- pending | code_generated | failed
    job_index       INTEGER,
    generated_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ideas_category ON ideas(category);
CREATE INDEX IF NOT EXISTS idx_ideas_status   ON ideas(status);
"""


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(IDEAS_DDL)
    conn.commit()
    return conn


def get_embed_db() -> sqlite3.Connection:
    conn = sqlite3.connect(IDEA_EMBED_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS idea_embeddings (
            idea_id     TEXT PRIMARY KEY,
            vector_blob BLOB NOT NULL
        );
    """)
    conn.commit()
    return conn


# ── Embedding helpers ─────────────────────────────────────────────────────────

def get_embedding(text: str) -> np.ndarray:
    vec = embed_logged(text[:2048])
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def load_existing_embeddings(econn: sqlite3.Connection) -> list[tuple[str, np.ndarray]]:
    rows = econn.execute(
        "SELECT idea_id, vector_blob FROM idea_embeddings"
    ).fetchall()
    return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]


def is_duplicate_idea(
    description: str,
    existing: list[tuple[str, np.ndarray]],
) -> tuple[bool, str]:
    """Check if this description is too similar to any existing idea."""
    vec = get_embedding(description)
    for prev_id, prev_vec in existing:
        sim = float(np.dot(vec, prev_vec))
        if sim >= IDEA_SIMILARITY_THRESHOLD:
            return True, prev_id, vec
    return False, "", vec


def store_embedding(
    idea_id: str,
    vec: np.ndarray,
    existing: list[tuple[str, np.ndarray]],
    econn: sqlite3.Connection,
):
    econn.execute(
        "INSERT OR REPLACE INTO idea_embeddings (idea_id, vector_blob) VALUES (?,?)",
        (idea_id, vec.tobytes()),
    )
    econn.commit()
    existing.append((idea_id, vec))


# ── Prompt ────────────────────────────────────────────────────────────────────

DIFFICULTY_INSTRUCTIONS = {
    "beginner":     "10-25 lines, single concept, clear and simple.",
    "intermediate": "25-60 lines, multiple related concepts, realistic error checking.",
    "advanced":     "40-100 lines, non-trivial patterns, unsafe or advanced idioms.",
}

IDEA_PROMPT = """\
You are curating a dataset of paired C and Rust code snippets. Name concepts that are
natural to implement in C and that can also be written cleanly in Rust (idiomatic where
possible; avoid ideas that would only translate as wall-to-wall unsafe Rust).

Generate ONE unique idea for such a pair in this category:
  Category   : {category_name}
  Description: {category_description}
  Difficulty : {difficulty} ({difficulty_instruction})

The idea must be different from these already-used concepts:
{seen_descriptions}

Rules:
- Respond with a SINGLE noun phrase (max 20 words) naming the concept the snippet demonstrates
- Start with "A", "An", or "The" -- never start with a verb or gerund (e.g. NOT "Implementing...", NOT "Using...")
- Be specific -- name the function, pattern, or concept (e.g. "A bounded string copy using strncpy with a fixed-size destination buffer")
- Do NOT generate ideas that require external network access, file system I/O, or interactive user input
- No explanation, no preamble, no punctuation at the end beyond a period
- Return the noun phrase only
"""


def render_idea_prompt(job: dict, seen_descriptions: list[str]) -> str:
    seen_str = "\n".join(f"  - {d}" for d in seen_descriptions[-15:]) or "  (none yet)"
    return IDEA_PROMPT.format(
        category_name=job["category_name"],
        category_description=job["category_description"],
        difficulty=job["difficulty"],
        difficulty_instruction=DIFFICULTY_INSTRUCTIONS[job["difficulty"]],
        seen_descriptions=seen_str,
    )


# ── Idea validation ──────────────────────────────────────────────────────────

BANNED_WORDS    = {"c", "rust"}          # language names
BANNED_STARTS   = ("implementing", "using", "creating", "demonstrating",
                   "showing", "writing", "building", "defining", "calling",
                   "working", "handling", "managing", "performing")
REQUIRED_STARTS = ("a ", "an ", "the ")  # must be a noun phrase


def validate_idea(text: str) -> tuple[bool, str]:
    """
    Check that an idea description meets content rules.
    Returns (is_valid, reason_if_invalid).
    """
    if len(text) < 10:
        return False, "too short"

    if "`" in text or "```" in text:
        return False, "contains backticks/code fragments"

    lower = text.lower()

    # Must not contain language names as standalone words
    import re
    for word in BANNED_WORDS:
        if re.search(rf"\b{word}\b", lower):
            return False, f"contains banned word '{word}'"

    # Must start with a noun phrase article
    if not any(lower.startswith(s) for s in REQUIRED_STARTS):
        return False, f"does not start with A/An/The (starts with: {text.split()[0]!r})"

    # Must not start with a gerund/verb
    first_word = lower.split()[0] if lower.split() else ""
    # Skip the article and check the next word
    words = lower.split()
    check_word = words[1] if len(words) > 1 else ""
    if check_word.endswith("ing") or any(lower.startswith(s) for s in BANNED_STARTS):
        return False, f"starts with a gerund/verb phrase"

    return True, ""


# ── Idea generation ───────────────────────────────────────────────────────────

def generate_idea(prompt: str, context_id: str = "") -> str:
    """Call the LLM and return a single cleaned description string."""
    text = llm_chat_logged(
        phase="2_ideas",
        context_id=context_id,
        prompt=prompt,
        temperature=0.9,
        max_tokens=NUM_PREDICT,
    )
    text = text.strip()
    # Strip quotes, bullets, numbering, newlines
    text = text.strip('"\'').strip()
    text = text.splitlines()[0].strip()
    # Remove leading list markers
    import re
    text = re.sub(r"^[\d\-\*\.\)]+\s*", "", text).strip()
    return text


def get_existing_ideas(conn: sqlite3.Connection) -> set[int]:
    """Return set of job_indexes that already have an idea."""
    rows = conn.execute("SELECT job_index FROM ideas WHERE job_index IS NOT NULL").fetchall()
    return {r[0] for r in rows}


def get_recent_descriptions(conn: sqlite3.Connection, category: str, limit: int = 15) -> list[str]:
    rows = conn.execute(
        "SELECT description FROM ideas WHERE category=? ORDER BY rowid DESC LIMIT ?",
        (category, limit),
    ).fetchall()
    return [r[0] for r in rows]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("output").mkdir(exist_ok=True)

    taxonomy     = json.loads(TAXONOMY_PATH.read_text())
    jobs         = taxonomy["jobs"]
    total_target = taxonomy["total_target"]

    from llm_backend import LLM_MODEL_PATH, EMBED_MODEL_PATH
    log.info(f"LLM model   : {LLM_MODEL_PATH}")
    log.info(f"NLP embed   : {EMBED_MODEL_PATH}")
    log.info(f"Target ideas: {total_target}")
    log.info(f"Dedup threshold: {IDEA_SIMILARITY_THRESHOLD}")

    conn  = get_db()
    econn = get_embed_db()

    existing_jobs        = get_existing_ideas(conn)
    existing_embeddings  = load_existing_embeddings(econn)
    total_stored         = len(existing_jobs)

    log.info(f"Resuming - {total_stored} ideas already stored, "
             f"{len(existing_embeddings)} embeddings loaded")

    # ── Persistent queue -- every slot is filled, guaranteed ────────────────
    # Failed slots go to the back of the queue and are retried after all
    # currently pending slots. The loop exits only when every slot is filled.
    from collections import deque

    queue: deque[tuple[int, dict, int]] = deque()
    for job_index, job in enumerate(jobs):
        if job_index not in existing_jobs:
            queue.append((job_index, job, 0))   # (job_index, job, rounds_attempted)

    log.info(f"{len(queue)} idea slot(s) queued")

    while queue:
        job_index, job, rounds = queue.popleft()

        cat  = job["category_slug"]
        diff = job["difficulty"]
        rounds += 1

        if rounds == 1:
            log.info(f"[{job_index+1}/{len(jobs)}] {cat} / {diff}")
        else:
            log.info(f"[re-queue round {rounds}] {cat} / {diff}  "
                     f"({len(queue)} slots still pending)")

        seen_descs       = get_recent_descriptions(conn, cat)
        idea_description = None
        vec              = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                context   = f"{cat}_{diff}_{job['job_index_within_cat']:04d}"
                prompt    = render_idea_prompt(job, seen_descs)
                candidate = generate_idea(prompt, context_id=context)

                if len(candidate) < 10:
                    log.warning(f"  Attempt {attempt}: too short: {repr(candidate)}")
                    continue

                is_valid, reason = validate_idea(candidate)
                if not is_valid:
                    log.info(f"  Attempt {attempt}: validation failed ({reason}): {repr(candidate)}")
                    seen_descs.append(candidate)
                    time.sleep(RETRY_DELAY)
                    continue

                is_dup, matched_id, candidate_vec = is_duplicate_idea(
                    candidate, existing_embeddings
                )
                if is_dup:
                    log.info(f"  Attempt {attempt}: duplicate of {matched_id} -- retrying")
                    seen_descs.append(candidate)
                    time.sleep(RETRY_DELAY)
                    continue

                idea_description = candidate
                vec = candidate_vec
                break

            except Exception as e:
                log.warning(f"  Attempt {attempt} error: {e}")
                time.sleep(RETRY_DELAY)

        if idea_description is None:
            # This round failed entirely -- send to the back of the queue.
            # It will be retried after every other pending slot gets a turn.
            log.warning(
                f"  All {MAX_RETRIES} attempts failed -- "
                f"re-queuing after {len(queue)} remaining slot(s)"
            )
            queue.append((job_index, job, rounds))
            time.sleep(RETRY_DELAY)
            continue

        idea_id = f"{cat}_{diff}_{job['job_index_within_cat']:04d}"
        conn.execute(
            "INSERT OR IGNORE INTO ideas (id, category, difficulty, description, job_index) "
            "VALUES (?,?,?,?,?)",
            (idea_id, cat, diff, idea_description, job_index),
        )
        conn.commit()

        store_embedding(idea_id, vec, existing_embeddings, econn)
        total_stored += 1

        log.info(f"  [+] {idea_id}")
        log.info(f"      {idea_description}")
        log.info(f"      total: {total_stored}/{total_target}  |  queue: {len(queue)} remaining")

        time.sleep(REQUEST_DELAY)

    log.info(f"Phase 2 complete. Unique ideas stored: {total_stored}")

    print("\n-- Ideas by category --")
    for r in conn.execute(
        "SELECT category, difficulty, COUNT(*) as n FROM ideas "
        "GROUP BY category, difficulty ORDER BY category, difficulty"
    ).fetchall():
        print(f"  {r['category']:<30} {r['difficulty']:<14} {r['n']:>4}")

    conn.close()
    econn.close()


if __name__ == "__main__":
    main()
