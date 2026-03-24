"""
Phase 1 -- Taxonomy Builder
Defines the category matrix, difficulty distribution, and prompt seeds.
Each job produces exactly ONE pair -- no batching.
Outputs: output/taxonomy.json
"""

import json
import random
from pathlib import Path

# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = {
    "memory_management": {
        "target": 750,
        "description": "malloc/free, heap allocation, pointer arithmetic, buffer management",
        "c_hints": ["malloc", "calloc", "realloc", "free", "pointer arithmetic", "buffer overflow prevention"],
        "rust_hints": ["Box", "Vec", "heap allocation", "ownership", "drop", "raw pointers with unsafe"],
        "difficulties": {"beginner": 0.3, "intermediate": 0.4, "advanced": 0.3},
    },
    "strings_and_buffers": {
        "target": 600,
        "description": "String manipulation, C-strings, byte buffers, encoding",
        "c_hints": ["strcpy", "strcat", "snprintf", "strtok", "memcpy", "null terminator"],
        "rust_hints": ["String", "&str", "format!", "split", "bytes()", "from_utf8"],
        "difficulties": {"beginner": 0.35, "intermediate": 0.4, "advanced": 0.25},
    },
    "io_and_files": {
        "target": 600,
        "description": "File I/O, stdio, reading/writing, error handling on I/O",
        "c_hints": ["fopen", "fclose", "fread", "fwrite", "fprintf", "fgets", "errno"],
        "rust_hints": ["File", "BufReader", "BufWriter", "read_to_string", "write_all", "std::io::Error"],
        "difficulties": {"beginner": 0.3, "intermediate": 0.45, "advanced": 0.25},
    },
    "data_structures": {
        "target": 750,
        "description": "Linked lists, trees, hash maps, stacks, queues",
        "c_hints": ["struct", "typedef", "linked list", "binary tree", "hash table", "void*"],
        "rust_hints": ["struct", "impl", "enum", "HashMap", "BTreeMap", "VecDeque", "Box<T>"],
        "difficulties": {"beginner": 0.2, "intermediate": 0.45, "advanced": 0.35},
    },
    "algorithms": {
        "target": 600,
        "description": "Sorting, searching, recursion, dynamic programming",
        "c_hints": ["qsort", "bsearch", "recursion", "iterative", "comparison function"],
        "rust_hints": ["sort_by", "binary_search", "iter()", "map()", "fold()", "collect()"],
        "difficulties": {"beginner": 0.25, "intermediate": 0.4, "advanced": 0.35},
    },
    "concurrency": {
        "target": 500,
        "description": "Threads, mutexes, condition variables, atomics",
        "c_hints": ["pthread_create", "pthread_mutex_lock", "pthread_cond_wait", "atomic"],
        "rust_hints": ["thread::spawn", "Mutex", "Arc", "RwLock", "channel", "atomic"],
        "difficulties": {"beginner": 0.15, "intermediate": 0.4, "advanced": 0.45},
    },
    "error_handling": {
        "target": 500,
        "description": "Error codes, errno, propagation, custom error types",
        "c_hints": ["errno", "perror", "return -1", "NULL check", "assert", "setjmp/longjmp"],
        "rust_hints": ["Result", "Option", "?", "unwrap_or_else", "thiserror", "anyhow", "map_err"],
        "difficulties": {"beginner": 0.3, "intermediate": 0.45, "advanced": 0.25},
    },
    "ffi_and_interop": {
        "target": 700,
        "description": "Calling C from Rust, extern declarations, unsafe interop",
        "c_hints": ["extern", "shared library", "function pointer", "struct layout", "ABI"],
        "rust_hints": ["extern \"C\"", "unsafe", "#[no_mangle]", "libc", "bindgen", "FFI types"],
        "difficulties": {"beginner": 0.1, "intermediate": 0.35, "advanced": 0.55},
    },
}

TOTAL_TARGET = sum(c["target"] for c in CATEGORIES.values())

# ── Difficulty instructions ───────────────────────────────────────────────────

DIFFICULTY_INSTRUCTIONS = {
    "beginner": (
        "The snippet should be short (10-25 lines), self-contained, and demonstrate "
        "a single concept clearly. Suitable for someone new to the language."
    ),
    "intermediate": (
        "The snippet should be 25-60 lines. It may use multiple related concepts, "
        "error checking, and realistic patterns used in production code."
    ),
    "advanced": (
        "The snippet should be 40-100 lines. It may involve unsafe code, non-trivial "
        "algorithms, or idiomatic patterns that require deeper language knowledge."
    ),
}

# ── Job plan generator ────────────────────────────────────────────────────────

def build_job_plan(categories: dict) -> list[dict]:
    """
    Expand category targets into a flat list of single-pair generation jobs.
    Each job = one API call = one C/Rust pair. No batching.

    Rounding is corrected per-category: the last difficulty bucket absorbs
    any remainder so each category hits its target exactly.
    """
    jobs = []
    for cat_slug, cat_info in categories.items():
        target     = cat_info["target"]
        dist       = cat_info["difficulties"]
        diff_items = list(dist.items())
        allocated  = 0

        for idx, (difficulty, fraction) in enumerate(diff_items):
            # Last bucket gets whatever is left to hit the target exactly
            if idx == len(diff_items) - 1:
                count = target - allocated
            else:
                count = round(target * fraction)
            allocated += count

            for i in range(count):
                jobs.append({
                    "category_slug":          cat_slug,
                    "category_name":          cat_slug.replace("_", " ").title(),
                    "category_description":   cat_info["description"],
                    "difficulty":             difficulty,
                    "difficulty_instruction": DIFFICULTY_INSTRUCTIONS[difficulty],
                    "job_index_within_cat":   i,
                    "c_hints":                cat_info["c_hints"],
                    "rust_hints":             cat_info["rust_hints"],
                })

    # Shuffle so categories interleave -- better variety and dedup coverage
    random.seed(42)
    random.shuffle(jobs)
    return jobs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    jobs = build_job_plan(CATEGORIES)

    taxonomy = {
        "categories":   CATEGORIES,
        "total_target": TOTAL_TARGET,
        "total_jobs":   len(jobs),   # now equals total_target (1 job = 1 pair)
        "jobs":         jobs,
    }

    out_path = output_dir / "taxonomy.json"
    out_path.write_text(json.dumps(taxonomy, indent=2))

    print(f"Taxonomy written to {out_path}")
    print(f"  Categories        : {len(CATEGORIES)}")
    print(f"  Total target pairs: {TOTAL_TARGET}")
    print(f"  Total jobs        : {len(jobs)}  (1 job = 1 pair = 1 API call)")
    for cat, info in CATEGORIES.items():
        print(f"  {cat:<30} -> {info['target']} pairs")


if __name__ == "__main__":
    main()
