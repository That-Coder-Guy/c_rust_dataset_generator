"""
llm_backend.py -- llama-cpp-python inference backend
Provides three model singletons loaded on first use:
    get_llm()        -- chat/generation LLM
    get_embed()      -- NLP embedding model (for idea deduplication in phase 2)
    get_code_embed() -- code embedding model (for snippet deduplication in phase 4)

Model paths are configured via environment variables or the constants below.

Environment variables (all optional, fall back to constants):
    LLM_MODEL_PATH        -- path to the LLM GGUF file
    EMBED_MODEL_PATH      -- path to the NLP embedding GGUF file
    CODE_EMBED_MODEL_PATH -- path to the code embedding GGUF file
    N_GPU_LAYERS          -- layers to offload to GPU (-1 = all)
    N_CTX                 -- context window size for the LLM
"""

import os
from pathlib import Path
from llama_cpp import Llama

# ── Default paths -- override with environment variables ──────────────────────

_MODELS = Path.home() / "c_rust_dataset" / "models"

LLM_MODEL_PATH        = os.environ.get("LLM_MODEL_PATH",
                        str(_MODELS / "qwen3-coder-30b.gguf"))

EMBED_MODEL_PATH      = os.environ.get("EMBED_MODEL_PATH",
                        str(_MODELS / "nomic-embed-text.gguf"))

CODE_EMBED_MODEL_PATH = os.environ.get("CODE_EMBED_MODEL_PATH",
                        str(_MODELS / "Qwen3-Embedding-8B-Q4_K_M.gguf"))

N_GPU_LAYERS          = int(os.environ.get("N_GPU_LAYERS", "-1"))
N_CTX                 = int(os.environ.get("N_CTX", "8192"))

# ── Singletons ────────────────────────────────────────────────────────────────

_llm        = None
_embed      = None
_code_embed = None


def get_llm() -> Llama:
    """Return the chat LLM, loading it on first call."""
    global _llm
    if _llm is None:
        print(f"Loading LLM: {LLM_MODEL_PATH}")
        _llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            n_batch=512,
            verbose=False,
        )
        print("LLM loaded OK")
    return _llm


def get_embed() -> Llama:
    """
    Return the NLP embedding model (nomic-embed-text).
    Used by phase 2 for idea description deduplication.
    """
    global _embed
    if _embed is None:
        print(f"Loading NLP embed model: {EMBED_MODEL_PATH}")
        _embed = Llama(
            model_path=EMBED_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=512,
            embedding=True,
            verbose=False,
        )
        print("NLP embed model loaded OK")
    return _embed


def get_code_embed() -> Llama:
    """
    Return the code embedding model (Qwen3-Embedding-8B).
    Used by phase 4 for code snippet deduplication.
    Understands code structure better than a general NLP model.
    """
    global _code_embed
    if _code_embed is None:
        print(f"Loading code embed model: {CODE_EMBED_MODEL_PATH}")
        _code_embed = Llama(
            model_path=CODE_EMBED_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=2048,   # code snippets can be longer than idea descriptions
            embedding=True,
            verbose=False,
        )
        print("Code embed model loaded OK")
    return _code_embed


# ── Inference functions ───────────────────────────────────────────────────────

def chat(prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """Send a prompt to the LLM and return the response text."""
    llm = get_llm()
    result = llm(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )
    return result["choices"][0]["text"].strip()


def _normalise(vec) -> list[float]:
    """Unit-normalise a vector for cosine similarity via dot product."""
    import numpy as np
    arr  = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).tolist()


def embed(text: str) -> list[float]:
    """
    NLP embedding -- use for natural language text (idea descriptions).
    Returns a unit-normalised vector.
    """
    return _normalise(get_embed().embed(text[:2048]))


def code_embed(text: str) -> list[float]:
    """
    Code embedding -- use for source code similarity (snippet deduplication).
    Returns a unit-normalised vector from Qwen3-Embedding-8B.
    """
    return _normalise(get_code_embed().embed(text[:4096]))
