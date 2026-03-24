"""
llm_backend.py -- llama-cpp-python inference backend
Replaces Ollama for running on HPC clusters without Ollama installed.

Provides two module-level singletons loaded on first use:
    get_llm()        -- returns the chat LLM
    get_embed()      -- returns the NLP embedding model (for ideas, phase 2)
    get_code_embed() -- returns the code embedding model (for snippets, phase 4)

Model paths and GPU settings are configured via environment variables
or the constants below, making it easy to swap models without editing
phase scripts.

Environment variables (all optional, fall back to constants):
    LLM_MODEL_PATH   -- path to the LLM GGUF file
    EMBED_MODEL_PATH      -- path to the NLP embedding GGUF file
    CODE_EMBED_MODEL_PATH -- path to the code embedding GGUF file
    N_GPU_LAYERS     -- number of layers to offload to GPU (-1 = all)
    N_CTX            -- context window size
"""

import os
from pathlib import Path
from llama_cpp import Llama

# ── Default paths -- override with environment variables ──────────────────────

_MODELS = Path(__file__).parent / "models"

LLM_MODEL_PATH        = os.environ.get("LLM_MODEL_PATH",
                        str(_MODELS / "qwen3-coder-30b.gguf"))
EMBED_MODEL_PATH      = os.environ.get("EMBED_MODEL_PATH",
                        str(_MODELS / "nomic-embed-text.gguf"))
CODE_EMBED_MODEL_PATH = os.environ.get("CODE_EMBED_MODEL_PATH",
                        str(_MODELS / "Qwen3-Embedding-8B-Q4_K_M.gguf"))

N_GPU_LAYERS     = int(os.environ.get("N_GPU_LAYERS", "-1"))
N_CTX            = int(os.environ.get("N_CTX", "8192"))

# ── Singletons -- loaded once, reused across all calls ────────────────────────

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
    """Return the NLP embedding model (nomic-embed-text). Used in phase 2."""
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
    """Return the code embedding model (Qwen3-Embedding-8B). Used in phase 4."""
    global _code_embed
    if _code_embed is None:
        print(f"Loading code embed model: {CODE_EMBED_MODEL_PATH}")
        _code_embed = Llama(
            model_path=CODE_EMBED_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=2048,
            embedding=True,
            verbose=False,
        )
        print("Code embed model loaded OK")
    return _code_embed


def chat(prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
    """
    Send a prompt to the LLM and return the response text.
    Uses a simple single-turn format compatible with instruction-tuned models.
    """
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
    import numpy as np
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm if norm > 0 else arr).tolist()


def embed(text: str) -> list[float]:
    """NLP embedding for natural language text (idea descriptions)."""
    return _normalise(get_embed().embed(text[:2048]))


def code_embed(text: str) -> list[float]:
    """Code embedding for source code similarity (Qwen3-Embedding-8B)."""
    return _normalise(get_code_embed().embed(text[:4096]))
