"""
chat_logger.py -- LLM chat history logger
Writes a human-readable plain-text log of every prompt and response.
Works with both Ollama and llama-cpp-python backends.

Output: output/chat_history.log
"""

import time
from datetime import datetime, timezone
from pathlib import Path

CHAT_LOG_PATH = Path("output/chat_history.log")

_WIDE = "=" * 64
_THIN = "-" * 64


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _clean(text: str) -> str:
    return text.strip()


def log_exchange(
    phase: str,
    context_id: str,
    prompt: str,
    response: str,
    duration_ms: float,
    success: bool,
    note: str = "",
):
    CHAT_LOG_PATH.parent.mkdir(exist_ok=True)

    status   = "OK" if success else "FAILED"
    duration = f"{duration_ms / 1000:.1f}s"
    note_str = f"Note: {note:<35}  |  " if note else ""

    entry = (
        f"\n{_WIDE}\n"
        f"[{_now()}]  Phase: {phase:<18}  |  ID: {context_id}\n"
        f"{note_str}Duration: {duration:<8}  |  Status: {status}\n"
        f"{_THIN}\n"
        f"PROMPT\n"
        f"{_THIN}\n"
        f"{_clean(prompt)}\n"
        f"\n{_THIN}\n"
        f"RESPONSE\n"
        f"{_THIN}\n"
        f"{_clean(response)}\n"
        f"{_WIDE}\n"
    )

    with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


def llm_chat_logged(
    phase: str,
    context_id: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    note: str = "",
) -> str:
    """
    Call the LLM via llm_backend.chat() and log the exchange.
    Returns the response text.
    Raises the original exception on failure (also logged).
    """
    from llm_backend import chat

    t0 = time.perf_counter()
    try:
        response    = chat(prompt, temperature=temperature, max_tokens=max_tokens)
        duration_ms = (time.perf_counter() - t0) * 1000
        log_exchange(phase, context_id, prompt, response, duration_ms, True, note)
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - t0) * 1000
        log_exchange(phase, context_id, prompt, f"[ERROR] {e}", duration_ms, False, note)
        raise


def embed_logged(text: str, context_id: str = "") -> list[float]:
    """NLP embedding (nomic-embed-text) -- used for idea deduplication in phase 2."""
    from llm_backend import embed
    return embed(text)


def code_embed_logged(text: str, context_id: str = "") -> list[float]:
    """Code embedding (Qwen3-Embedding-8B) -- used for snippet deduplication in phase 4."""
    from llm_backend import code_embed
    return code_embed(text)
