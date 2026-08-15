"""Provider settings: gemini (default) | openai | ollama.

Campus guidance allows alternatives when OpenAI billing/keys are unavailable.
Gemini works locally and on Streamlit Cloud. Ollama is local-only.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    val = (os.getenv(name) or "").strip()
    if val:
        return val
    try:
        import streamlit as st

        return str(st.secrets.get(name, default) or "").strip()
    except Exception:
        return default


def provider() -> str:
    return (_secret("LLM_PROVIDER") or "gemini").lower()


def get_gemini_api_key() -> str:
    key = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")
    if key and key not in {"your_key_here", "your_gemini_key_here"}:
        return key
    raise RuntimeError(
        "GEMINI_API_KEY is missing.\n"
        "1) Open https://aistudio.google.com/apikey\n"
        "2) Create an API key (free)\n"
        "3) Put it in .env as GEMINI_API_KEY=...\n"
        "   or in Streamlit Cloud Secrets."
    )


def get_openai_api_key() -> str:
    key = _secret("OPENAI_API_KEY")
    if key and key != "your_key_here":
        return key
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to .env or Streamlit Cloud Secrets."
    )


def openai_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": get_openai_api_key()}
    base_url = _secret("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def embedding_model() -> str:
    p = provider()
    if p == "gemini":
        return _secret("EMBEDDING_MODEL") or "models/text-embedding-004"
    if p == "ollama":
        return _secret("EMBEDDING_MODEL") or "nomic-embed-text"
    return _secret("EMBEDDING_MODEL") or "text-embedding-3-small"


def chat_model() -> str:
    p = provider()
    if p == "gemini":
        # Accurate default for assignment; override with CHAT_MODEL if needed
        return _secret("CHAT_MODEL") or "gemini-2.0-flash"
    if p == "ollama":
        return _secret("CHAT_MODEL") or "llama3.2"
    return _secret("CHAT_MODEL") or "gpt-4o"


def ensure_provider_ready() -> str:
    """Validate the active provider can authenticate. Returns provider name."""
    p = provider()
    if p == "gemini":
        get_gemini_api_key()
    elif p == "openai":
        get_openai_api_key()
    elif p == "ollama":
        # Local daemon — validated when first call is made
        pass
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER={p}. Use gemini, openai, or ollama.")
    return p
