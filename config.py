"""Provider settings: gemini (default) | openai | ollama.

Campus guidance allows alternatives when OpenAI billing/keys are unavailable.
"""

from __future__ import annotations

import os
import ssl
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _configure_ssl() -> None:
    """Windows networks that intercept HTTPS often break cert verification."""
    if (os.getenv("ALLOW_INSECURE_SSL") or "0").strip() != "1":
        return
    try:
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
    except Exception:
        pass
    try:
        import httpx

        _client_init = httpx.Client.__init__
        _async_init = httpx.AsyncClient.__init__

        def _patched_client_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("verify", False)
            return _client_init(self, *args, **kwargs)

        def _patched_async_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("verify", False)
            return _async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client_init  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore[method-assign]
    except Exception:
        pass


_configure_ssl()


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
        "GEMINI_API_KEY is missing. Get one at https://aistudio.google.com/apikey "
        "and put it in .env"
    )


def get_openai_api_key() -> str:
    key = _secret("OPENAI_API_KEY")
    if key and key != "your_key_here":
        return key
    raise RuntimeError("OPENAI_API_KEY is missing.")


def openai_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": get_openai_api_key()}
    base_url = _secret("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def embedding_model() -> str:
    p = provider()
    if p == "gemini":
        return _secret("EMBEDDING_MODEL") or "models/gemini-embedding-001"
    if p == "ollama":
        return _secret("EMBEDDING_MODEL") or "nomic-embed-text"
    return _secret("EMBEDDING_MODEL") or "text-embedding-3-small"


def chat_model() -> str:
    p = provider()
    if p == "gemini":
        return _secret("CHAT_MODEL") or "gemini-flash-latest"
    if p == "ollama":
        return _secret("CHAT_MODEL") or "llama3.2"
    return _secret("CHAT_MODEL") or "gpt-4o"


def ensure_provider_ready() -> str:
    p = provider()
    if p == "gemini":
        get_gemini_api_key()
    elif p == "openai":
        get_openai_api_key()
    elif p == "ollama":
        pass
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER={p}")
    return p
