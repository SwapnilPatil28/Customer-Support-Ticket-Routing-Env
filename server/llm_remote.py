"""Thin client for calling a remote LLM from the FastAPI server.

Used by the dashboard's "live inference" panel so a Hugging Face Space can
delegate the expensive forward pass to a dedicated HF Inference Endpoint
(GPU-backed) without loading the model inside the Space container.

Two backends are supported:

- ``chat`` (default) — OpenAI-compatible ``/v1/chat/completions`` endpoint.
  Hugging Face TGI-based Inference Endpoints expose this path, as do most
  vLLM deployments. This is the recommended setup.
- ``generate`` — Raw TGI ``/generate`` endpoint. Useful when chat templating
  is already baked into the prompt and you just want raw text completion.

Configuration via environment variables (set them as HF Space secrets):

- ``LLM_ENDPOINT_URL``  — **required** to enable the panel. E.g.
  ``https://abc.us-east-1.aws.endpoints.huggingface.cloud``. Without this
  env var, ``is_configured()`` returns ``False`` and the dashboard shows a
  setup hint instead of the demo.
- ``HF_TOKEN``          — **required**. A Hugging Face token with ``read``
  scope over the model repo powering the endpoint.
- ``LLM_ENDPOINT_MODE`` — optional, one of ``chat`` / ``generate``
  (default: ``chat``).
- ``LLM_MODEL_ID``      — optional display / routing hint the endpoint
  sometimes cares about (default: ``"tgi"``).
- ``LLM_MAX_NEW_TOKENS``— optional integer (default: ``160``).
- ``LLM_TIMEOUT_S``     — optional integer (default: ``25``).

The module uses only the Python stdlib (``urllib.request``) so it adds
zero extra dependencies to the HF Space Docker image.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

_LOG = logging.getLogger("icc.llm_remote")


@dataclass(frozen=True)
class RemoteLLMConfig:
    endpoint_url: str
    token: str
    mode: str = "chat"          # "chat" | "generate"
    model_id: str = "tgi"
    max_new_tokens: int = 160
    timeout_s: int = 25

    @classmethod
    def from_env(cls) -> Optional["RemoteLLMConfig"]:
        url = os.environ.get("LLM_ENDPOINT_URL", "").strip()
        token = os.environ.get("HF_TOKEN", "").strip()
        if not url or not token:
            return None
        return cls(
            endpoint_url=url.rstrip("/"),
            token=token,
            mode=os.environ.get("LLM_ENDPOINT_MODE", "chat").strip().lower() or "chat",
            model_id=os.environ.get("LLM_MODEL_ID", "tgi").strip() or "tgi",
            max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "160")),
            timeout_s=int(os.environ.get("LLM_TIMEOUT_S", "25")),
        )


def is_configured() -> bool:
    """Return True iff env vars required for remote inference are set."""
    return RemoteLLMConfig.from_env() is not None


def status_summary() -> Dict[str, Any]:
    """Lightweight status object for the dashboard to surface."""
    cfg = RemoteLLMConfig.from_env()
    if cfg is None:
        return {
            "configured": False,
            "reason": (
                "Set LLM_ENDPOINT_URL and HF_TOKEN as Space secrets to enable "
                "the live inference panel."
            ),
        }
    return {
        "configured": True,
        "mode": cfg.mode,
        "model_id": cfg.model_id,
        "max_new_tokens": cfg.max_new_tokens,
        # Never surface the token; just confirm it is present.
        "token_present": bool(cfg.token),
        # Only expose the host (not the full URL, in case a query-string key
        # ever leaks into env by accident).
        "host": _safe_host(cfg.endpoint_url),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _safe_host(url: str) -> str:
    try:
        return url.split("://", 1)[-1].split("/", 1)[0]
    except Exception:
        return "(unknown)"


def _http_post(url: str, headers: Dict[str, str], body: bytes, timeout_s: int) -> str:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"LLM endpoint returned HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}"
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"LLM endpoint unreachable: {exc}") from exc


def _call_chat(cfg: RemoteLLMConfig, prompt: str) -> str:
    url = f"{cfg.endpoint_url}/v1/chat/completions"
    payload = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": cfg.max_new_tokens,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.token}",
    }
    raw = _http_post(url, headers, json.dumps(payload).encode("utf-8"), cfg.timeout_s)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM endpoint returned non-JSON: {raw[:400]}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected chat response shape: {raw[:400]}") from exc


def _call_generate(cfg: RemoteLLMConfig, prompt: str) -> str:
    url = f"{cfg.endpoint_url}/generate"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
            "return_full_text": False,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.token}",
    }
    raw = _http_post(url, headers, json.dumps(payload).encode("utf-8"), cfg.timeout_s)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM endpoint returned non-JSON: {raw[:400]}") from exc
    # TGI returns either {"generated_text": "..."} or a list of such objects.
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"])
    raise RuntimeError(f"Unexpected /generate response shape: {raw[:400]}")


def generate(prompt: str) -> str:
    """Send ``prompt`` to the configured remote endpoint and return raw text.

    Raises RuntimeError with a human-readable message on any failure so the
    caller (the FastAPI demo endpoint) can surface it in the dashboard.
    """
    cfg = RemoteLLMConfig.from_env()
    if cfg is None:
        raise RuntimeError(
            "Remote LLM not configured. Set LLM_ENDPOINT_URL and HF_TOKEN."
        )
    _LOG.info("Calling remote LLM %s mode=%s", _safe_host(cfg.endpoint_url), cfg.mode)
    if cfg.mode == "generate":
        return _call_generate(cfg, prompt)
    return _call_chat(cfg, prompt)
