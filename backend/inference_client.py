"""HTTP client for the self-hosted model server.

Both vLLM and llama.cpp expose an OpenAI-compatible REST API at
``/v1/chat/completions``.  This module wraps that endpoint so the rest of
the application has no dependency on the OpenAI Python SDK for the
generation step.
"""
import httpx

from backend.config import Config
from backend.logger import app_logger


class InferenceError(Exception):
    """Raised when the model server returns a non-2xx response or is
    unreachable."""


def generate(prompt: str) -> str:
    """Send *prompt* to the self-hosted model server and return the
    generated text.

    Parameters
    ----------
    prompt:
        The full prompt string (system + user content merged into a single
        user message).

    Returns
    -------
    str
        The assistant reply from the model server.

    Raises
    ------
    InferenceError
        If the server is unreachable or returns an error response.
    """
    url = f"{Config.LLM_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": Config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": Config.LLM_TEMPERATURE,
    }

    try:
        response = httpx.post(url, json=payload, timeout=Config.LLM_TIMEOUT_SECS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        app_logger.error(
            "Model server returned %s for %s: %s",
            exc.response.status_code, url, exc.response.text,
        )
        raise InferenceError(
            f"Model server error {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        app_logger.error("Could not reach model server at %s: %s", url, exc)
        raise InferenceError(f"Model server unreachable: {exc}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        app_logger.error("Unexpected model server response shape: %s", data)
        raise InferenceError("Unexpected response shape from model server") from exc
