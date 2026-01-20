"""HTTP client for the self-hosted model server.

Both vLLM and llama.cpp expose an OpenAI-compatible REST API at
``/v1/chat/completions``.  This module wraps that endpoint so the rest of
the application has no dependency on the OpenAI Python SDK for the
generation step.

The active backend is selected by the ``LLM_BACKEND`` environment variable
(``vllm`` or ``llamacpp``).  The base URL is resolved from ``LLM_BASE_URL``
when set, otherwise from the per-backend default in ``Config.get_llm_base_url``.
"""
import httpx

from backend.config import Config
from backend.logger import app_logger


class InferenceError(Exception):
    """Raised when the model server returns a non-2xx response or is
    unreachable."""


def generate(prompt: str) -> str:
    """Send *prompt* to the configured self-hosted model server and return
    the generated text.

    Parameters
    ----------
    prompt:
        The full prompt string sent as a single user message.

    Returns
    -------
    str
        The assistant reply from the model server.

    Raises
    ------
    InferenceError
        If the server is unreachable or returns an error response.
    """
    base_url = Config.get_llm_base_url()
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": Config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": Config.LLM_TEMPERATURE,
    }

    app_logger.debug(
        "Calling %s backend at %s (model=%s)",
        Config.LLM_BACKEND, url, Config.LLM_MODEL,
    )

    try:
        response = httpx.post(url, json=payload, timeout=Config.LLM_TIMEOUT_SECS)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        app_logger.error(
            "Model server (%s) returned %s: %s",
            Config.LLM_BACKEND, exc.response.status_code, exc.response.text,
        )
        raise InferenceError(
            f"Model server error {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        app_logger.error(
            "Could not reach %s server at %s: %s",
            Config.LLM_BACKEND, url, exc,
        )
        raise InferenceError(f"Model server unreachable: {exc}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        app_logger.error(
            "Unexpected response shape from %s server: %s",
            Config.LLM_BACKEND, data,
        )
        raise InferenceError("Unexpected response shape from model server") from exc
