"""Tests for LLM backend config resolution and inference client error handling.

These tests never start a real model server.  inference_client calls are
intercepted with unittest.mock so the module can be tested in isolation.

Run with:
    python -m pytest tests/test_config_and_inference.py -v
"""
import importlib
import os
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Config.get_llm_base_url() — backend selection and explicit override
# ---------------------------------------------------------------------------

class TestGetLlmBaseUrl:
    def _reload_config(self, env: dict):
        """Reload backend.config with a patched environment so class-level
        attributes (LLM_BACKEND etc.) are re-evaluated."""
        with mock.patch.dict(os.environ, env, clear=False):
            import backend.config as cfg_mod
            importlib.reload(cfg_mod)
            return cfg_mod.Config

    def test_vllm_default(self):
        Config = self._reload_config({"LLM_BACKEND": "vllm", "LLM_BASE_URL": ""})
        url = Config.get_llm_base_url()
        assert "8000" in url, f"vLLM default should use port 8000, got {url}"

    def test_llamacpp_default(self):
        Config = self._reload_config({"LLM_BACKEND": "llamacpp", "LLM_BASE_URL": ""})
        url = Config.get_llm_base_url()
        assert "8080" in url, f"llama.cpp default should use port 8080, got {url}"

    def test_explicit_base_url_overrides_backend(self):
        Config = self._reload_config({
            "LLM_BACKEND": "vllm",
            "LLM_BASE_URL": "http://custom-host:9999",
        })
        url = Config.get_llm_base_url()
        assert url == "http://custom-host:9999", (
            f"Explicit LLM_BASE_URL should override backend default, got {url}"
        )

    def test_trailing_slash_stripped(self):
        Config = self._reload_config({
            "LLM_BACKEND": "vllm",
            "LLM_BASE_URL": "http://host:8000/",
        })
        url = Config.get_llm_base_url()
        assert not url.endswith("/"), "get_llm_base_url must strip trailing slash"

    def test_unknown_backend_falls_back_to_safe_default(self):
        Config = self._reload_config({"LLM_BACKEND": "unknown_backend", "LLM_BASE_URL": ""})
        url = Config.get_llm_base_url()
        assert url.startswith("http://"), "Unknown backend should still return a valid URL"


# ---------------------------------------------------------------------------
# inference_client.generate() — success and failure paths
# ---------------------------------------------------------------------------

class TestInferenceClientGenerate:
    def _make_response(self, content: str, status_code: int = 200):
        resp = mock.MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        resp.raise_for_status = mock.MagicMock()
        if status_code >= 400:
            import httpx
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=mock.MagicMock(), response=resp
            )
        return resp

    def test_successful_generation(self):
        from backend import inference_client
        fake_resp = self._make_response("  The answer is 42.  ")
        with mock.patch("httpx.post", return_value=fake_resp):
            result = inference_client.generate("What is the answer?")
        assert result == "The answer is 42.", f"Expected stripped content, got {result!r}"

    def test_http_error_raises_inference_error(self):
        import httpx
        from backend import inference_client
        from backend.inference_client import InferenceError
        fake_resp = self._make_response("Server error", status_code=500)
        with mock.patch("httpx.post", return_value=fake_resp):
            with pytest.raises(InferenceError):
                inference_client.generate("anything")

    def test_connection_error_raises_inference_error(self):
        import httpx
        from backend import inference_client
        from backend.inference_client import InferenceError
        with mock.patch(
            "httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with pytest.raises(InferenceError):
                inference_client.generate("anything")

    def test_malformed_response_raises_inference_error(self):
        from backend import inference_client
        from backend.inference_client import InferenceError
        bad_resp = mock.MagicMock()
        bad_resp.raise_for_status = mock.MagicMock()
        bad_resp.json.return_value = {"unexpected": "shape"}
        with mock.patch("httpx.post", return_value=bad_resp):
            with pytest.raises(InferenceError):
                inference_client.generate("anything")
