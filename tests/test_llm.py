"""Tests for LLM client with proxy support."""
import os
import pytest

from arsi.foundation.llm import LLMClient, LLMConfig, LLMResponse, create_llm_client


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.use_proxy is False
        assert config.proxy_url == "http://127.0.0.1:7890"
        assert config.fallback_to_heuristic is True

    def test_proxy_config(self):
        config = LLMConfig(use_proxy=True, proxy_url="http://127.0.0.1:7890")
        assert config.use_proxy is True
        assert "7890" in config.proxy_url


class TestLLMClient:
    def test_init_without_api_key(self):
        """Client should initialize even without API key (placeholder)."""
        config = LLMConfig(api_key="", fallback_to_heuristic=True)
        client = LLMClient(config)
        # May or may not be available depending on openai package
        assert client is not None

    def test_proxy_env_setup(self):
        """Proxy environment variables should be set when enabled."""
        config = LLMConfig(use_proxy=True, proxy_url="http://127.0.0.1:7890")
        client = LLMClient(config)
        assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"
        assert os.environ.get("HTTPS_PROXY") == "http://127.0.0.1:7890"

    def test_proxy_env_cleanup(self):
        """Proxy env vars should be cleared when disabled."""
        # First enable
        os.environ["HTTP_PROXY"] = "http://test:1234"
        config = LLMConfig(use_proxy=False)
        LLMClient(config)
        assert "HTTP_PROXY" not in os.environ

    def test_heuristic_fallback(self):
        """Should fallback to heuristic when LLM unavailable."""
        config = LLMConfig(api_key="", fallback_to_heuristic=True)
        client = LLMClient(config)
        # Force unavailable
        client._client = None
        response = client.chat("test prompt")
        assert response.success is True
        assert "[heuristic]" in response.content

    def test_no_fallback_returns_error(self):
        """Should return error when LLM unavailable and no fallback."""
        config = LLMConfig(api_key="", fallback_to_heuristic=False)
        client = LLMClient(config)
        client._client = None
        response = client.chat("test")
        assert response.success is False

    def test_chat_json_fallback(self):
        """chat_json should handle heuristic fallback gracefully."""
        config = LLMConfig(api_key="", fallback_to_heuristic=True)
        client = LLMClient(config)
        client._client = None
        result = client.chat_json("test")
        # Heuristic returns non-JSON, should get error dict
        assert "error" in result or "raw" in result

    def test_create_llm_client_factory(self):
        """Factory function should create a client."""
        client = create_llm_client(
            provider="openai",
            model="gpt-4o-mini",
            api_key="sk-test",
            use_proxy=True,
            proxy_url="http://127.0.0.1:7890",
        )
        assert client is not None
        assert client.config.use_proxy is True
        assert client.config.proxy_url == "http://127.0.0.1:7890"

    def test_available_property(self):
        """available should reflect client state."""
        config = LLMConfig(api_key="sk-test", fallback_to_heuristic=True)
        client = LLMClient(config)
        # If openai package is installed, client may be available
        # If not, should be None
        assert isinstance(client.available, bool)


class TestLLMResponse:
    def test_default_response(self):
        r = LLMResponse()
        assert r.success is False
        assert r.content == ""

    def test_success_response(self):
        r = LLMResponse(content="hello", success=True)
        assert r.success is True
        assert r.content == "hello"
