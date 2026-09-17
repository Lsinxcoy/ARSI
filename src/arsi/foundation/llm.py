"""LLM client with proxy support.

Supports:
- OpenAI-compatible APIs (OpenAI, DeepSeek, Moonshot, etc.)
- Anthropic API
- Proxy configuration (HTTP/HTTPS/SOCKS)
- Fallback to local heuristic when LLM unavailable
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # openai | anthropic | custom
    model: str = "gpt-4o-mini"
    api_key: str = ""
    api_base: str = ""
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout: int = 60
    extra_headers: dict = field(default_factory=dict)
    # Proxy settings
    use_proxy: bool = False
    proxy_url: str = "http://127.0.0.1:7890"
    # Fallback
    fallback_to_heuristic: bool = True


@dataclass
class LLMResponse:
    content: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    success: bool = False
    error: str = ""


class LLMClient:
    """Unified LLM client with proxy support.

    Usage:
        client = LLMClient(LLMConfig(use_proxy=True, proxy_url="http://127.0.0.1:7890"))
        response = client.chat("Hello, who are you?")
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = None
        self._setup_proxy()
        self._init_client()

    def _setup_proxy(self) -> None:
        """Configure proxy environment variables if enabled."""
        if not self.config.use_proxy:
            # Clear any existing proxy env vars
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
                os.environ.pop(key, None)
            return

        proxy = self.config.proxy_url
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        logger.info(f"Proxy enabled: {proxy}")

    def _init_client(self) -> None:
        """Initialize the underlying LLM client."""
        try:
            if self.config.provider == "openai" or self.config.provider == "custom":
                self._init_openai()
            elif self.config.provider == "anthropic":
                self._init_anthropic()
            else:
                logger.warning(f"Unknown provider: {self.config.provider}")
        except Exception as e:
            logger.warning(f"LLM client init failed: {e}. Will use heuristic fallback.")
            self._client = None

    def _init_openai(self) -> None:
        """Initialize OpenAI-compatible client."""
        try:
            from openai import OpenAI

            kwargs = {"api_key": self.config.api_key or "sk-placeholder"}
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base
            if self.config.extra_headers:
                kwargs["default_headers"] = self.config.extra_headers

            # Proxy support via httpx
            if self.config.use_proxy:
                import httpx
                kwargs["http_client"] = httpx.Client(
                    proxy=self.config.proxy_url,
                    timeout=self.config.timeout,
                )

            self._client = OpenAI(**kwargs)
            logger.info(f"OpenAI client initialized (model={self.config.model})")
        except ImportError:
            logger.warning("openai package not installed. Run: pip install openai")

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            kwargs = {"api_key": self.config.api_key or "sk-placeholder"}
            if self.config.use_proxy:
                import httpx
                kwargs["http_client"] = httpx.Client(
                    proxy=self.config.proxy_url,
                    timeout=self.config.timeout,
                )

            self._client = anthropic.Anthropic(**kwargs)
            logger.info(f"Anthropic client initialized (model={self.config.model})")
        except ImportError:
            logger.warning("anthropic package not installed. Run: pip install anthropic")

    @property
    def available(self) -> bool:
        """Check if LLM is available."""
        return self._client is not None

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Falls back to heuristic if LLM unavailable and fallback is enabled.
        """
        if not self.available:
            if self.config.fallback_to_heuristic:
                return self._heuristic_fallback(prompt)
            return LLMResponse(success=False, error="LLM not available")

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            if self.config.provider in ("openai", "custom"):
                return self._chat_openai(prompt, system, temp, tokens)
            elif self.config.provider == "anthropic":
                return self._chat_anthropic(prompt, system, temp, tokens)
            else:
                return LLMResponse(success=False, error=f"Unknown provider: {self.config.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            if self.config.fallback_to_heuristic:
                return self._heuristic_fallback(prompt)
            return LLMResponse(success=False, error=str(e))

    def _chat_openai(self, prompt: str, system: str, temp: float, tokens: int) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            success=True,
        )

    def _chat_anthropic(self, prompt: str, system: str, temp: float, tokens: int) -> LLMResponse:
        kwargs = {
            "model": self.config.model,
            "max_tokens": tokens,
            "temperature": temp,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            success=True,
        )

    def _heuristic_fallback(self, prompt: str) -> LLMResponse:
        """Heuristic fallback when LLM is unavailable."""
        return LLMResponse(
            content=f"[heuristic] Based on the prompt pattern: {prompt[:100]}...",
            model="heuristic",
            success=True,
        )

    def chat_json(self, prompt: str, system: str = "") -> dict:
        """Send a request and parse JSON response."""
        response = self.chat(prompt, system=system)
        if not response.success:
            return {"error": response.error}

        try:
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw": response.content}


def create_llm_client(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    api_base: str = "",
    use_proxy: bool = False,
    proxy_url: str = "http://127.0.0.1:7890",
) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: "openai", "anthropic", or "custom"
        model: Model name
        api_key: API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)
        api_base: Custom API base URL (for DeepSeek, Moonshot, etc.)
        use_proxy: Whether to use proxy
        proxy_url: Proxy URL (default: http://127.0.0.1:7890)
    """
    if not api_key:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        use_proxy=use_proxy,
        proxy_url=proxy_url,
    )
    return LLMClient(config)
