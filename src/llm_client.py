import os
from typing import Any, Optional

try:
    from llama_index.core.llms import LLM  # type: ignore[import]
    from llama_index.llms.gemini import Gemini  # type: ignore[import]
    from llama_index.llms.ollama import Ollama  # type: ignore[import]
    from llama_index.llms.openai import OpenAI  # type: ignore[import]
    from llama_index.llms.openrouter import OpenRouter  # type: ignore[import]

    LI_AVAILABLE = True
except ImportError:
    LI_AVAILABLE = False
    LLM = Any  # Dummy type for annotation
    OpenAI = Gemini = OpenRouter = Ollama = None

try:
    from llama_index.llms.openai_like import OpenAILike  # type: ignore[import]
except ImportError:
    OpenAILike = None


class LLMClient:
    """
    Wrapper for various LLM providers including local and cloud options.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        if not LI_AVAILABLE:
            raise ImportError(
                "llama_index is not installed. LLM features are unavailable."
            )

        self.llm = self._create_llm()

    def _get_api_key(self, provider: str) -> str:
        # Local providers usually don't need a key, or allow any string
        if provider in ["ollama", "local", "lmstudio", "mlx"]:
            return "lm-studio"  # Dummy key for local

        key_name = f"{provider.upper()}_API_KEY"
        key = self.api_key or os.getenv(key_name)
        if not key:
            raise ValueError(
                f"API key for {provider} not found. Please set {key_name}."
            )
        return key

    def _ensure_v1_suffix(self, url: str) -> str:
        if not url.endswith("/v1"):
            return f"{url.rstrip('/')}/v1"
        return url

    def _create_llm(self) -> LLM:
        if self.provider == "openai":
            return OpenAI(
                model=self.model or "gpt-4o", api_key=self._get_api_key("openai")
            )

        elif self.provider == "gemini":
            return Gemini(
                model=self.model or "models/gemini-1.5-pro",
                api_key=self._get_api_key("gemini"),
            )

        elif self.provider == "openrouter":
            return OpenRouter(
                model=self.model or "google/gemini-pro-1.5",
                api_key=self._get_api_key("openrouter"),
            )

        elif self.provider == "ollama":
            # Default to a model that's commonly available
            # Try llama3.2 first (most commonly available), then gemma2
            model_name = self.model or "llama3.2:latest"
            return Ollama(
                model=model_name,
                base_url=self.base_url or "http://localhost:11434",
                request_timeout=600.0,  # Increased from 300.0 for slower models
                timeout=600.0,  # Additional timeout for some llama-index versions
            )

        elif self.provider in ["local", "lmstudio", "mlx"]:
            # Use OpenAILike if available to bypass strict model validation
            default_base_url = (
                "http://localhost:8080"
                if self.provider == "mlx"
                else "http://localhost:1234"
            )
            if OpenAILike:
                return OpenAILike(
                    model=self.model or "local-model",
                    api_key=self._get_api_key("local"),
                    api_base=self._ensure_v1_suffix(self.base_url or default_base_url),
                    is_chat_model=True,
                )
            else:
                # Fallback: Use a "valid" OpenAI model name to bypass validation if OpenAILike is missing
                print(
                    "Warning: llama-index-llms-openai-like not found. Fallback to OpenAI class with 'gpt-3.5-turbo'."
                )
                return OpenAI(
                    model="gpt-3.5-turbo",  # Hack to pass validation
                    api_key=self._get_api_key("local"),
                    api_base=self._ensure_v1_suffix(self.base_url or default_base_url),
                )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def complete(self, prompt: str) -> str:
        try:
            response = self.llm.complete(prompt)
            return response.text
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"LLM complete failed: {type(e).__name__}: {e}")
            logger.error(
                f"Provider: {self.provider}, Model: {self.model}, Base URL: {self.base_url}"
            )
            # Re-raise to allow fallback
            raise

    def stream_complete(self, prompt: str):
        """
        Stream completion tokens from LLM.

        Args:
            prompt: Input prompt for completion.

        Yields:
            str: Each token/chunk as it becomes available.
        """
        response_gen = self.llm.stream_complete(prompt)
        for response in response_gen:
            yield response.delta

    def cache_namespace(self) -> str:
        parts = [self.provider or "", self.model or "", self.base_url or ""]
        return "|".join(parts)
