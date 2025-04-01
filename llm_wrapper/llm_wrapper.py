import asyncio
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Type

import diskcache as dc
from together import Together
import openai
import requests
import tenacity
import tqdm.asyncio

from .config import MODEL_COSTS, RETRY_CONFIG

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
BYTES_PER_GB = 1024 * 1024 * 1024


def get_cache_root() -> str:
    """Get the cache root directory from a list of candidates."""
    if os.getenv("LLM_WRAPPER_CACHE_ROOT"):
        cache_root = os.getenv("LLM_WRAPPER_CACHE_ROOT")
    else:
        cache_root = os.path.expanduser("~/.cache")
    print(f"Using cache root: {cache_root}")
    return cache_root


@dataclass
class LLMConfig:
    """Configuration for LLM providers and caching"""

    cache_size_gb: int = 2
    cache_policy: str = "least-recently-used"
    max_concurrent_requests: int = 16


class TokenCounter:
    CHARS_PER_TOKEN = 4

    def __init__(self, model: str):
        if model in MODEL_COSTS:
            self.input_cost, self.output_cost = (
                MODEL_COSTS[model]["input"],
                MODEL_COSTS[model]["output"],
            )
        else:
            self.input_cost, self.output_cost = 0.0, 0.0
        self.input_total, self.output_total = 0, 0
        self.input_new, self.output_new = 0, 0

    @staticmethod
    def format_token_count(tokens: float) -> str:
        """Convert token count to human readable string with appropriate unit."""
        if tokens > 1_000_000:
            return f"{tokens/1_000_000:.2f}M"
        elif tokens > 1000:
            return f"{tokens/1000:.1f}k"
        return str(int(tokens))

    def _update(self, prompt, response):
        if "usage" in response.raw_response:
            usage = response.raw_response["usage"]
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
        else:
            input_tokens = len(prompt) / self.CHARS_PER_TOKEN
            output_tokens = len(response.text) / self.CHARS_PER_TOKEN

        self.input_total += input_tokens
        self.output_total += output_tokens
        if not response.cached:
            self.input_new += input_tokens
            self.output_new += output_tokens

    def update(self, prompt, responses):
        for response in responses:
            self._update(prompt, response)

    def print_cost(self):
        cost_input_total = self.input_total * self.input_cost / 1_000_000
        cost_output_total = self.output_total * self.output_cost / 1_000_000
        cost_input_new = self.input_new * self.input_cost / 1_000_000
        cost_output_new = self.output_new * self.output_cost / 1_000_000

        input_total_str = self.format_token_count(self.input_total)
        output_total_str = self.format_token_count(self.output_total)
        input_new_str = self.format_token_count(self.input_new)
        output_new_str = self.format_token_count(self.output_new)

        print(
            f"Total: ${cost_input_total + cost_output_total:.4f} ({input_total_str} input, {output_total_str} output)"
        )
        print(
            f"New: ${cost_input_new + cost_output_new:.4f} ({input_new_str} input, {output_new_str} output)"
        )


@dataclass
class LLMResponse:
    """Standardized response format across different LLM providers"""

    provider: str
    model: str
    raw_response: Dict[str, Any]
    text: str
    cached: bool = False


def create_retry_decorator(
    max_retries: int = RETRY_CONFIG["max_retries"],
    base_delay: float = RETRY_CONFIG["base_delay"],
    max_delay: float = RETRY_CONFIG["max_delay"],
) -> Callable:
    """Create a retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds

    Returns:
        Retry decorator configured with exponential backoff
    """
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(max_retries),
        wait=tenacity.wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
        retry=tenacity.retry_if_exception_type(RETRY_CONFIG["retry_exceptions"]),
        before_sleep=lambda retry_state: logging.warning(
            f"Retrying after {retry_state.next_action.sleep} seconds due to {retry_state.outcome.exception()}"
        ),
        retry_error_callback=lambda retry_state: retry_state.outcome.result(),
    )


class BaseProvider(ABC):
    """A wrapper for interacting with LLM providers. Supports local caching, async, retries, and cost tracking."""

    def __init__(self, model: str, config: LLMConfig = LLMConfig()):
        self.model = model
        self.config = config

        cache_root = get_cache_root()
        cache_dir = os.path.join(cache_root, "llm_wrapper")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache dir: {cache_dir}")

        self.cache = dc.Cache(
            cache_dir,
            size_limit=self.config.cache_size_gb * BYTES_PER_GB,
            eviction_policy=self.config.cache_policy,
        )
        size_bytes = self.cache.volume()
        cache_path = self.cache.directory
        print(f"Cache path: {cache_path}")
        print(f"Current cache size: {size_bytes / BYTES_PER_GB:.2f}/{self.config.cache_size_gb}GB")

    def _get_cache_key(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate cache key for a single message."""
        prompt_hash = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        system_hash = (
            hashlib.sha256(system_prompt.encode()).hexdigest()
            if system_prompt
            else "no_system_prompt"
        )
        kwargs_str = "|".join(f"{k}:{str(v)}" for k, v in sorted(kwargs.items()))
        kwargs_hash = hashlib.sha256(kwargs_str.encode()).hexdigest()
        components = [self.provider, self.model, system_hash, prompt_hash, kwargs_hash]
        final_key = "|".join(components)
        return hashlib.sha256(final_key.encode()).hexdigest()

    async def _check_cache_and_generate_single(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> LLMResponse:
        """Generate a single chat completion with caching."""
        cache_key = self._get_cache_key(prompt, system_prompt, **kwargs)
        cached_response = self.cache.get(cache_key)
        if cached_response is not None:
            choices = cached_response["raw_response"]["choices"]
            return [
                LLMResponse(
                    text=choice["message"]["content"],
                    raw_response=cached_response["raw_response"],
                    provider=self.provider,
                    model=self.model,
                    cached=True,
                )
                for choice in choices
            ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        responses = await self._chat_completions_create(messages, **kwargs)
        self.cache[cache_key] = {"raw_response": responses[0].raw_response}
        # The "raw response" contains all n responses
        return responses

    async def _check_cache_and_generate(
        self, prompts: List[str], system_prompt: str = "", **kwargs
    ) -> List[LLMResponse]:
        semaphore = asyncio.Semaphore(value=self.config.max_concurrent_requests)
        pbar = kwargs.pop("progress_bar", None)  # Get progress bar from kwargs

        async def run_single(prompt: str) -> LLMResponse:
            async with semaphore:
                response = await self._check_cache_and_generate_single(
                    prompt, system_prompt, **kwargs
                )
                if pbar:
                    pbar.update(1)
                return response

        return await asyncio.gather(*[run_single(p) for p in prompts])

    @abstractmethod
    async def _chat_completions_create(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> List[LLMResponse]:
        """Provider-specific implementation of chat completion."""
        pass

    def estimate_tokens(self, prompts, all_responses):
        token_counter = TokenCounter(self.model)
        for prompt, responses in zip(prompts, all_responses):
            token_counter.update(prompt, responses)
        token_counter.print_cost()

    def generate_from_messages(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, **kwargs: Any
    ) -> List[LLMResponse]:
        async def _async_generate_single(messages: List[Dict[str, str]]):
            cache_key = self._get_cache_key(messages, system_prompt, **kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                choices = cached_response["raw_response"]["choices"]
                responses = [
                    LLMResponse(
                        text=choice["message"]["content"],
                        raw_response=cached_response["raw_response"],
                        provider=self.provider,
                        model=self.model,
                        cached=True,
                    )
                    for choice in choices
                ]
                return [response.text for response in responses]
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}] + messages
            responses = await self._chat_completions_create(messages, **kwargs)
            self.cache[cache_key] = {"raw_response": responses[0].raw_response}
            return [response.text for response in responses]

        try:
            return asyncio.run(_async_generate_single(messages))
        except RuntimeError:
            # For running inside a notebook
            import nest_asyncio

            nest_asyncio.apply()
            event_loop = asyncio.get_event_loop()
            return event_loop.run_until_complete(_async_generate_single(messages))

    def generate(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        text_only: bool = True,
        silent: bool = False,
        **kwargs: Any,
    ) -> List[List[str]]:
        async def _async_generate() -> List[LLMResponse]:
            prompt_list = prompts if isinstance(prompts, list) else [prompts]

            pbar = tqdm.asyncio.tqdm(
                total=len(prompt_list), desc="Generating responses", disable=silent
            )
            try:
                all_responses = await self._check_cache_and_generate(
                    prompts=prompt_list, system_prompt=system_prompt, progress_bar=pbar, **kwargs
                )
            except Exception as e:
                print(f"Error processing prompts: {str(e)}")
                raise
            finally:
                pbar.close()

            if not silent:
                self.estimate_tokens(prompts, all_responses)

            if text_only:
                all_responses = [
                    [response.text for response in responses] for responses in all_responses
                ]
            return all_responses

        try:
            return asyncio.run(_async_generate())
        except RuntimeError:
            import nest_asyncio

            nest_asyncio.apply()
            event_loop = asyncio.get_event_loop()
            return event_loop.run_until_complete(_async_generate())
        except KeyboardInterrupt:
            print("Received interrupt signal, shutting down gracefully...")
            raise
        except Exception as e:
            print(f"Error during response generation: {str(e)}")
            raise


class OpenAIProvider(BaseProvider):
    provider = "openai"

    def __init__(self, model: str, api_key: str | None = None):
        super().__init__(model=model)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    @create_retry_decorator()
    async def _chat_completions_create(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> List[LLMResponse]:
        """OpenAI-specific implementation of chat completion."""
        # Handle max_tokens parameter for different model requirements
        if self.model in ["o1", "o3-mini", "o1-mini"]:
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return [
            LLMResponse(
                text=choice.message.content,
                raw_response=response.model_dump(),
                provider=self.provider,
                model=self.model,
                cached=False,
            )
            for choice in response.choices
        ]


class LocalProvider(BaseProvider):
    provider = "local"

    def __init__(self, model: str | None, host: str, port: int):
        super().__init__(model=model)
        self.base_url = f"http://{host}:{port}"
        self._check_server_health()
        self.available_models = self._get_available_models()
        if self.model is None:
            self.model = self.available_models[0]
            print(f"Using default model: {self.model}")
        assert self.model in self.available_models
        self.client = openai.AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key=None)

    def _check_server_health(self) -> None:
        """Check server health and verify model availability."""
        requests.get(f"{self.base_url}/health", timeout=5)
        print(f"Successfully connected to SGLang server at {self.base_url}")

    def _get_available_models(self) -> List[str]:
        """Retrieve list of available models from the server."""
        response = requests.get(f"{self.base_url}/v1/models", timeout=5)
        available_models = response.json()["data"]
        model_ids = [model["id"] for model in available_models]
        print(f"Available models: {model_ids}")
        return model_ids

    @create_retry_decorator()
    async def _chat_completions_create(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> List[LLMResponse]:
        """Local hosted implementation of chat completion with rate limiting."""
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return [
            LLMResponse(
                text=choice.message.content,
                raw_response=response.model_dump(),
                provider=self.provider,
                model=self.model,
                cached=False,
            )
            for choice in response.choices
        ]


class TogetherProvider(BaseProvider):
    provider = "together"

    def __init__(self, model: str, api_key: str | None = None):
        super().__init__(model=model)

        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key is required")
        self.client = Together(api_key=self.api_key)

    @create_retry_decorator()
    async def _chat_completions_create(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> List[LLMResponse]:
        """Together-specific implementation of chat completion."""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            ),
        )
        return [
            LLMResponse(
                text=choice.message.content,
                raw_response=response.model_dump(),
                provider=self.provider,
                model=self.model,
                cached=False,
            )
            for choice in response.choices
        ]


class Provider(BaseProvider):
    """Factory class to create the appropriate provider instance."""

    @classmethod
    def _get_provider_class(cls, model: str) -> Type[BaseProvider]:
        if model in ["gpt-4o-mini", "gpt-4o", "o3-mini", "o1-mini", "o1", "gpt-4.5-preview"]:
            print(f"Using OpenAI provider for {model}")
            return OpenAIProvider
        elif model in [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]:
            print(f"Using Together provider for {model}")
            return TogetherProvider
        else:
            print(f"Using Local provider for {model}")
            return LocalProvider

    def __new__(
        cls,
        model: str = None,
        api_key: Optional[str] = None,
        host: str = "iris-hgx-1.stanford.edu",
        port: int = 8000,
        **kwargs,
    ) -> BaseProvider:
        """Factory method to create the appropriate provider instance."""
        provider_class = cls._get_provider_class(model)

        if provider_class == LocalProvider:
            return provider_class(model=model, host=host, port=port, **kwargs)
        else:
            return provider_class(model=model, api_key=api_key, **kwargs)
