"""LLM Wrapper with caching, async, retries, and cost tracking."""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Type

import diskcache as dc
from together import Together
import openai
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
    max_concurrent_requests: int = 64


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


class BaseProvider(ABC):
    """A wrapper for interacting with LLM providers. Supports local caching, async, retries, and cost tracking."""

    def __init__(self, model: str, config: LLMConfig = LLMConfig()):
        self.model = model
        self.config = config

        cache_root = get_cache_root()
        cwd = Path.cwd().name
        cache_dir = os.path.join(cache_root, "llm_wrapper", cwd)
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

    def estimate_tokens(self, prompts, all_responses):
        token_counter = TokenCounter(self.model)
        for prompt, responses in zip(prompts, all_responses):
            token_counter.update(prompt, responses)
        token_counter.print_cost()

    def _generate_cache_key(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate deterministic cache key for a single message."""
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

    def get_cached_response(self, prompt: str, system_prompt: str = "", **kwargs) -> LLMResponse:
        cache_key = self._generate_cache_key(prompt, system_prompt, **kwargs)
        cached_response = self.cache.get(cache_key)
        if cached_response == None:
            return None
        choices = cached_response["raw_response"]["choices"]
        response_list = [
            LLMResponse(
                text=choice["message"]["content"],
                raw_response=cached_response["raw_response"],
                provider=self.provider,
                model=self.model,
                cached=True,
            )
            for choice in choices
        ]
        return response_list

    def generate(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        text_only: bool = True,
        silent: bool = False,
        force_new: bool = False,
        **kwargs: Any,
    ) -> List[List[str]] | List[List[LLMResponse]]:
        prompts = prompts if isinstance(prompts, list) else [prompts]
        if force_new:
            cached_responses = [None] * len(prompts)
        else:
            cached_responses = [
                self.get_cached_response(prompt, system_prompt, **kwargs) for prompt in prompts
            ]

        uncached_idxs = [i for i, response in enumerate(cached_responses) if response is None]
        if len(uncached_idxs) == 0:
            all_responses = cached_responses
            if text_only:
                all_responses = [
                    [response.text for response in responses] for responses in all_responses
                ]
            return all_responses

        uncached_prompts = [prompts[i] for i in uncached_idxs]
        uncached_responses = self.generate_uncached(
            uncached_prompts, system_prompt, silent, **kwargs
        )
        uncached_keys = [
            self._generate_cache_key(uncached_prompts[i], system_prompt, **kwargs)
            for i in range(len(uncached_prompts))
        ]
        for key, response in zip(uncached_keys, uncached_responses):
            self.cache[key] = {"raw_response": response[0].raw_response}

        all_responses = []
        for i in range(len(prompts)):
            if cached_responses[i] is not None:
                all_responses.append(cached_responses[i])
            else:
                all_responses.append(uncached_responses[uncached_idxs.index(i)])
        if text_only:
            all_responses = [
                [response.text for response in responses] for responses in all_responses
            ]
        return all_responses

    @abstractmethod
    def generate_uncached(
        self, prompts: List[str], system_prompt: str = "", **kwargs
    ) -> List[List[LLMResponse]]:
        pass


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


class APIProviderMixin:
    """A mixin for interacting with LLM providers that use API calls."""

    async def _generate_single(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> List[LLMResponse]:
        """Generate a single chat completion with caching."""
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        responses = await self._chat_completions_create(messages, **kwargs)
        return responses

    async def _generate(
        self, prompts: List[str], system_prompt: str = "", **kwargs
    ) -> List[List[LLMResponse]]:
        semaphore = asyncio.Semaphore(value=self.config.max_concurrent_requests)
        pbar = kwargs.pop("progress_bar", None)

        async def run_single(prompt: str) -> List[LLMResponse]:
            async with semaphore:
                response = await self._generate_single(prompt, system_prompt, **kwargs)
                if pbar:
                    pbar.update(1)
                return response

        return await asyncio.gather(*[run_single(p) for p in prompts])

    def generate_uncached(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> List[List[str]] | List[List[LLMResponse]]:
        async def _async_generate() -> List[List[LLMResponse]]:
            prompt_list = prompts if isinstance(prompts, list) else [prompts]

            pbar = tqdm.asyncio.tqdm(
                total=len(prompt_list), desc="Generating responses", disable=silent
            )
            all_responses = await self._generate(
                prompts=prompt_list, system_prompt=system_prompt, progress_bar=pbar, **kwargs
            )
            pbar.close()

            if not silent:
                self.estimate_tokens(prompts, all_responses)
            return all_responses

        try:
            try:
                loop = asyncio.get_running_loop()
                is_running = loop.is_running()
            except RuntimeError:
                loop = None
                is_running = False

            if is_running:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(_async_generate())
            else:
                return asyncio.run(_async_generate())
        except KeyboardInterrupt:
            print("Received interrupt signal, shutting down gracefully...")
            raise
        except Exception as e:
            print(f"Error during response generation: {str(e)}")
            raise


class OpenAIProvider(APIProviderMixin, BaseProvider):
    provider = "openai"
    reasoning_models = ["o1", "o3", "o1-mini", "o3-mini", "o4-mini"]

    def __init__(self, model: str):
        super().__init__(model)
        self.client = openai.AsyncOpenAI()

    @create_retry_decorator()
    async def _chat_completions_create(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> List[LLMResponse]:
        """OpenAI-specific implementation of chat completion."""
        # Handle max_tokens parameter for different model requirements
        if self.model in self.reasoning_models:
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


class TogetherProvider(APIProviderMixin, BaseProvider):
    provider = "together"

    def __init__(self, model: str):
        super().__init__(model=model)
        self.client = Together()

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


class Provider:
    """Factory class to create the appropriate provider instance."""

    @classmethod
    def _get_provider_class(cls, model: Optional[str]) -> Type[BaseProvider]:
        client = Together()
        together_models = [model.id for model in client.models.list()]

        if model in OpenAIProvider.reasoning_models or "gpt" in model:
            print(f"Using OpenAI provider for {model}")
            return OpenAIProvider
        elif model in together_models:
            print(f"Using Together provider for {model}")
            return TogetherProvider
        else:
            raise ValueError(f"Unsupported model: {model}")

    def __new__(cls, model: str = None, **kwargs) -> BaseProvider:
        """Factory method to create the appropriate provider instance."""
        provider_class = cls._get_provider_class(model)
        return provider_class(model=model, **kwargs)
