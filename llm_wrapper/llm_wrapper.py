"""LLM Wrapper with caching, async, retries, and cost tracking."""
import transformers
from vllm import LLM, SamplingParams
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

@dataclass
class LLMConfig:
    """Configuration for LLM providers and caching"""

    cache_size_gb: int = 2
    cache_policy: str = "least-recently-used"
    max_concurrent_requests: int = 64


class TokenCounter:
    CHARS_PER_TOKEN = 4

    def __init__(self, model: str):
        costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        self.input_cost, self.output_cost = costs["input"], costs["output"]
        self.input_total = self.output_total = self.input_new = self.output_new = 0

    @staticmethod
    def format_token_count(tokens: float) -> str:
        return (
            f"{tokens/1_000_000:.2f}M"
            if tokens > 1_000_000
            else f"{tokens/1000:.1f}k" if tokens > 1000 else str(int(tokens))
        )

    def _update(self, prompt, response):
        if "usage" in response.raw_response:
            usage = response.raw_response["usage"]
            input_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
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
        def calc_stats(input_tokens, output_tokens):
            cost = (input_tokens * self.input_cost + output_tokens * self.output_cost) / 1_000_000
            return (
                cost,
                self.format_token_count(input_tokens),
                self.format_token_count(output_tokens),
            )

        total_cost, total_in, total_out = calc_stats(self.input_total, self.output_total)
        new_cost, new_in, new_out = calc_stats(self.input_new, self.output_new)

        print(f"Total: ${total_cost:.4f} ({total_in} input, {total_out} output)")
        print(f"New: ${new_cost:.4f} ({new_in} input, {new_out} output)")


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
        self.token_counter = TokenCounter(model)

        cache_root = os.getenv("LLM_WRAPPER_CACHE_ROOT", os.path.expanduser("~/.cache"))
        model_name_safe = model.replace("/", "_")[-20:]
        current_dir = os.path.basename(os.getcwd())
        cache_dir = f"{cache_root}/llm_wrapper/{current_dir}/{model_name_safe}"
        os.makedirs(cache_dir, exist_ok=True)

        self.cache = dc.Cache(
            cache_dir,
            size_limit=self.config.cache_size_gb * BYTES_PER_GB,
            eviction_policy=self.config.cache_policy,
        )
        print(f"Cache path: {self.cache.directory}")
        print(
            f"Current cache size: {self.cache.volume() / BYTES_PER_GB:.2f}/{self.config.cache_size_gb}GB"
        )

    def _generate_cache_key(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generate deterministic cache key for a single message."""
        components = [
            self.provider,
            self.model,
            (
                hashlib.sha256(system_prompt.encode()).hexdigest()
                if system_prompt
                else "no_system_prompt"
            ),
            hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            hashlib.sha256(
                "|".join(f"{k}:{str(v)}" for k, v in sorted(kwargs.items())).encode()
            ).hexdigest(),
        ]
        return hashlib.sha256("|".join(components).encode()).hexdigest()

    def get_cached_response(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> Optional[List[LLMResponse]]:
        cached = self.cache.get(self._generate_cache_key(prompt, system_prompt, **kwargs))
        if not cached:
            return None
        return [
            LLMResponse(
                text=choice["message"]["content"],
                raw_response=cached["raw_response"],
                provider=self.provider,
                model=self.model,
                cached=True,
            )
            for choice in cached["raw_response"]["choices"]
        ]

    def generate(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        text_only: bool = True,
        silent: bool = False,
        force_new: bool = False,
        **kwargs: Any,
    ) -> Union[List[List[str]], List[List[LLMResponse]]]:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        cached_responses = [
            None if force_new else self.get_cached_response(p, system_prompt, **kwargs)
            for p in prompts
        ]

        uncached_idxs = [i for i, r in enumerate(cached_responses) if r is None]
        if uncached_idxs:
            uncached_prompts = [prompts[i] for i in uncached_idxs]
            new_responses = self.generate_uncached(
                uncached_prompts, system_prompt, silent, **kwargs
            )
            for i, resp in zip(uncached_idxs, new_responses):
                cached_responses[i] = resp
                key = self._generate_cache_key(prompts[i], system_prompt, **kwargs)
                self.cache[key] = {"raw_response": resp[0].raw_response}

        return [
            [r.text for r in responses] if text_only else responses
            for responses in cached_responses
        ]

    @abstractmethod
    def generate_uncached(
        self, prompts: List[str], system_prompt: str = "", **kwargs
    ) -> List[List[LLMResponse]]:
        pass

    def estimate_tokens(
        self, prompts: Union[str, List[str]], responses: List[List[LLMResponse]]
    ) -> None:
        """Update token counter with prompt and response tokens."""
        prompts = [prompts] if isinstance(prompts, str) else prompts
        for prompt, response_list in zip(prompts, responses):
            self.token_counter.update(prompt, response_list)

    def print_cost(self) -> None:
        self.token_counter.print_cost()


def create_retry_decorator(
    max_retries: int = RETRY_CONFIG["max_retries"],
    base_delay: float = RETRY_CONFIG["base_delay"],
    max_delay: float = RETRY_CONFIG["max_delay"],
) -> Callable:
    """Create a retry decorator with exponential backoff."""
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
                self.print_cost()
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


class vLLMProvider(BaseProvider):
    provider = "vllm"

    def __init__(
        self, model: str, context_length: int = 1024, compile_model: bool = False, **kwargs
    ):
        self.compile_model = compile_model
        self.set_vllm_environment()
        super().__init__(model=model)
        self.context_length = context_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.llm = None

    def lazy_load_model(self):
        if self.llm is None:
            self.llm = LLM(
                model=self.model,
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=self.context_length,
                max_seq_len_to_capture=self.context_length,
                enforce_eager=not self.compile_model
            )

    def set_vllm_environment(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        local_scratch = "/scr-ssd" if os.path.exists("/scr-ssd") else "/scr"
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(local_scratch, "yoonho", "vllm_cache")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0;8.9+PTX"
        if "TRANSFORMERS_CACHE" in os.environ:
            os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
            del os.environ["TRANSFORMERS_CACHE"]

        if self.compile_model:
            os.environ["VLLM_USE_PRECOMPILED"] = "1"
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            os.environ["TORCH_DYNAMO_DISABLE"] = "1"
            os.environ["VLLM_USE_PRECOMPILED"] = "0"
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "0"
            os.environ["VLLM_ATTENTION_BACKEND"] = "PYTORCH"

    def generate_uncached(
        self, prompts: List[str], system_prompt: str = "", silent: bool = False, **kwargs: Any
    ) -> List[List[LLMResponse]]:
        """Generate responses for uncached prompts using vLLM."""
        self.lazy_load_model()  # Ensure model is loaded

        conversations = []
        for prompt in prompts:
            conversation = [{"role": "user", "content": prompt}]
            if system_prompt:
                conversation = [{"role": "system", "content": system_prompt}] + conversation
            conversations.append(conversation)
        conversations_chat = self.tokenizer.apply_chat_template(conversations, tokenize=False)

        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.context_length),
            temperature=kwargs.get("temperature", 0.6),
            top_p=kwargs.get("top_p", 1.0),
            n=kwargs.get("n", 1),
        )

        all_responses = self.llm.generate(conversations_chat, sampling_params=sampling_params)
        llm_response_list = []
        for prompt_outputs in all_responses:
            llm_responses = []
            choices = []
            for output in prompt_outputs.outputs:
                choices.append({
                    "message": {"content": output.text, "role": "assistant"},
                    "finish_reason": output.finish_reason
                })
                
            raw_response = {
                "choices": choices,
                "model": self.model,
                "usage": {
                    "prompt_tokens": len(self.tokenizer.encode(prompt_outputs.prompt)),
                    "completion_tokens": sum(len(self.tokenizer.encode(output.text)) for output in prompt_outputs.outputs),
                    "total_tokens": len(self.tokenizer.encode(prompt_outputs.prompt)) + 
                                  sum(len(self.tokenizer.encode(output.text)) for output in prompt_outputs.outputs)
                },
            }
            
            for output in prompt_outputs.outputs:
                llm_responses.append(
                    LLMResponse(
                        text=output.text,
                        raw_response=raw_response,
                        provider=self.provider,
                        model=self.model,
                        cached=False,
                    )
                )
            llm_response_list.append(llm_responses)
        
        if not silent:
            self.estimate_tokens(prompts, llm_response_list)
            self.print_cost()
            
        return llm_response_list


class Provider:
    """Factory class to create the appropriate provider instance."""

    @classmethod
    def _get_provider_class(cls, model: str) -> Type[BaseProvider]:
        together_models = [m.id for m in Together().models.list()]
        if model in OpenAIProvider.reasoning_models or "gpt" in model:
            return OpenAIProvider
        elif model in together_models:
            return TogetherProvider
        else:
            return vLLMProvider

    def __new__(cls, model: str = None, **kwargs) -> BaseProvider:
        return cls._get_provider_class(model)(model=model, **kwargs)
