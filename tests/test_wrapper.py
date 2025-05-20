import random
import string
import pytest
from typing import List
from llm_wrapper import Provider
import time

MODELS_TO_TEST = ["gpt-4o-mini", "o3-mini", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]


@pytest.fixture
def test_prompts() -> List[str]:
    """Fixture to generate test prompts."""
    all_chars = string.ascii_letters + string.digits + string.punctuation
    random_string = "".join(random.choices(all_chars, k=5))
    reverse_prompt = "Reverse the string: " + random_string
    return ["What is the capital of France?", reverse_prompt]


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_wrapper(model_name: str, test_prompts: List[str]) -> None:
    """Test different models with the same prompts."""
    print(f"\n\n\n{'='*20} Testing {model_name} {'='*20}")

    start = time.time()
    provider = Provider(model=model_name)
    generation_kwargs = {"temperature": 1.0, "n": 1, "max_tokens": 1000}

    all_responses = provider.generate(prompts=test_prompts, **generation_kwargs)

    # Basic validation
    assert len(all_responses) == len(test_prompts)

    # Print and validate each response
    for responses, prompt in zip(all_responses, test_prompts):
        assert isinstance(responses, list)
        assert len(responses) == generation_kwargs["n"]
        assert all(isinstance(r, str) for r in responses)
        assert all(len(r) > 0 for r in responses)

        print(f"\nPrompt: {prompt[:50]}...")
        for i, response in enumerate(responses, 1):
            print(f"Response {i}: {response[:100]}...")
        print(f"Token count: {len(response.split())}")  # Simple token count

    print(f"Time taken: {time.time() - start} seconds")


def test_cache():
    print(f"\n\n\n{'='*20} Testing cache {'='*20}")
    prompts = ["What is the capital of France?"]
    provider = Provider(model="gpt-4o-mini")
    provider.generate(prompts, force_new=True)
    provider.generate(prompts * 1000)  # Should be cached


def test_force_new():
    print(f"\n\n\n{'='*20} Testing force_new {'='*20}")
    prompts = ["What is the capital of France?"]
    provider = Provider(model="gpt-4o-mini")
    provider.generate(prompts, force_new=True)
    provider.generate(prompts, force_new=True)  # Should be cached
