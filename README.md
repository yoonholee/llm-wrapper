# LLM Wrapper

A flexible wrapper for LLM providers with caching, async support, and cost logging.

<https://pypi.org/project/llm-wrapper-yl/>

## Installation

```bash
pip install llm-wrapper-yl
```

## Quick Start

```python
from llm_wrapper import OpenAIProvider

# Initialize the provider
provider = OpenAIProvider(model="gpt-4o-mini", api_key="your-api-key")

# Generate a single response
response = provider.generate("What is the capital of France?")

# Generate multiple responses with caching
responses = provider.generate(
    ["What is 2+2?", "What is the weather?"],
    system_prompt="You are a helpful assistant.",
    temperature=0.7
)

# For the second call, you will retrieve responses from cache
responses = provider.generate(
    ["What is 2+2?", "What is the weather?"],
    system_prompt="You are a helpful assistant.",
    temperature=0.7
)
```

## Usage with Local Models

```python
from llm_wrapper import LocalProvider

# Initialize the provider with a local model
provider = LocalProvider(
    model="your-model-name",
    host="your-server-host",
    port=8000
)

# Generate responses
responses = provider.generate(["Your prompt here"])
```

## Testing

```bash
# Install local package with test dependencies
uv pip install '.[test]'
# Run tests with output
python -m pytest -s tests
```

## Deploying

```bash
hatch build
twine upload dist/*
```
