"""Configuration settings for LLM wrapper."""

from typing import Dict

import openai
import requests

# Model costs in USD per 1M tokens
# Source: https://platform.openai.com/docs/pricing#latest-models
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o1": {"input": 15.00, "output": 60.00},
    "gpt-4.5-preview": {"input": 75.0, "output": 150.0},
}

RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 0.1,
    "max_delay": 60.0,
    "retry_exceptions": (
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        requests.exceptions.RequestException,
    ),
}
