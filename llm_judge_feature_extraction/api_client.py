"""LLM API client with provider abstraction and prefix caching.

Supports both Anthropic and OpenAI providers with unified interface.
Uses the new OpenAI Responses API per project guidelines.
Designed for batched extraction: prefix (cacheable) + suffix (varies per batch).
"""

import asyncio
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Try to import API clients
try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Rough context limits per model (in tokens)
MODEL_CONTEXT_LIMITS = {
    "gpt-5.4-2026-03-05": 128_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
}

# Default context limit for unknown models
DEFAULT_CONTEXT_LIMIT = 128_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def validate_prompt_length(
    text: str,
    model: str,
    max_fraction: float = 0.9,
) -> None:
    """Raise if prompt is too close to model's context limit.

    Args:
        text: Full prompt text
        model: Model name
        max_fraction: Maximum fraction of context to use (default 0.9)

    Raises:
        ValueError: If estimated token count exceeds max_fraction of context limit
    """
    estimated_tokens = estimate_tokens(text)
    context_limit = MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)
    max_tokens = int(context_limit * max_fraction)

    if estimated_tokens > max_tokens:
        raise ValueError(
            f"Prompt too long for model '{model}': ~{estimated_tokens:,} tokens "
            f"(estimated) exceeds {max_fraction:.0%} of {context_limit:,} context limit "
            f"({max_tokens:,} tokens). Shorten the prompt or use a model with a "
            f"larger context window."
        )


class LLMApiClient:
    """Unified API client for Anthropic and OpenAI with prefix caching.

    The primary interface is call_with_prefix(prefix, suffix) which enables
    prompt caching: multiple calls sharing the same prefix benefit from
    cached prefix processing.

    - OpenAI: prefix + suffix concatenated as single input (auto prefix caching)
    - Anthropic: prefix as system message with cache_control, suffix as user message
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-opus-4-6",
        "openai": "gpt-5.4-2026-03-05",
    }

    # Pricing per 1M tokens (input, output) in USD
    TOKEN_PRICING = {
        "anthropic": {
            "claude-opus-4-6": (5.0, 25.0),
            "claude-sonnet-4-6": (3.0, 15.0),
        },
        "openai": {
            "gpt-5.4-2026-03-05": (2.5, 10.0),
        },
    }

    ESTIMATED_INPUT_TOKENS = 3000
    ESTIMATED_OUTPUT_TOKENS = 200

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """Initialize the API client.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name (defaults to provider's default model)

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider's SDK is not installed
        """
        if provider not in self.DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider}. Supported: {list(self.DEFAULT_MODELS.keys())}")

        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]

        if provider == "anthropic" and not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        if provider == "openai" and not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")

    def call_with_prefix(
        self,
        prefix: str,
        suffix: str,
        max_tokens: int = 1024,
    ) -> str:
        """Call the LLM with a cacheable prefix and varying suffix.

        Args:
            prefix: Cacheable portion (system intro + task info)
            suffix: Varying portion (feature scales + output format)
            max_tokens: Maximum tokens in response

        Returns:
            Response text from the LLM

        Raises:
            ValueError: If combined prompt exceeds model context limit
        """
        full_text = prefix + "\n\n" + suffix
        validate_prompt_length(full_text, self.model)

        if self.provider == "anthropic":
            return self._call_anthropic_cached(prefix, suffix, max_tokens)
        else:
            return self._call_openai_cached(prefix, suffix, max_tokens)

    async def call_with_prefix_async(
        self,
        prefix: str,
        suffix: str,
        max_tokens: int = 1024,
    ) -> str:
        """Async version of call_with_prefix."""
        full_text = prefix + "\n\n" + suffix
        validate_prompt_length(full_text, self.model)

        if self.provider == "anthropic":
            return await self._call_anthropic_cached_async(prefix, suffix, max_tokens)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._call_openai_cached, prefix, suffix, max_tokens
            )

    def _call_anthropic_cached(self, prefix: str, suffix: str, max_tokens: int) -> str:
        """Call Anthropic with prefix as cached system message."""
        client = anthropic.Anthropic()

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": suffix}],
        )

        return response.content[0].text.strip()

    async def _call_anthropic_cached_async(self, prefix: str, suffix: str, max_tokens: int) -> str:
        """Async call to Anthropic with prefix caching."""
        client = anthropic.AsyncAnthropic()

        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": suffix}],
        )

        return response.content[0].text.strip()

    def _call_openai_cached(self, prefix: str, suffix: str, max_tokens: int) -> str:
        """Call OpenAI with concatenated prompt (auto prefix caching)."""
        client = openai.OpenAI()

        full_prompt = prefix + "\n\n" + suffix
        response = client.responses.create(
            model=self.model,
            input=full_prompt,
            max_output_tokens=max_tokens,
        )

        return response.output_text.strip()

    def estimate_cost(
        self,
        num_calls: int,
        input_tokens_per_call: int = ESTIMATED_INPUT_TOKENS,
        output_tokens_per_call: int = ESTIMATED_OUTPUT_TOKENS,
    ) -> dict:
        """Estimate total cost for given number of API calls.

        Args:
            num_calls: Number of API calls
            input_tokens_per_call: Estimated input tokens per call
            output_tokens_per_call: Estimated output tokens per call

        Returns:
            Dictionary with cost breakdown
        """
        total_input_tokens = num_calls * input_tokens_per_call
        total_output_tokens = num_calls * output_tokens_per_call

        provider_pricing = self.TOKEN_PRICING.get(self.provider, {})
        input_price_per_1m, output_price_per_1m = provider_pricing.get(
            self.model,
            (5.0, 15.0)  # Default fallback pricing
        )

        input_cost = (total_input_tokens / 1_000_000) * input_price_per_1m
        output_cost = (total_output_tokens / 1_000_000) * output_price_per_1m
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "pricing_note": f"Estimated: ${input_price_per_1m}/1M input, ${output_price_per_1m}/1M output",
        }

    def get_info(self) -> str:
        """Return human-readable info about the client configuration."""
        return f"Provider: {self.provider}, Model: {self.model}"
