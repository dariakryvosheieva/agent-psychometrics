"""LLM API client with provider abstraction.

Supports both Anthropic and OpenAI providers with unified interface.
Uses the new OpenAI Responses API per project guidelines.
Supports both sync and async calls for parallelization.
"""

import asyncio
from typing import Optional

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


class LLMApiClient:
    """Unified API client for Anthropic and OpenAI.

    Provides a single interface for calling different LLM providers,
    with cost estimation and provider-specific configuration.

    Attributes:
        provider: LLM provider ("anthropic" or "openai")
        model: Model name to use
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-opus-4-5-20251101",
        "openai": "gpt-5.2",
    }

    # Pricing per 1M tokens (input, output) in USD
    # These are rough estimates and should be updated as pricing changes
    TOKEN_PRICING = {
        "anthropic": {
            "claude-opus-4-5-20251101": (15.0, 75.0),   # $15/1M input, $75/1M output
            "claude-sonnet-4-20250514": (3.0, 15.0),   # $3/1M input, $15/1M output
        },
        "openai": {
            "gpt-5.2": (2.5, 10.0),   # Estimated pricing
            "gpt-4o": (2.5, 10.0),    # $2.50/1M input, $10/1M output
        },
    }

    # Estimated tokens per task (rough averages for LLM judge prompts)
    # These are based on typical prompt sizes for SWE-bench and TerminalBench
    ESTIMATED_INPUT_TOKENS = 3000   # ~3K tokens for prompt + task context
    ESTIMATED_OUTPUT_TOKENS = 200   # ~200 tokens for JSON response

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None):
        """Initialize the API client.

        Args:
            provider: LLM provider ("anthropic" or "openai")
            model: Model name to use (defaults to provider's default model)

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider's SDK is not installed
        """
        if provider not in self.DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider}. Supported: {list(self.DEFAULT_MODELS.keys())}")

        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]

        # Validate SDK availability
        if provider == "anthropic" and not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        if provider == "openai" and not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")

    def call(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call the LLM and return response text.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Response text from the LLM

        Raises:
            Exception: If API call fails
        """
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, max_tokens)
        elif self.provider == "openai":
            return self._call_openai(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Call Anthropic API."""
        client = anthropic.Anthropic()

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()

    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API using the new Responses API.

        Note: The Responses API uses max_output_tokens instead of max_tokens.
        """
        client = openai.OpenAI()

        response = client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens,
        )

        return response.output_text.strip()

    async def call_async(self, prompt: str, max_tokens: int = 1024) -> str:
        """Async version of call() for parallel processing.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in response

        Returns:
            Response text from the LLM
        """
        if self.provider == "anthropic":
            return await self._call_anthropic_async(prompt, max_tokens)
        elif self.provider == "openai":
            # OpenAI async not implemented yet, fallback to sync in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._call_openai, prompt, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _call_anthropic_async(self, prompt: str, max_tokens: int) -> str:
        """Async call to Anthropic API."""
        client = anthropic.AsyncAnthropic()

        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()

    def estimate_cost(
        self,
        num_tasks: int,
        input_tokens_per_task: int = ESTIMATED_INPUT_TOKENS,
        output_tokens_per_task: int = ESTIMATED_OUTPUT_TOKENS,
    ) -> dict:
        """Estimate total cost for given number of tasks.

        Args:
            num_tasks: Number of tasks to process
            input_tokens_per_task: Estimated input tokens per task
            output_tokens_per_task: Estimated output tokens per task

        Returns:
            Dictionary with cost breakdown:
            - input_cost: Cost for input tokens
            - output_cost: Cost for output tokens
            - total_cost: Total estimated cost
            - input_tokens: Total input tokens
            - output_tokens: Total output tokens
            - pricing_note: Note about pricing source
        """
        total_input_tokens = num_tasks * input_tokens_per_task
        total_output_tokens = num_tasks * output_tokens_per_task

        # Get pricing for this model
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
