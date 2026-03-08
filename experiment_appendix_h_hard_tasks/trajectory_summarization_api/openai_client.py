"""Async OpenAI client using Responses API with rate limiting and retry logic."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import openai
from openai import AsyncOpenAI


@dataclass
class UsageStats:
    """Track API usage statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_errors: int = 0
    model: str = "gpt-5-mini"

    def add_call(self, input_tokens: int, output_tokens: int):
        """Record a successful API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def add_error(self):
        """Record a failed API call."""
        self.total_errors += 1

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD for gpt-5-mini pricing.

        Pricing: $0.25/1M input, $2.00/1M output
        """
        input_cost = self.total_input_tokens * 0.25 / 1_000_000
        output_cost = self.total_output_tokens * 2.00 / 1_000_000
        return input_cost + output_cost

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "model": self.model,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }


class RateLimiter:
    """Token bucket rate limiter for RPM limits."""

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if needed to respect rate limiting."""
        async with self._lock:
            now = time.monotonic()
            wait_time = self.last_request_time + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request_time = time.monotonic()


class AsyncOpenAIClient:
    """Async OpenAI client using Responses API with rate limiting and retry logic."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        max_concurrent: int = 200,
        requests_per_minute: int = 5000,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
    ):
        """Initialize the async client.

        Args:
            model: OpenAI model ID
            max_concurrent: Maximum concurrent requests (semaphore limit)
            requests_per_minute: Rate limit for requests
            max_retries: Maximum retry attempts per request
            base_retry_delay: Base delay for exponential backoff
        """
        self.client = AsyncOpenAI()
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.usage = UsageStats(model=model)

    async def summarize(
        self,
        prompt: str,
        max_tokens: int = 1000,
    ) -> Optional[str]:
        """Make API call using Responses API with retry logic and rate limiting.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum output tokens

        Returns:
            Response text or None if all retries failed

        Note:
            Temperature parameter is not supported with gpt-5-mini model.
        """
        async with self.semaphore:
            await self.rate_limiter.acquire()

            last_error = None
            for attempt in range(self.max_retries):
                try:
                    # Use the new Responses API instead of chat.completions
                    # Note: temperature is not supported for gpt-5-mini
                    response = await self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        max_output_tokens=max_tokens,
                    )

                    # Track usage
                    self.usage.add_call(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    )

                    # Check response status - retry if incomplete
                    status = getattr(response, 'status', 'completed')
                    if status == 'incomplete':
                        import logging
                        logger = logging.getLogger(__name__)
                        incomplete_reason = getattr(response, 'incomplete_details', {})
                        logger.warning(
                            f"Incomplete response (attempt {attempt + 1}), "
                            f"reason: {incomplete_reason}. Retrying..."
                        )
                        # Treat as retryable error
                        await asyncio.sleep(self.base_retry_delay * (2 ** attempt))
                        continue

                    # Responses API uses output_text instead of choices[0].message.content
                    return response.output_text

                except openai.RateLimitError as e:
                    last_error = e
                    # Longer backoff for rate limits: 10s, 20s, 40s
                    wait_time = (2 ** attempt) * 10
                    await asyncio.sleep(wait_time)

                except openai.APIError as e:
                    last_error = e
                    self.usage.add_error()
                    # Shorter backoff for other errors: 1s, 2s, 4s
                    wait_time = self.base_retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)

            self.usage.add_error()
            return None
