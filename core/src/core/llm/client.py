"""One entry point for LLM calls: dispatch to a provider, retry, time, price and record."""

import random
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from core.llm.errors import RetryableError
from core.llm.ledger import Ledger
from core.llm.pricing import PriceTable
from core.llm.records import CallRecord, Provider
from core.llm.types import Request, Response


class ProviderAdapter(Protocol):
    """What the client needs from a provider: a name and a way to complete a request."""

    name: Provider

    def complete(self, request: Request) -> Response: ...


class Completion(BaseModel):
    """A response together with the record of the call that produced it."""

    model_config = ConfigDict(frozen=True)

    response: Response
    record: CallRecord


class RetryPolicy(BaseModel):
    """Exponential backoff with full jitter for retryable errors."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    max_attempts: PositiveInt = 4
    """Total tries including the first; 1 disables retries."""
    base_delay: PositiveFloat = 1.0
    """Seconds; the backoff ceiling doubles after each failed attempt."""
    max_delay: PositiveFloat = 30.0
    """Longest wait between attempts, in seconds."""

    def delay(self, attempt: int, retry_after: float | None, rng: random.Random) -> float | None:
        """Seconds to wait after a failed attempt, or None to stop retrying."""
        if retry_after is not None:
            return retry_after if retry_after <= self.max_delay else None
        ceiling = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return rng.uniform(0, ceiling)


class LLMClient:
    def __init__(
        self,
        providers: Mapping[str, ProviderAdapter],
        prices: PriceTable | None = None,
        ledger: Ledger | None = None,
        run_id: str | None = None,
        retry: RetryPolicy | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._prices = prices
        self._ledger = ledger
        self._run_id = run_id
        self._retry = retry or RetryPolicy()
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()

    def complete(self, provider: str, request: Request) -> Completion:
        """Send a request, retrying retryable errors; every attempt is recorded."""
        adapter = self._providers.get(provider)
        if adapter is None:
            raise ValueError(f"unknown provider {provider!r}; have {sorted(self._providers)}")

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return self._attempt(adapter, request, attempt)
            except RetryableError as exc:
                if attempt == self._retry.max_attempts:
                    raise
                wait = self._retry.delay(attempt, exc.retry_after, self._rng)
                if wait is None:
                    raise
                self._sleep(wait)
        raise AssertionError("unreachable: the loop always returns or raises")

    def _attempt(self, adapter: ProviderAdapter, request: Request, attempt: int) -> Completion:
        """One try: call the provider and record the outcome, success or failure."""
        started_at = datetime.now(UTC)
        start = self._clock()
        try:
            response = adapter.complete(request)
        except Exception as exc:
            self._save(
                CallRecord(
                    started_at=started_at,
                    run_id=self._run_id,
                    provider=adapter.name,
                    model=request.model,
                    attempt=attempt,
                    latency_ms=(self._clock() - start) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        latency_ms = (self._clock() - start) * 1000

        record = CallRecord(
            started_at=started_at,
            run_id=self._run_id,
            provider=adapter.name,
            model=request.model,
            attempt=attempt,
            usage=response.usage,
            cost_usd=self._prices.cost(request.model, response.usage) if self._prices else None,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
        )
        self._save(record)
        return Completion(response=response, record=record)

    def _save(self, record: CallRecord) -> None:
        if self._ledger is not None:
            self._ledger.append(record)
