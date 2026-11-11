"""One entry point for LLM calls: dispatch to a provider, time, price and record each call."""

import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

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


class LLMClient:
    def __init__(
        self,
        providers: Mapping[str, ProviderAdapter],
        prices: PriceTable | None = None,
        ledger: Ledger | None = None,
        run_id: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._providers = dict(providers)
        self._prices = prices
        self._ledger = ledger
        self._run_id = run_id
        self._clock = clock

    def complete(self, provider: str, request: Request) -> Completion:
        """Send a request; record the call whether it succeeds or fails."""
        adapter = self._providers.get(provider)
        if adapter is None:
            raise ValueError(f"unknown provider {provider!r}; have {sorted(self._providers)}")

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
