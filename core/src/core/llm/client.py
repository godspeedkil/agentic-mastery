"""One entry point for LLM calls: dispatch to a provider, retry, time, price and record."""

import random
import time
from collections.abc import Callable, Generator, Mapping
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, ValidationError

from core.llm.errors import RetryableError, StructuredOutputError
from core.llm.ledger import Ledger
from core.llm.pricing import PriceTable
from core.llm.records import CallRecord, Provider
from core.llm.types import Message, Request, Response, StreamEvent, TextDelta


class ProviderAdapter(Protocol):
    """What the client needs from a provider: a name, and a request completed or streamed."""

    name: Provider

    def complete(self, request: Request) -> Response: ...

    def stream(self, request: Request) -> Generator[StreamEvent, None, None]: ...


class Completion(BaseModel):
    """A response together with the record of the call that produced it."""

    model_config = ConfigDict(frozen=True)

    response: Response
    record: CallRecord


class Parsed[T: BaseModel](BaseModel):
    """A validated structured answer and every call it took to get it."""

    model_config = ConfigDict(frozen=True)

    value: T
    completions: list[Completion]
    """The first call, followed by any repair calls."""


UNPARSEABLE_STOP_REASONS = {"max_tokens", "refusal"}
"""Stop reasons after which the answer cannot be a complete, valid object."""


def repair_turns(answer: str, error: ValidationError) -> list[Message]:
    """The invalid answer as the assistant's turn, then the validation error as feedback."""
    feedback = f"Your answer did not validate:\n{error}\nReply with the corrected JSON only."
    return [Message.assistant(answer), Message.user(feedback)]


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

    def complete(
        self,
        provider: str,
        request: Request,
        *,
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
    ) -> Completion:
        """Send a request, retrying retryable errors; every attempt is recorded.

        Streaming (`stream=True`, implied by `on_text`) measures time to first token and
        passes each piece of text to `on_text` as it arrives. Once any text has been
        delivered, a failure is not retried, because the caller would see text twice.
        """
        adapter = self._providers.get(provider)
        if adapter is None:
            raise ValueError(f"unknown provider {provider!r}; have {sorted(self._providers)}")
        stream = stream or on_text is not None

        delivered = False

        def deliver(text: str) -> None:
            nonlocal delivered
            delivered = True
            if on_text is not None:
                on_text(text)

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return self._attempt(adapter, request, attempt, deliver if stream else None)
            except RetryableError as exc:
                if delivered or attempt == self._retry.max_attempts:
                    raise
                wait = self._retry.delay(attempt, exc.retry_after, self._rng)
                if wait is None:
                    raise
                self._sleep(wait)
        raise AssertionError("unreachable: the loop always returns or raises")

    def parse[T: BaseModel](
        self,
        provider: str,
        request: Request,
        output_type: type[T],
        *,
        max_repairs: int = 1,
    ) -> Parsed[T]:
        """Ask for an answer matching `output_type` and validate it.

        If validation fails, the error is sent back for a corrected answer, up to
        `max_repairs` times. A cut-off, refused or empty answer is not repaired.
        """
        if max_repairs < 0:
            raise ValueError("max_repairs cannot be negative")
        request = request.model_copy(update={"output_type": output_type})
        completions: list[Completion] = []
        last_error: ValidationError | None = None
        for _ in range(max_repairs + 1):
            completion = self.complete(provider, request)
            completions.append(completion)
            response = completion.response
            if response.stop_reason in UNPARSEABLE_STOP_REASONS:
                raise StructuredOutputError(
                    f"answer ended with stop reason {response.stop_reason!r}; not parsing it"
                )
            if not response.text:
                raise StructuredOutputError("answer contained no text")
            try:
                value = output_type.model_validate_json(response.text)
            except ValidationError as exc:
                last_error = exc
                request = request.model_copy(
                    update={"messages": [*request.messages, *repair_turns(response.text, exc)]}
                )
                continue
            return Parsed(value=value, completions=completions)
        raise StructuredOutputError(
            f"answer still invalid after {max_repairs} repair attempt(s)"
        ) from last_error

    def _attempt(
        self,
        adapter: ProviderAdapter,
        request: Request,
        attempt: int,
        deliver: Callable[[str], None] | None,
    ) -> Completion:
        """One try: call the provider and record the outcome, success or failure.

        `deliver` set means stream, passing text to it; None means a plain call.
        """
        started_at = datetime.now(UTC)
        start = self._clock()
        ttft_ms = None
        try:
            if deliver is None:
                response = adapter.complete(request)
            else:
                response, first_token_at = self._consume(adapter.stream(request), deliver)
                if first_token_at is not None:
                    ttft_ms = (first_token_at - start) * 1000
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
            ttft_ms=ttft_ms,
            stop_reason=response.stop_reason,
        )
        self._save(record)
        return Completion(response=response, record=record)

    def _consume(
        self, events: Generator[StreamEvent, None, None], deliver: Callable[[str], None]
    ) -> tuple[Response, float | None]:
        """Drain a stream: deliver text, note when the first piece arrived, return the Response."""
        first_token_at = None
        for event in events:
            if isinstance(event, TextDelta):
                if first_token_at is None:
                    first_token_at = self._clock()
                deliver(event.text)
            else:
                return event, first_token_at
        raise RuntimeError("stream ended without a final Response")

    def _save(self, record: CallRecord) -> None:
        if self._ledger is not None:
            self._ledger.append(record)
