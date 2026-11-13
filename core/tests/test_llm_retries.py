import random

import pytest

from core.llm import (
    Ledger,
    LLMClient,
    Message,
    Request,
    Response,
    RetryPolicy,
    TextBlock,
    Usage,
    read_records,
)
from core.llm.errors import ProviderUnavailableError, RateLimitedError, RequestRejectedError

REQUEST = Request(model="m", messages=[Message.user("hi")])


class FlakyProvider:
    """Raises the given errors in order, then answers."""

    name = "anthropic"

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    def complete(self, request: Request) -> Response:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return Response(
            model="m", content=[TextBlock(text="ok")], stop_reason="end_turn", usage=Usage()
        )


def unavailable() -> ProviderUnavailableError:
    return ProviderUnavailableError("overloaded", provider="anthropic", status_code=529)


def make_client(provider, tmp_path, **kwargs) -> tuple[LLMClient, list[float]]:
    sleeps: list[float] = []
    client = LLMClient(
        {"anthropic": provider},
        ledger=Ledger(tmp_path / "run.jsonl"),
        sleep=sleeps.append,
        rng=random.Random(0),
        **kwargs,
    )
    return client, sleeps


def test_retryable_errors_are_retried_and_every_attempt_recorded(tmp_path):
    provider = FlakyProvider(unavailable(), unavailable())
    client, sleeps = make_client(provider, tmp_path)

    completion = client.complete("anthropic", REQUEST)

    assert completion.response.text == "ok"
    assert completion.record.attempt == 3
    records = read_records(tmp_path / "run.jsonl")
    assert [r.attempt for r in records] == [1, 2, 3]
    assert [r.error is not None for r in records] == [True, True, False]
    assert len(sleeps) == 2


def test_non_retryable_errors_fail_immediately(tmp_path):
    provider = FlakyProvider(RequestRejectedError("bad model", provider="anthropic"))
    client, sleeps = make_client(provider, tmp_path)

    with pytest.raises(RequestRejectedError):
        client.complete("anthropic", REQUEST)
    assert provider.calls == 1
    assert sleeps == []


def test_gives_up_after_max_attempts(tmp_path):
    provider = FlakyProvider(*(unavailable() for _ in range(5)))
    client, sleeps = make_client(provider, tmp_path, retry=RetryPolicy(max_attempts=3))

    with pytest.raises(ProviderUnavailableError):
        client.complete("anthropic", REQUEST)
    assert provider.calls == 3
    assert len(sleeps) == 2
    assert len(read_records(tmp_path / "run.jsonl")) == 3


def test_server_retry_after_is_honored(tmp_path):
    limited = RateLimitedError("slow down", provider="anthropic", retry_after=7.0)
    client, sleeps = make_client(FlakyProvider(limited), tmp_path)

    client.complete("anthropic", REQUEST)
    assert sleeps == [7.0]


def test_retry_after_beyond_max_delay_stops_retrying(tmp_path):
    limited = RateLimitedError("slow down", provider="anthropic", retry_after=120.0)
    provider = FlakyProvider(limited)
    client, sleeps = make_client(provider, tmp_path)

    with pytest.raises(RateLimitedError):
        client.complete("anthropic", REQUEST)
    assert provider.calls == 1
    assert sleeps == []


def test_one_attempt_disables_retries(tmp_path):
    provider = FlakyProvider(unavailable())
    client, _ = make_client(provider, tmp_path, retry=RetryPolicy(max_attempts=1))
    with pytest.raises(ProviderUnavailableError):
        client.complete("anthropic", REQUEST)
    assert provider.calls == 1


@pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5, 6, 7])
def test_backoff_stays_within_its_doubling_ceiling(attempt):
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
    rng = random.Random(attempt)
    ceiling = min(30.0, 2 ** (attempt - 1))
    for _ in range(100):
        assert 0 <= policy.delay(attempt, None, rng) <= ceiling
