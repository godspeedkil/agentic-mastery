import pytest

from core.llm.errors import (
    ProviderUnavailableError,
    RateLimitedError,
    RequestRejectedError,
    RetryableError,
    error_from_status,
    parse_retry_after,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, RateLimitedError),
        (500, ProviderUnavailableError),
        (529, ProviderUnavailableError),
        (408, ProviderUnavailableError),
        (409, ProviderUnavailableError),
        (400, RequestRejectedError),
        (401, RequestRejectedError),
        (404, RequestRejectedError),
    ],
)
def test_status_codes_are_classified(status, expected):
    error = error_from_status("anthropic", status, "msg", {})
    assert type(error) is expected
    assert error.status_code == status
    assert error.provider == "anthropic"


def test_rate_limits_and_server_errors_are_retryable():
    assert isinstance(error_from_status("openai", 429, "m", {}), RetryableError)
    assert isinstance(error_from_status("openai", 503, "m", {}), RetryableError)
    assert not isinstance(error_from_status("openai", 400, "m", {}), RetryableError)


def test_should_retry_header_overrides_status():
    assert type(error_from_status("openai", 400, "m", {"X-Should-Retry": "true"})) is (
        ProviderUnavailableError
    )
    assert type(error_from_status("openai", 503, "m", {"x-should-retry": "false"})) is (
        RequestRejectedError
    )


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "2"}, 2.0),
        ({"Retry-After": "1.5"}, 1.5),
        ({"retry-after-ms": "250"}, 0.25),
        ({"retry-after-ms": "250", "retry-after": "9"}, 0.25),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({}, None),
    ],
)
def test_retry_after_is_parsed_in_seconds(headers, expected):
    assert parse_retry_after(headers) == expected


def test_retry_after_is_attached_to_the_error():
    assert error_from_status("anthropic", 429, "m", {"retry-after": "3"}).retry_after == 3.0
