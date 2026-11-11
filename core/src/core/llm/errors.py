"""Provider-independent errors, classified by whether retrying the same request can help."""

from collections.abc import Mapping

from core.llm.records import Provider


class LLMError(Exception):
    """Base class for errors raised by provider adapters."""

    def __init__(
        self,
        message: str,
        *,
        provider: Provider,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        """Seconds the server asked us to wait before retrying, if it said."""


class RetryableError(LLMError):
    """The same request may succeed if sent again later."""


class RateLimitedError(RetryableError):
    """HTTP 429: too many requests or tokens for the account's limits."""


class ProviderUnavailableError(RetryableError):
    """Server errors, overload, timeouts and network failures."""


class RequestRejectedError(LLMError):
    """The request itself is wrong (bad input, auth, unknown model); retrying won't help."""


RETRYABLE_CLIENT_STATUSES = {408, 409}
"""Request timeout and lock conflict: client-range codes that are worth retrying."""


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds from `retry-after-ms` or `retry-after`; None if absent or not a number."""
    lowered = {key.lower(): value for key, value in headers.items()}
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            return float(lowered[key]) * scale
        except (KeyError, ValueError):
            continue
    return None


def error_from_status(
    provider: Provider, status_code: int, message: str, headers: Mapping[str, str]
) -> LLMError:
    """Classify an HTTP error response the way the provider SDKs do."""
    lowered = {key.lower(): value for key, value in headers.items()}
    retry_after = parse_retry_after(lowered)
    details = {"provider": provider, "status_code": status_code, "retry_after": retry_after}

    should_retry = lowered.get("x-should-retry")
    if should_retry == "false":
        return RequestRejectedError(message, **details)
    if status_code == 429:
        return RateLimitedError(message, **details)
    if should_retry == "true" or status_code >= 500 or status_code in RETRYABLE_CLIENT_STATUSES:
        return ProviderUnavailableError(message, **details)
    return RequestRejectedError(message, **details)
