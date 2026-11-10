"""Central configuration, loaded from environment variables or a local .env file."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    def require(self, name: str) -> str:
        """Return a secret's value, failing loudly if it is not configured."""
        value: SecretStr | None = getattr(self, name)
        if value is None:
            raise RuntimeError(f"{name.upper()} is not set. Add it to .env (see .env.example).")
        return value.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
