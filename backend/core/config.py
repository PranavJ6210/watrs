"""
core/config.py
──────────────
Application configuration loaded from environment variables via pydantic-settings.
The .env file in the backend root is parsed automatically.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── Security ────────────────────────────────────────────────────────
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Database ────────────────────────────────────────────────────────
    MONGODB_URL: str

    # ── External APIs ───────────────────────────────────────────────────
    WEATHERAPI_API_KEY: str = ""
    GOOGLE_PLACES_API_KEY: str

    # ── Cloud Storage ───────────────────────────────────────────────────
    CLOUDINARY_URL: str = ""

    # ── Rate Limiting ───────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── CORS ────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse the comma-separated CORS_ORIGINS string into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    return Settings()
