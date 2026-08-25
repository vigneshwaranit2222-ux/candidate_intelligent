"""
Core Application Configuration Module.

Analogy for Beginners:
Think of this file like the master control dashboard of a space station.
Before any rockets launch or doors open, the station reads its environment variables
(like fuel levels or oxygen settings) from a hidden configuration file (.env).
Pydantic v2 validates all these settings so our application never runs with missing configuration!
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """
    Application Settings using Pydantic v2 Settings Management.
    Automatically loads environment variables from a `.env` file if present,
    or falls back to production-grade defaults.
    """

    # Application Information
    APP_NAME: str = "AI Candidate Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database Configuration
    # Default points to local SQLite async driver for zero-dependency standalone local dev.
    # Docker Compose or .env automatically overrides this with PostgreSQL asyncpg URL.
    DATABASE_URL: str = "sqlite+aiosqlite:///./candidate_app.db"

    # AI & LLM Settings (Google Gemini)
    GEMINI_API_KEY: str = "mock-gemini-key-for-testing"
    LLM_MODEL: str = "gemini-1.5-flash"
    
    # Toggle mock mode for zero-dependency local testing without live API keys
    USE_MOCK_LLM: bool = True

    # Embedding & RAG Vector Settings
    # Vector dimension 1536 matches standard text-embedding-3-small or text-embedding-ada-002
    VECTOR_DIMENSION: int = 1536
    READINESS_TIERS: list[dict] = [
        {"label": "Foundation", "minimum": 0}, {"label": "Emerging", "minimum": 45},
        {"label": "Developing", "minimum": 60}, {"label": "Intermediate Potential", "minimum": 70},
        {"label": "Job Ready", "minimum": 80}, {"label": "Advanced Ready", "minimum": 90},
    ]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def normalize_debug_mode(cls, value):
        if isinstance(value, str) and value.lower() in {"release", "production"}:
            return False
        return value

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Instantiate a global singleton settings instance accessible across the app
settings = Settings()
