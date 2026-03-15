from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class CognitiveSenseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cognitivesense_server_port: int | None = None
    state_tracker_type: str = "rule"
    openai_api_key: str | None = None
    cognitivesense_log_level: str = "DEBUG"
    speech_tone_backend: str = "transformer" # heuristic or transformer

    # Performance
    target_fps: int = 15
    llm_state_tracker_min_interval: float = 5.0
    llm_state_tracker_cooldown: float = 10.0
    llm_state_tracker_model: str = "gpt-4.1-mini"
    llm_feedback_model: str = "gpt-4.1-nano"
    llm_feedback_cooldown: float = 30.0


_settings: CognitiveSenseSettings | None = None


def get_settings() -> CognitiveSenseSettings:
    global _settings
    if _settings is None:
        _settings = CognitiveSenseSettings()
    return _settings
