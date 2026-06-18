from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


GAME_ROOT = Path(__file__).resolve().parents[1]
ENV_LOCAL = GAME_ROOT / ".env.local"


@dataclass(frozen=True)
class AIConfig:
    provider: str
    base_url: str | None
    api_key_present: bool
    llm_model: str | None
    image_model: str | None


def load_ai_config(env_path: Path = ENV_LOCAL) -> AIConfig:
    values = dotenv_values(env_path) if env_path.exists() else {}

    openai_key = values.get("OPENAI_API_KEY")
    return AIConfig(
        provider="openai",
        base_url=values.get("OPENAI_BASE_URL"),
        api_key_present=bool(openai_key),
        llm_model=values.get("OPENAI_LLM_MODEL"),
        image_model=values.get("OPENAI_IMAGE_MODEL") or "gpt-image-2",
    )
