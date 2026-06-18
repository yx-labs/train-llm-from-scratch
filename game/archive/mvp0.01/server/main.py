from __future__ import annotations

import platform
import sys

from fastapi import FastAPI

from server.api import arena, artifacts, missions, progress
from server.config import ENV_LOCAL, load_ai_config

app = FastAPI(title="LLM Learning Game API")

app.include_router(missions.router)
app.include_router(progress.router)
app.include_router(artifacts.router)
app.include_router(arena.router)


@app.get("/api/health")
def health() -> dict[str, object]:
    ai = load_ai_config()
    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env_local_present": ENV_LOCAL.exists(),
        "ai": {
            "provider": ai.provider,
            "base_url_present": bool(ai.base_url),
            "api_key_present": ai.api_key_present,
            "llm_model": ai.llm_model,
            "image_model": ai.image_model,
        },
    }
