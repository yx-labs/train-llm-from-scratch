from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from server.config import GAME_ROOT


MISSIONS_PATH = GAME_ROOT / "missions" / "mvp.json"
ARTIFACTS_PATH = GAME_ROOT / "artifacts" / "manifest.json"
SAMPLE_OUTPUTS_PATH = GAME_ROOT / "artifacts" / "samples" / "model_outputs.json"
FAILURE_RULES_PATH = GAME_ROOT / "simulator" / "rules" / "mvp_failures.json"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_missions() -> list[dict[str, Any]]:
    missions = _read_json(MISSIONS_PATH)
    return sorted(missions, key=lambda item: item["order"])


@lru_cache(maxsize=1)
def load_artifact_manifest() -> dict[str, Any]:
    return _read_json(ARTIFACTS_PATH)


@lru_cache(maxsize=1)
def load_failure_rules() -> dict[str, dict[str, Any]]:
    rules = _read_json(FAILURE_RULES_PATH)
    return {rule["id"]: rule for rule in rules}


@lru_cache(maxsize=1)
def load_sample_outputs() -> list[dict[str, Any]]:
    if not SAMPLE_OUTPUTS_PATH.exists():
        return []
    with SAMPLE_OUTPUTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_mission(mission_id: str) -> dict[str, Any]:
    for mission in load_missions():
        if mission["id"] == mission_id:
            return mission
    raise KeyError(mission_id)


def get_next_mission_id(mission_id: str) -> str | None:
    missions = load_missions()
    for index, mission in enumerate(missions):
        if mission["id"] == mission_id:
            if index + 1 < len(missions):
                return missions[index + 1]["id"]
            return None
    return None
