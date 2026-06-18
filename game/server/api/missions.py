from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.core.content import get_mission
from server.core.mission_runtime import list_missions_with_progress, submit_mission


router = APIRouter(prefix="/api/missions", tags=["missions"])


class MissionSubmitRequest(BaseModel):
    submission: dict[str, Any]


@router.get("")
def list_missions() -> list[dict[str, Any]]:
    return list_missions_with_progress()


@router.get("/{mission_id}")
def mission_detail(mission_id: str) -> dict[str, Any]:
    try:
        return get_mission(mission_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Mission not found: {mission_id}") from exc


@router.post("/{mission_id}/submit")
def submit(mission_id: str, request: MissionSubmitRequest) -> dict[str, Any]:
    try:
        return submit_mission(mission_id, request.submission)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Mission not found: {mission_id}") from exc
