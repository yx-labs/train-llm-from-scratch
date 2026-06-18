from __future__ import annotations

from fastapi import APIRouter

from server.core.artifact_registry import list_artifacts


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("")
def artifacts() -> list[dict[str, object]]:
    return list_artifacts()
