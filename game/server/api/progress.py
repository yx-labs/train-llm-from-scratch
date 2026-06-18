from __future__ import annotations

from fastapi import APIRouter

from server.core.progress_store import get_progress, reset_progress


router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("")
def current_progress() -> dict[str, object]:
    return get_progress()


@router.post("/reset")
def reset() -> dict[str, object]:
    reset_progress()
    return get_progress()
