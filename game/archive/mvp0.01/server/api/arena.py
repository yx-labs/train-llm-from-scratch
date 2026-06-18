from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from server.core.artifact_registry import compare_outputs


router = APIRouter(prefix="/api/arena", tags=["arena"])


class CompareRequest(BaseModel):
    prompt: str
    artifact_ids: list[str]
    temperature: float = 0.7
    top_p: float = 0.9


@router.post("/compare")
def compare(request: CompareRequest) -> dict[str, object]:
    return {
        **compare_outputs(request.prompt, request.artifact_ids),
        "generation": {
            "temperature": request.temperature,
            "top_p": request.top_p,
        },
    }
