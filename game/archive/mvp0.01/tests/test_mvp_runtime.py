from __future__ import annotations

import sys
from pathlib import Path


GAME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GAME_ROOT))

from server.core.artifact_registry import compare_outputs, list_artifacts  # noqa: E402
from server.core.mission_runtime import list_missions_with_progress, submit_mission  # noqa: E402
from server.core.progress_store import reset_progress  # noqa: E402


def test_mvp_mission_chain_loads() -> None:
    reset_progress()
    missions = list_missions_with_progress()

    assert len(missions) == 8
    assert missions[0]["id"] == "01_clean_text"
    assert missions[-1]["id"] == "08_sft_mask"


def test_causal_mask_rejects_future_leak_and_accepts_lower_triangle() -> None:
    reset_progress()

    bad = submit_mission("04_causal_mask", {"mask": [[True] * 4 for _ in range(4)]})
    assert bad["passed"] is False
    assert bad["incident_report"]["id"] == "causal_mask_leak"

    good = submit_mission(
        "04_causal_mask",
        {
            "mask": [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        },
    )
    assert good["passed"] is True
    assert "attention.causal_mask" in good["unlocked_modules"]


def test_pretrain_sim_unlocks_base_artifact() -> None:
    reset_progress()

    result = submit_mission(
        "06_pretrain_sim",
        {
            "learning_rate": 0.0003,
            "batch_size": 16,
            "context_length": 256,
            "data_quality": 0.8,
            "causal_mask": True,
        },
    )

    assert result["passed"] is True
    assert result["metrics"]
    assert "pretrain_good" in result["unlocked_artifacts"]


def test_arena_uses_sample_output_fallback() -> None:
    reset_progress()

    artifacts = list_artifacts()
    assert {artifact["id"] for artifact in artifacts} == {"pretrain_good", "sft"}

    result = compare_outputs(
        "请用 <think> 和 <answer> 回答：23 + 48 = ?",
        ["pretrain_good", "sft"],
    )

    assert len(result["results"]) == 2
    assert result["results"][0]["source"] == "sample_outputs"
    assert "<answer>71</answer>" in result["results"][1]["output"]
