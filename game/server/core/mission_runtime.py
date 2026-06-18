from __future__ import annotations

from typing import Any

from server.core.content import get_mission, get_next_mission_id, load_failure_rules, load_missions
from server.core.progress_store import (
    get_progress,
    record_attempt,
    unlock_artifacts,
    unlock_concept,
    unlock_modules,
)
from server.core.simulator import incident, simulate_generation, simulate_pretraining


def list_missions_with_progress() -> list[dict[str, Any]]:
    progress = get_progress()
    completed = set(progress["completed_missions"])
    result: list[dict[str, Any]] = []
    for mission in load_missions():
        prerequisites = set(mission.get("prerequisites", []))
        unlocked = prerequisites.issubset(completed)
        item = {
            **mission,
            "status": "completed" if mission["id"] in completed else ("unlocked" if unlocked else "locked"),
        }
        result.append(item)
    return result


def _first_rule(mission: dict[str, Any]) -> str:
    return mission.get("failure_rules", ["mission_failed"])[0]


def _score(passed: bool) -> float:
    return 1.0 if passed else 0.0


def _evaluate(mission: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    success = mission["success"]
    success_type = success["type"]

    if success_type == "exact_kept_samples":
        kept = set(submission.get("kept_sample_ids", []))
        target = set(success["target_kept_sample_ids"])
        if kept == target:
            return {"passed": True}
        sample_by_id = {sample["id"]: sample for sample in mission["inputs"]["samples"]}
        if any(sample_by_id.get(sample_id, {}).get("label", "").startswith("bad") for sample_id in kept):
            return {"passed": False, "incident_report": incident("dirty_data_kept")}
        if any(sample_by_id.get(sample_id, {}).get("label") == "duplicate" for sample_id in kept):
            return {"passed": False, "incident_report": incident("duplicate_data_kept")}
        return {"passed": False, "incident_report": incident(_first_rule(mission))}

    if success_type == "token_merge_contains":
        tokens = submission.get("tokens", [])
        required = success["required_tokens"]
        passed = all(token in tokens for token in required) and len(tokens) <= int(success["max_token_count"])
        return {"passed": passed, "incident_report": None if passed else incident("tokenizer_too_fragmented")}

    if success_type == "target_offset":
        passed = int(submission.get("target_offset", 0)) == int(success["target_offset"])
        return {"passed": passed, "incident_report": None if passed else incident("target_shift_wrong")}

    if success_type == "lower_triangular_mask":
        mask = submission.get("mask", [])
        size = int(mission["inputs"]["size"])
        expected = [[col <= row for col in range(size)] for row in range(size)]
        passed = mask == expected
        return {"passed": passed, "incident_report": None if passed else incident("causal_mask_leak")}

    if success_type == "module_set":
        selected = set(submission.get("selected_modules", []))
        required = set(success["required_modules"])
        forbidden = set(success.get("forbidden_modules", []))
        if selected & forbidden:
            return {"passed": False, "incident_report": incident("bidirectional_attention_used")}
        passed = required.issubset(selected)
        return {"passed": passed, "incident_report": None if passed else incident("transformer_missing_stability")}

    if success_type == "training_simulation":
        return simulate_pretraining(submission)

    if success_type == "generation_stable":
        result = simulate_generation(submission)
        if result["passed"]:
            temperature = float(submission.get("temperature", 0.0))
            top_p = float(submission.get("top_p", 0.0))
            repetition_penalty = float(submission.get("repetition_penalty", 0.0))
            passed = (
                float(success["temperature_min"]) <= temperature <= float(success["temperature_max"])
                and float(success["top_p_min"]) <= top_p <= float(success["top_p_max"])
                and repetition_penalty >= float(success["repetition_penalty_min"])
            )
            result["passed"] = passed
            if not passed and result.get("incident_report") is None:
                result["incident_report"] = incident("sampling_too_hot")
        return result

    if success_type == "exact_loss_mask":
        expected = [bool(token["expected_loss"]) for token in mission["inputs"]["tokens"]]
        actual = [bool(item) for item in submission.get("loss_mask", [])]
        passed = actual == expected
        return {"passed": passed, "incident_report": None if passed else incident("sft_loss_mask_wrong")}

    return {"passed": False, "incident_report": {"title": f"Unknown success type: {success_type}"}}


def submit_mission(mission_id: str, submission: dict[str, Any]) -> dict[str, Any]:
    mission = get_mission(mission_id)
    result = _evaluate(mission, submission)
    passed = bool(result.get("passed"))
    record_attempt(mission_id, passed=passed, score=_score(passed))

    unlocked_modules: list[str] = []
    unlocked_artifacts: list[str] = []
    if passed:
        unlocked_modules = list(mission.get("unlocks", []))
        unlocked_artifacts = list(mission.get("unlock_artifacts", []))
        unlock_modules(unlocked_modules)
        unlock_artifacts(unlocked_artifacts)
        unlock_concept(mission.get("concept"))

    rules = load_failure_rules()
    response = {
        "mission_id": mission_id,
        "passed": passed,
        "unlocked_modules": unlocked_modules,
        "unlocked_artifacts": unlocked_artifacts,
        "next_mission_id": get_next_mission_id(mission_id) if passed else mission_id,
        "progress": get_progress(),
        "incident_report": result.get("incident_report"),
        "metrics": result.get("metrics"),
        "events": result.get("events", []),
        "preview": result.get("preview"),
        "known_failure_rules": [rules[rule_id] for rule_id in mission.get("failure_rules", []) if rule_id in rules],
    }
    return response
