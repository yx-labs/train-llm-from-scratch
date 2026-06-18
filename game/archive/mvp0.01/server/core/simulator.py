from __future__ import annotations

from typing import Any

from server.core.content import load_failure_rules


def _curve(template: str) -> list[dict[str, float]]:
    base_steps = [0, 100, 200, 300, 400, 500]
    templates = {
        "healthy_decay": ([4.8, 4.1, 3.5, 3.0, 2.7, 2.45], [4.9, 4.25, 3.65, 3.18, 2.88, 2.62]),
        "loss_explosion": ([4.8, 3.8, 3.2, 6.4, 10.5, 14.0], [4.9, 4.0, 3.6, 7.1, 11.8, 15.2]),
        "noisy_decay": ([4.8, 4.4, 3.7, 4.1, 3.2, 3.5], [4.9, 4.6, 4.0, 4.4, 3.7, 3.9]),
        "flatline": ([4.8, 4.78, 4.75, 4.73, 4.72, 4.7], [4.9, 4.9, 4.88, 4.87, 4.87, 4.86]),
        "suspicious_fast_drop": ([4.8, 2.2, 0.8, 0.35, 0.18, 0.12], [4.9, 2.1, 0.75, 0.32, 0.16, 0.1]),
    }
    train, val = templates.get(template, templates["healthy_decay"])
    return [
        {
            "step": step,
            "train_loss": train[index],
            "val_loss": val[index],
            "gpu_memory_mb": 900 + index * 45,
        }
        for index, step in enumerate(base_steps)
    ]


def incident(rule_id: str) -> dict[str, Any]:
    rule = load_failure_rules().get(rule_id, {})
    return {
        "id": rule_id,
        "title": rule.get("title", "训练异常"),
        "severity": rule.get("severity", "warning"),
        "symptoms": rule.get("symptoms", []),
        "suspected_modules": rule.get("suspected_modules", []),
        "hints": rule.get("hints", []),
    }


def simulate_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    lr = float(config.get("learning_rate", 0.0003))
    batch_size = int(config.get("batch_size", 16))
    data_quality = float(config.get("data_quality", 0.8))
    causal_mask = bool(config.get("causal_mask", True))

    if not causal_mask:
        return {
            "passed": False,
            "template": "suspicious_fast_drop",
            "metrics": _curve("suspicious_fast_drop"),
            "events": [{"step": 200, "type": "future_leak", "severity": "error"}],
            "incident_report": incident("causal_mask_leak"),
        }
    if lr > 0.001:
        return {
            "passed": False,
            "template": "loss_explosion",
            "metrics": _curve("loss_explosion"),
            "events": [{"step": 300, "type": "loss_explosion", "severity": "error"}],
            "incident_report": incident("learning_rate_too_high"),
        }
    if lr < 0.0002:
        report = incident("learning_rate_too_high")
        report["id"] = "learning_rate_too_low"
        report["title"] = "学习率过低导致预算内学不动"
        report["symptoms"] = ["loss 几乎不下降", "训练预算被消耗但模型能力变化很小"]
        report["hints"] = ["略微提高学习率。", "MVP 中先尝试 0.0003。"]
        return {
            "passed": False,
            "template": "flatline",
            "metrics": _curve("flatline"),
            "events": [{"step": 400, "type": "slow_learning", "severity": "warning"}],
            "incident_report": report,
        }
    if data_quality < 0.7:
        return {
            "passed": False,
            "template": "flatline",
            "metrics": _curve("flatline"),
            "events": [{"step": 250, "type": "data_quality_low", "severity": "error"}],
            "incident_report": incident("data_quality_too_low"),
        }
    if batch_size < 16:
        return {
            "passed": False,
            "template": "noisy_decay",
            "metrics": _curve("noisy_decay"),
            "events": [{"step": 200, "type": "noisy_batches", "severity": "warning"}],
            "incident_report": incident("batch_too_small"),
        }

    return {
        "passed": True,
        "template": "healthy_decay",
        "metrics": _curve("healthy_decay"),
        "events": [{"step": 500, "type": "checkpoint_unlocked", "severity": "info"}],
        "incident_report": None,
    }


def simulate_generation(config: dict[str, Any]) -> dict[str, Any]:
    temperature = float(config.get("temperature", 0.7))
    top_p = float(config.get("top_p", 0.9))
    repetition_penalty = float(config.get("repetition_penalty", 1.05))

    if temperature > 1.0:
        return {
            "passed": False,
            "preview": "The model learns orbital blue syntax because tokens bloom unrelated ideas...",
            "incident_report": incident("sampling_too_hot"),
        }
    if top_p < 0.75 or temperature < 0.45:
        return {
            "passed": False,
            "preview": "The model learns learns learns the model learns",
            "incident_report": incident("sampling_too_narrow"),
        }
    if repetition_penalty < 1.05:
        return {
            "passed": False,
            "preview": "The model learns tokens tokens tokens tokens",
            "incident_report": incident("repetition_loop"),
        }
    return {
        "passed": True,
        "preview": "The model learns statistical structure by predicting the next token from prior context.",
        "incident_report": None,
    }
