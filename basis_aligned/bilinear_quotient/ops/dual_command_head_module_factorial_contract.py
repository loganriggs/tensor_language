#!/usr/bin/env python3
"""Pure accounting for paired dual-command head/module interventions."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def paired_cell(cell, role):
    if len(cell) != 2 or any(bit not in "01" for bit in cell):
        raise ValueError("cell must be a binary pair")
    position = {"temporal": 0, "iswas": 1}.get(role)
    if position is None: raise ValueError("unknown command role")
    bits = list(cell); bits[position] = "1" if bits[position] == "0" else "0"
    return "".join(bits)


def effect_metrics(target_effect, target_gold, non_target_effect):
    effect = np.asarray(target_effect, dtype=np.float64)
    gold = np.asarray(target_gold, dtype=np.float64)
    non_target = np.asarray(non_target_effect, dtype=np.float64)
    if effect.shape != gold.shape or non_target.shape != gold.shape or effect.ndim != 1:
        raise ValueError("effects must be same-shaped vectors")
    gold_norm = float(np.linalg.norm(gold))
    effect_norm = float(np.linalg.norm(effect))
    if gold_norm <= 0: raise ValueError("gold intervention effect is zero")
    signed = float(effect @ gold / (gold @ gold))
    cosine = float(effect @ gold / (effect_norm * gold_norm)) if effect_norm else 0.0
    return {"count": int(effect.size), "signed_recovery": signed, "cosine": cosine,
            "direction_agreement": float(np.mean(np.sign(effect) == np.sign(gold))),
            "relative_residual": float(np.linalg.norm(effect - gold) / gold_norm),
            "non_target_to_target_gold_norm": float(np.linalg.norm(non_target) / gold_norm)}


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["role"], row["phase"], row["template_id"], row["arm"])].append(row)
        grouped[(row["role"], row["phase"], "ALL", row["arm"])].append(row)
    reports = []
    for (role, phase, template, arm), rows in sorted(grouped.items()):
        metrics = effect_metrics([row["target_effect"] for row in rows],
                                 [row["target_gold"] for row in rows],
                                 [row["non_target_effect"] for row in rows])
        reports.append({"role": role, "phase": phase, "template_id": template,
                        "arm": arm, **metrics})
    return reports


def union_additivity(union_effect, singleton_effects):
    union = np.asarray(union_effect, dtype=np.float64)
    singletons = [np.asarray(value, dtype=np.float64) for value in singleton_effects]
    if not singletons or any(value.shape != union.shape for value in singletons):
        raise ValueError("singleton effects must match a nonempty union")
    denominator = float(np.linalg.norm(union))
    if denominator <= 0: return math.inf
    return float(np.linalg.norm(union - sum(singletons)) / denominator)
