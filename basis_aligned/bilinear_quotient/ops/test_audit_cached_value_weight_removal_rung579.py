import importlib.util
import math
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("audit_cached_value_weight_removal_rung579.py")
SPEC = importlib.util.spec_from_file_location("r579_audit", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def cell(damage: float, ce: float, rms: float = 2.0, norm: float = 3.0, answer: bool = True):
    return {"margin_damage": damage, "ce_increase": ce,
            "full_vocabulary_logit_rms": rms,
            "compiled_residual_term_norm": norm,
            "answer_remains_best": answer}


def test_fail_closed_ratio():
    assert AUDIT.ratio(2.0, 4.0) == .5
    assert math.isinf(AUDIT.ratio(0.0, 0.0))


def test_target_audit_independently_matches_saved_values(monkeypatch):
    monkeypatch.setattr(AUDIT, "BOOTSTRAPS", 25)
    raw = {family: {endpoint: [cell(2.0, 1.0), cell(4.0, 2.0)]
                    for endpoint in AUDIT.ENDPOINTS} for family in ("example",)}
    saved = {"example": {}}
    seed = 7
    for endpoint in AUDIT.ENDPOINTS:
        cells = raw["example"][endpoint]
        damage = [item["margin_damage"] for item in cells]
        ce = [item["ce_increase"] for item in cells]
        saved["example"][endpoint] = {
            "n": 2,
            "mean_margin_damage": float(np.mean(damage)),
            "median_margin_damage": float(np.median(damage)),
            "positive_margin_damage_fraction": 1.0,
            "bootstrap95_lower_mean_margin_damage": AUDIT.lower(damage, seed),
            "mean_ce_increase": float(np.mean(ce)),
            "bootstrap95_lower_mean_ce_increase": AUDIT.lower(ce, seed + 1),
            "median_logit_rms": 2.0,
            "median_term_norm": 3.0,
            "passed": True,
        }
        seed += 2
    checks, passed, next_seed = AUDIT.audit_target_report(raw, saved, ("example",), 7)
    assert all(checks)
    assert passed
    assert next_seed == 11
