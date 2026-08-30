from __future__ import annotations

import induction_equality_tensor_final_ood as final


def _support():
    return {name: {"powered": True} for name in ("positive", "matched_negative", "off_target", "all")}


def _effects(target=0.52, specificity=0.55, off=0.006, extraction=0.97, deranged=0.0):
    return {
        "target_damage": {"mean": target, "bootstrap_95_low": target - 0.1, "bootstrap_95_high": target + 0.1},
        "specificity": {"mean": specificity, "bootstrap_95_low": specificity - 0.1, "bootstrap_95_high": specificity + 0.1},
        "off_target_damage": {"mean": off, "bootstrap_95_low": off - 0.002, "bootstrap_95_high": off + 0.002},
        "extraction_recovery": {"mean": extraction, "bootstrap_95_low": extraction - 0.05, "bootstrap_95_high": extraction + 0.02},
        "deranged_recovery": {"mean": deranged, "bootstrap_95_low": deranged - 0.01, "bootstrap_95_high": deranged + 0.01},
    }


def test_final_gate_accepts_discovery_scale_effect():
    gates = final.role_gates("final_natural", _support(), _effects(), True)
    assert all(gates.values())


def test_ood_gate_enforces_transport_floor():
    effects = _effects(target=0.20)
    gates = final.role_gates("ood_code", _support(), effects, True)
    assert not gates["target_transport"]


def test_final_gate_enforces_collateral_upper_bound():
    effects = _effects(off=0.02)
    gates = final.role_gates("final_natural", _support(), effects, True)
    assert not gates["collateral"]


def test_role_bindings_are_exactly_final_and_code():
    assert set(final.ROLE_PATHS) == {"final_natural", "ood_code"}
    assert set(final.ROLE_SHA256S) == set(final.ROLE_PATHS)
