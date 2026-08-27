from __future__ import annotations

import ast
import json

import pytest
import torch

import early_mlp_state_complete_compiler_v21_site0 as site0


def test_metrics_recompute_only_from_raw_aggregates(monkeypatch) -> None:
    state = {"grammar": "dummy"}
    price = {"total_reals": 7}
    monkeypatch.setattr(site0.authority.selection, "state_price", lambda _: price)
    context = {"teacher_denominator": 2.0, "copy_baseline": 3.0}
    raw = {
        "candidate_teacher_kl_sum": 8.0,
        "candidate_teacher_kl_count": 8,
        "global_ce_sum": 32.0,
        "global_ce_count": 8,
        "copy_ce_sum": 13.0,
        "copy_ce_count": 4,
    }
    metrics = site0.metrics_from_sufficient_statistics(state, context, raw)
    assert metrics["candidate_teacher_kl"] == 1.0
    assert metrics["recovery"] == 0.5
    assert metrics["global_ce"] == 4.0
    assert metrics["copy_worsening"] == 0.25
    assert metrics["price"] == price
    assert metrics["raw_sufficient_statistics"] == raw


def test_metrics_reject_misaligned_counts(monkeypatch) -> None:
    monkeypatch.setattr(site0.authority.selection, "state_price", lambda _: {})
    raw = {
        "candidate_teacher_kl_sum": 1.0,
        "candidate_teacher_kl_count": 2,
        "global_ce_sum": 1.0,
        "global_ce_count": 3,
        "copy_ce_sum": 1.0,
        "copy_ce_count": 1,
    }
    with pytest.raises(RuntimeError, match="counts are invalid"):
        site0.metrics_from_sufficient_statistics(
            {}, {"teacher_denominator": 1.0, "copy_baseline": 1.0}, raw,
        )


def test_capture_hash_binds_keys_shapes_dtypes_and_values() -> None:
    first = {"z": torch.zeros(2, 3), "p": torch.ones(2, 1)}
    reordered = {"p": first["p"], "z": first["z"]}
    changed = {"z": torch.ones(2, 3), "p": torch.ones(2, 1)}
    assert site0._capture_sha256(first) == site0._capture_sha256(reordered)
    assert site0._capture_sha256(first) != site0._capture_sha256(changed)


def test_full_native_control_binds_measurement() -> None:
    state = {"family": "full-native", "weight": torch.ones(1)}
    gate = {
        "passed": True,
        "physical_tolerance": 8e-6,
        "physical_max_abs_error": 1e-7,
        "poison_calls": {0: 0, 1: 24, 2: 0},
        "row_ce_max_abs_drift": 1e-7,
    }
    control = site0.full_native_control(
        state, gate, {0: 24, 1: 0, 2: 0}, validation_identity="v" * 64,
        physical_reference_scale=2.0,
    )
    measurement = {
        "context": control["context"],
        "upstream_state_sha256": control["upstream_state_sha256"],
        "validation_document_ids_sha256": control[
            "validation_document_ids_sha256"
        ],
        "scorer": control["scorer"],
        "state_sha256": site0.authority.state_logical_sha256(state),
        "integrity_gates": control["integrity_gates"],
        "observed": control["observed"],
    }
    assert control["measurement_sha256"] == site0.authority.logical_json_sha256(
        measurement
    )


def test_full_native_control_rejects_failed_gate() -> None:
    gate = {
        "passed": False,
        "physical_tolerance": 4e-6,
        "physical_max_abs_error": 0.0,
        "poison_calls": {0: 0, 1: 24, 2: 0},
        "row_ce_max_abs_drift": 0.0,
    }
    with pytest.raises(RuntimeError, match="did not pass"):
        site0.full_native_control(
            {"weight": torch.ones(1)}, gate, {0: 24, 1: 0, 2: 0},
            validation_identity="v" * 64, physical_reference_scale=1.0,
        )


def test_both_complete_banks_freeze_before_any_selector(monkeypatch) -> None:
    events = []
    ledgers = {
        name: {f"c{i}": {} for i in range(108)}
        for name in ("true_site0", "shuffle_site0")
    }

    def freeze(stage, frozen_ledgers, controls, diagnostics, *, launch_state):
        assert stage == "site0"
        assert {name: len(bank) for name, bank in frozen_ledgers.items()} == {
            "true_site0": 108, "shuffle_site0": 108,
        }
        events.append("freeze")

    def select(stage):
        assert stage == "site0"
        events.append("select")
        return {"selected": {}}

    monkeypatch.setattr(site0.lifecycle, "freeze_preselector_stage", freeze)
    monkeypatch.setattr(site0.lifecycle, "select_frozen_stage", select)
    result = site0.freeze_and_select_site0(
        ledgers, {}, {}, launch_state=object(),
    )
    assert result == {"selected": {}}
    assert events == ["freeze", "select"]


def test_teacher_and_candidate_use_registered_oon_non_qon_routing(monkeypatch) -> None:
    rows = torch.zeros(1, 257, dtype=torch.long)

    class Hook:
        def __init__(self):
            self.configs = []
            self.programs = {}

        def configure(self, states, **kwargs):
            self.configs.append((dict(states), dict(kwargs)))
            self.states = dict(states)

    class Guard:
        def __init__(self, blocks, allowed):
            self.counts = {site: (1 if site in allowed else 0) for site in (0, 1, 2)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def assert_contract(self, **kwargs):
            return None

    hook = Hook()

    class SA:
        DEV = torch.device("cpu")
        H = [object(), object(), object()]

        @staticmethod
        def fwd_arm(idx, all_attention, twall, mlps):
            logits = torch.zeros(len(idx), 256, 3)
            if hook.states == {1: "O"}:  # NON differs from OON.
                logits[..., 0] = 1.0
            elif hook.states == {0: "Q", 1: "O"}:  # QON candidate.
                logits[..., 1] = 0.5
            return logits

    monkeypatch.setattr(
        site0.old_site0.runtime, "OriginalMLPCallGuard", Guard,
    )
    monkeypatch.setattr(site0.authority, "VALIDATION_TOKEN_COUNT", 192)
    monkeypatch.setattr(
        site0.old_site0, "_copy_mask",
        lambda idx, targets: torch.ones_like(targets, dtype=torch.bool),
    )
    teacher, context, _ = site0.teacher_bank(SA, hook, rows, {}, frozenset())
    assert [config for config, _ in hook.configs] == [
        {0: "O", 1: "O"}, {1: "O"},
    ]
    assert teacher.shape == (1, 192, 3)
    assert context["teacher_token_count"] == 192
    assert context["copy_token_count"] == 192

    state = {"grammar": "dummy"}
    monkeypatch.setattr(site0.old_site0, "_device_state", lambda value, device: value)
    monkeypatch.setattr(
        site0.authority.selection, "state_price", lambda _: {"total_reals": 1},
    )
    metrics, _ = site0.score_candidate(
        SA, hook, rows, {}, frozenset(), "candidate", state, teacher, context,
    )
    assert hook.configs[-1] == (
        {0: "Q", 1: "O"}, {"program_name": "candidate"},
    )
    assert metrics["raw_sufficient_statistics"]["candidate_teacher_kl_count"] == 192
    assert metrics["raw_sufficient_statistics"]["copy_ce_count"] == 192


def test_main_refuses_existing_manifest(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "historical"}))
    monkeypatch.setattr(site0, "MANIFEST", manifest)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        site0.main()


def test_stage_source_never_requests_or_names_final_role() -> None:
    source = site0.Path(site0.__file__).read_text()
    assert "load_final_for_scoring" not in source
    tree = ast.parse(source)
    role_loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_roles_and_validate"
    ]
    assert len(role_loads) == 1
    requested = {
        value.value for value in ast.walk(role_loads[0])
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    assert requested == {"compiler_fit_v21", "compiler_validation_v21"}


def test_program_source_closure_contains_site0_runner_and_test() -> None:
    names = {path.name for path in site0.authority.PROGRAM_SOURCE_CLOSURE}
    assert "early_mlp_state_complete_compiler_v21_site0.py" in names
    assert "test_early_mlp_state_complete_compiler_v21_site0.py" in names
