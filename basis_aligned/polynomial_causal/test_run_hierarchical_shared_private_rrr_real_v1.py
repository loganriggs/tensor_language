from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import hierarchical_shared_private_rrr as hybrid
import run_hierarchical_shared_private_rrr_real_v1 as run


def _synthetic_state(n_sites=3, dimension=4, seed=7):
    generator = torch.Generator().manual_seed(seed)
    solved = tuple(torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64,
    ) for _ in range(n_sites))
    merits = tuple(value.T @ value for value in solved)
    independent = tuple(run.base._descending_eigh(value) for value in merits)
    global_values, global_vectors = run.base._descending_eigh(sum(
        merits[1:], merits[0].clone(),
    ))
    state = run.base.SpectralState(
        gram=torch.eye(dimension, dtype=torch.float64),
        crosses=solved, y2=tuple(100.0 for _ in range(n_sites)), solved=solved,
        independent_values=tuple(value for value, _ in independent),
        independent_vectors=tuple(value for _, value in independent),
        global_values=global_values, global_vectors=global_vectors,
        typed_values={}, typed_vectors={}, legacy_svd=None,
    )
    state.hierarchical_merits = merits
    state.hierarchical_residual_cache = None
    return state


def test_seven_arm_grid_prices_and_call_schedule_are_frozen():
    descriptors = run.arm_descriptors()
    assert len(descriptors) == 7
    assert [(value["budget_name"], value["shared_rank"]) for value in descriptors] == [
        ("global_q512", 0), ("typed_q512", 0), ("independent_q512", 0),
        ("global_q512", 128), ("typed_q512", 128), ("independent_q512", 128),
        ("global_q512", 512),
    ]
    assert run.expected_call_schedule() == {
        "fit_native_outer": 22, "native_reference_outer": 120,
        "compiled_outer_per_arm": 120, "compiled_arm_count": 7,
        "compiled_outer_total": 840, "outer_total": 982,
        "native_component_calls_per_kind": 2556,
        "compiled_component_calls": 30240,
        "optimizer_calls": 0, "backward_calls": 0,
    }


def test_parent_receipt_and_source_input_closure_are_exact():
    parent = run.verify_parent()
    assert parent["results_file_sha256"] == (
        "19d65e2c6d4a0cff19ddfb76ddbe62dcd26c462a695e006c457da85a89adc053"
    )
    assert run.PARENT_HASHES.items() <= run.FILE_PINS.items()
    for path in (run.RUNNER, run.TEST, run.ADDENDUM, run.MATH_PREREG,
                 run.MATH_CORE, run.MATH_TEST):
        assert str(path.relative_to(run.ROOT)) in run.SOURCE_PATHS


def test_pinned_parent_read_rejects_between_hash_drift(tmp_path, monkeypatch):
    path = tmp_path / "parent.json"
    path.write_text('{"ok": true}')
    expected = run.base.file_sha256(path)
    observed = iter((expected, "0" * 64))
    monkeypatch.setattr(run.base, "file_sha256", lambda _path: next(observed))
    with pytest.raises(RuntimeError, match="changed while reading"):
        run._read_pinned_json(path, expected)


def test_configuration_is_isolated_and_restorable():
    original = {name: getattr(run.base, name) for name in run._BASE_DEFAULTS}
    try:
        run.configure_base()
        assert run.base.AUTHORITY == run.AUTHORITY
        assert run.base.arm_descriptors is run.arm_descriptors
        assert run.base.fit_program is run.fit_program
        assert run.base.AutonomousProgram is run.AutonomousProgram
        assert run.base.SOURCE_PATHS == run.SOURCE_PATHS
    finally:
        run.restore_base_defaults()
    assert all(getattr(run.base, name) is value if callable(value) else (
        getattr(run.base, name) == value
    ) for name, value in original.items())


def test_exact_seven_arm_call_ledger_replays():
    try:
        run.configure_base()
        ledger = run.base.PhysicalCallLedger()
        for phase, count in (("fit", run.base.FIT_OUTER_CALLS),
                             ("native_reference", run.base.EVAL_CALLS_PER_ARM)):
            for _ in range(count):
                ledger.record_native_outer(phase)
                for kind in ("attn", "mlp"):
                    for site in range(18):
                        ledger.record_native_site(phase, kind, site)
        for descriptor in run.arm_descriptors():
            for _ in range(run.base.EVAL_CALLS_PER_ARM):
                ledger.record_compiled_outer(descriptor["name"])
                for kind in ("attn", "mlp"):
                    for site in range(18):
                        ledger.record_compiled_site(descriptor["name"], kind, site)
        receipt = ledger.receipt()
        run.base.semantic_validate_call_ledger(receipt)
        assert receipt["registered"] == run.expected_call_schedule()
    finally:
        run.restore_base_defaults()


def test_run_restores_base_even_when_preflight_refuses(monkeypatch):
    original = run.base.AUTHORITY
    monkeypatch.setattr(run, "verify_parent", lambda: (_ for _ in ()).throw(
        RuntimeError("prospective refusal")
    ))
    with pytest.raises(RuntimeError, match="prospective refusal"):
        run.run(device="cpu")
    assert run.base.AUTHORITY == original


def test_q0_zero_is_literal_independent_and_full_shared_is_literal_global(monkeypatch):
    monkeypatch.setattr(run, "D", 4)
    monkeypatch.setattr(run, "N_SITES", 3)
    monkeypatch.setattr(run, "COMMON_TABLE_FLOATS", 120)
    state = _synthetic_state()
    q0 = run.fit_program({
        "name": "q0", "family": "hierarchical_shared_private",
        "budget_name": "small", "map_float_budget": 8 * 4, "shared_rank": 0,
    }, state)
    assert q0.diagnostics["endpoint_controls"] == {
        "q0_zero_exact_price_independent": True,
        "zero_private_exact_global": None,
    }
    assert sum(q0.deployed.private_ranks) == 4
    full = run.fit_program({
        "name": "full", "family": "hierarchical_shared_private",
        "budget_name": "small", "map_float_budget": 4 * 4 * 4, "shared_rank": 4,
    }, state)
    assert full.diagnostics["endpoint_controls"] == {
        "q0_zero_exact_price_independent": None,
        "zero_private_exact_global": True,
    }
    assert sum(full.deployed.private_ranks) == 0


def test_interior_fit_replays_allocation_price_hashes_and_gaps(monkeypatch):
    monkeypatch.setattr(run, "D", 4)
    monkeypatch.setattr(run, "N_SITES", 3)
    monkeypatch.setattr(run, "COMMON_TABLE_FLOATS", 120)
    state = _synthetic_state()
    descriptor = {
        "name": "middle", "family": "hierarchical_shared_private",
        "budget_name": "small", "map_float_budget": 4 * 4 * 1 + 8 * 3,
        "shared_rank": 1,
    }
    program = run.fit_program(descriptor, state)
    run.semantic_validate_diagnostics(program.diagnostics, descriptor)
    run.semantic_validate_diagnostics(
        json.loads(json.dumps(program.diagnostics)), descriptor,
    )
    assert program.deployed.shared_basis.dtype == torch.float32
    assert program.diagnostics["deployed_hash_receipt"]["serialized_program_authority"] is False
    corrupt = deepcopy(program.diagnostics)
    corrupt["private_ranks_by_site"][0] += 1
    with pytest.raises(RuntimeError, match="allocation|price"):
        run.semantic_validate_diagnostics(corrupt, descriptor)
    assert program.diagnostics["ranks_by_site"] == [
        descriptor["shared_rank"] + rank
        for rank in program.diagnostics["private_ranks_by_site"]
    ]
    corrupt = deepcopy(program.diagnostics)
    corrupt["deployed_hash_receipt"]["coefficient_map_sha256s"][0] = "0" * 64
    with pytest.raises(RuntimeError, match="does not replay"):
        run.semantic_validate_diagnostics(corrupt, descriptor)
    corrupt = deepcopy(program.diagnostics)
    corrupt["global_eigenvalues"][0] += 1.0
    with pytest.raises(RuntimeError, match="merit replay"):
        run.semantic_validate_diagnostics(corrupt, descriptor)


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10)

    def forward(self, tokens):
        return self.weight[tokens]


def test_autonomous_adapter_uses_shared_then_private_and_covered_table(monkeypatch):
    monkeypatch.setattr(run, "D", 4)
    monkeypatch.setattr(run, "N_SITES", 1)
    monkeypatch.setattr(run.base, "SITE_TO_INDEX", {("mlp", 0): 0, ("attn", 0): 0})
    deployed = hybrid.DeployedHierarchicalProgram(
        shared_basis=torch.eye(4, dtype=torch.float32)[:, :1],
        shared_input_maps=(torch.eye(4, dtype=torch.float32)[:, :1],),
        private_bases=(torch.eye(4, dtype=torch.float32)[:, 1:2],),
        private_input_maps=(torch.eye(4, dtype=torch.float32)[:, 1:2],),
        private_ranks=(1,),
    )
    program = run.HierarchicalProgram("arm", {"name": "arm"}, deployed, {})
    ledger = SimpleNamespace(record_compiled_site=lambda *args: None)
    model = SimpleNamespace(transformer=SimpleNamespace(wte=_Embedding()))
    table = torch.tensor([[[9., 8., 7., 6.]]])
    adapter = run.AutonomousProgram(
        model, table, torch.tensor([-1, 0, -1, -1, -1]), program, ledger, "cpu",
    )
    tokens = torch.tensor([[1, 2]])
    expected = model.transformer.wte(tokens).clone()
    expected[..., 2:] = 0
    expected[0, 0] = table[0, 0]
    assert torch.equal(adapter._write(0, tokens), expected)


def _metric(ce):
    return {
        "all": {"ce": ce}, "covered": {"ce": 0.5}, "uncovered": {"ce": ce},
    }


def test_registered_gates_use_rolewise_covered_control_and_literal_endpoints():
    arms = {}
    for descriptor in run.arm_descriptors():
        budget, q0 = descriptor["budget_name"], descriptor["shared_rank"]
        ce = 1.0 if q0 == 128 else 1.2
        controls = {
            "q0_zero_exact_price_independent": True if q0 == 0 else None,
            "zero_private_exact_global": True if q0 == 512 else None,
        }
        arms[descriptor["name"]] = {
            "roles": {role: _metric(ce) for role in run.base.ROLE_PATHS},
            "diagnostics": {"endpoint_controls": controls},
        }
    # Parent replay is intentionally a known-answer control; use its exact CE for
    # endpoints rather than weakening the production 0.002 threshold.
    parent = run._parent_arms()
    for name, parent_name in {
        run.arm_name("global_q512", 0): "price_global_q512",
        run.arm_name("global_q512", 512): "global_q512",
        run.arm_name("typed_q512", 0): "price_typed_q512",
    }.items():
        for role in run.base.ROLE_PATHS:
            arms[name]["roles"][role]["all"]["ce"] = parent[parent_name]["roles"][role]["all"]["ce"]
    gates = run._result_gates(arms, run.base.COVERAGE)
    assert gates["covered_identity_control"] is True
    assert gates["literal_endpoint_controls"] is True
    assert gates["integrity_conjunction"] is True


def test_authority_payload_is_nonauthoritative_discovery_and_binds_parent():
    source = {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64}
    checkpoint = {"weights_sha256": "c" * 64}
    authority = run.authority_payload(source, run.FILE_PINS, checkpoint)
    assert authority["status"] == "frozen_before_any_row_tensor_or_model_load"
    assert authority["protocol"]["parent"]["results_file_sha256"] == (
        run.PARENT_HASHES[str(run.PARENT_RESULTS.relative_to(run.ROOT))]
    )
    assert authority["protocol"]["authority_scope"] == (
        "discovery_only_no_validation_final_or_semantic_coordinates"
    )
    assert authority["protocol"]["call_schedule"] == run.expected_call_schedule()


def test_production_namespace_is_still_unopened():
    assert not any(path.exists() for path in (
        run.AUTHORITY, run.RESULTS, run.FAILURE, run.RECEIPT, run.LOCK,
    ))
