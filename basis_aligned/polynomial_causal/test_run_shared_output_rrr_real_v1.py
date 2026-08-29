from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import run_shared_output_rrr_real_v1 as run
import simultaneous_shared_output_rrr as core


def test_registered_arm_bank_prices_and_exact_direct_pair():
    arms = run.arm_descriptors()
    assert len(arms) == 24
    assert {a["name"] for a in arms} >= {
        "global_q64", "typed_q512", "independent_q256",
        "price_global_q128", "price_typed_q512", "global_q494",
        "typed_q481", "legacy_svd_q64", "legacy_svd_q512",
    }
    assert 37 * 494 == 38 * 481
    assert run.RIDGE == pytest.approx(0.01 * 5419 / 1152)
    schedule = run.expected_call_schedule()
    assert schedule == {
        "fit_native_outer": 22,
        "native_reference_outer": 120,
        "compiled_outer_per_arm": 120,
        "compiled_arm_count": 24,
        "compiled_outer_total": 2880,
        "outer_total": 3022,
        "native_component_calls_per_kind": 2556,
        "compiled_component_calls": 103680,
        "optimizer_calls": 0,
        "backward_calls": 0,
    }


def _synthetic_state(d=4, sites=4):
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(20, d, generator=generator, dtype=torch.float64)
    ys = torch.randn(sites, 20, d, generator=generator, dtype=torch.float64)
    gram = x.T @ x
    crosses = tuple(x.T @ ys[j] for j in range(sites))
    ridge = 0.1
    chol = torch.linalg.cholesky(gram + ridge * torch.eye(d, dtype=torch.float64))
    solved = tuple(torch.cholesky_solve(c, chol) for c in crosses)
    merits = tuple(c.T @ s for c, s in zip(crosses, solved, strict=True))
    indep = tuple(run._descending_eigh(m) for m in merits)
    gv, ge = run._descending_eigh(sum(merits[1:], merits[0].clone()))
    tv0, te0 = run._descending_eigh(merits[0] + merits[1])
    tv1, te1 = run._descending_eigh(merits[2] + merits[3])
    return run.SpectralState(
        gram, crosses, tuple(float(y.square().sum()) for y in ys), solved,
        tuple(v for v, _ in indep), tuple(e for _, e in indep), gv, ge,
        {"mlp": tv0, "attn": tv1}, {"mlp": te0, "attn": te1},
    )


def test_spectral_solver_matches_core_for_global_known_answer(monkeypatch):
    state = _synthetic_state()
    expected = core.fit_shared_output_basis(
        [state.gram] * 4, list(state.crosses), rank=2, ridge=0.1,
    )
    ours = state.global_vectors[:, :2]
    assert torch.allclose(ours @ ours.T, expected["projector"], atol=1e-10, rtol=1e-10)


def test_fit_program_exact_price_and_float32(monkeypatch):
    # Exercise the production pricing path at small dimension by replacing constants.
    monkeypatch.setattr(run, "D", 4)
    monkeypatch.setattr(run, "N_SITES", 4)
    monkeypatch.setattr(run, "COMMON_TABLE_FLOATS", 4 * 5 * 4)
    state = _synthetic_state()
    descriptor = {"name": "small", "family": "global", "rank": 2}
    program = run.fit_program(descriptor, state)
    assert len(program.input_maps) == 4
    assert all(value.dtype == torch.float32 for value in program.input_maps)
    assert all(value.dtype == torch.float32 for value in program.bases.values())
    assert program.diagnostics["map_float_count"] == (4 + 1) * 4 * 2
    assert program.diagnostics["dense_multiplies_per_uncovered_token"] == 4 * 2 * 4 * 2


def test_exact_storage_allocation_uses_fit_spectra(monkeypatch):
    monkeypatch.setattr(run, "D", 4)
    monkeypatch.setattr(run, "N_SITES", 4)
    monkeypatch.setattr(run, "COMMON_TABLE_FLOATS", 80)
    state = _synthetic_state()
    descriptor = {"name": "small", "family": "price_independent", "rank": 2,
                  "budget_bases": 2}
    program = run.fit_program(descriptor, state)
    target = core.grouped_map_price(4, 2, 4, 4, 2).grouped_float_count
    assert program.diagnostics["map_float_count"] == target
    assert sum(program.ranks_by_site) == target // 8


def test_serialized_exact_storage_allocation_replays_and_rejects_rank_corruption(monkeypatch):
    monkeypatch.setattr(run, "D", 4)
    monkeypatch.setattr(run, "N_SITES", 4)
    spectra = [
        [10.0, 9.0, 8.0, 0.1], [7.0, 0.6, 0.5, 0.4],
        [6.0, 5.0, 0.3, 0.2], [4.0, 3.0, 2.0, 1.0],
    ]
    descriptor = {"family": "price_independent", "rank": 2, "budget_bases": 2}
    ranks = run.replay_price_allocation(spectra, descriptor)
    assert sum(ranks) == core.grouped_map_price(4, 2, 4, 4, 2).grouped_float_count // 8
    corrupted = list(ranks)
    corrupted[0] -= 1
    corrupted[1] += 1
    assert tuple(corrupted) != run.replay_price_allocation(spectra, descriptor)


def test_frontier_anchor_replay_is_exact(tmp_path):
    payload = {"results": {role: {
        "m64_full": {"all": run.FRONTIER_ANCHORS["legacy_svd_q64"][role]},
        "m512_full": {"all": run.FRONTIER_ANCHORS["legacy_svd_q512"][role]},
    } for role in run.ROLE_PATHS}}
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps(payload))
    run.validate_frontier_anchors(path)
    payload["results"]["skip7000"]["m64_full"]["all"] += 1e-6
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="known-answer"):
        run.validate_frontier_anchors(path)


class _Embedding(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(vocab * d, dtype=torch.float32).reshape(vocab, d) / 10)

    def forward(self, tokens):
        return self.weight[tokens]


class _Attn(nn.Module):
    n_head = 1
    head_dim = 2


class _Block:
    def __init__(self):
        self.attn = _Attn()


def test_autonomous_program_uses_token_only_and_zero_v1(monkeypatch):
    monkeypatch.setattr(run, "D", 2)
    monkeypatch.setattr(run, "N_SITES", 2)
    monkeypatch.setattr(run, "SITE_TO_INDEX", {("mlp", 0): 0, ("attn", 0): 1})
    model = SimpleNamespace(transformer=SimpleNamespace(wte=_Embedding(5, 2)))
    table = torch.tensor([[[10., 11.]], [[20., 21.]]])
    mapping = torch.tensor([-1, 0, -1, -1, -1])
    factors = run.FactorProgram(
        "arm", {"name": "arm"}, {"g": torch.eye(2)}, ("g", "g"),
        (torch.eye(2), torch.eye(2)), (2, 2), {},
    )
    ledger = run.PhysicalCallLedger()
    monkeypatch.setattr(run, "arm_descriptors", lambda: ({"name": "arm"},))
    program = run.AutonomousProgram(model, table, mapping, factors, ledger, "cpu")
    poison = object()  # state must never be inspected
    event = SimpleNamespace(site=0, block=_Block(), state=poison,
                            tokens=torch.tensor([[1, 2]]), first_value=poison)
    write, sentinel = program.attention(event)
    assert torch.equal(write[0, 0], table[1, 0])
    assert torch.allclose(write[0, 1], model.transformer.wte.weight[2])
    assert sentinel.shape == (1, 2, 1, 2)
    assert torch.count_nonzero(sentinel) == 0


class _TinyAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n_head, self.head_dim = 1, d
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, state, first):
        value = self.proj(state)
        if first is None:
            first = value + 7.0
        return value + (0.01 * first), first


class _TinyMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, state):
        return self.proj(state)


class _TinyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]), requires_grad=False)
        self.attn, self.mlp = _TinyAttention(d), _TinyMLP(d)


class _TinyModel(nn.Module):
    def __init__(self, d=2, vocab=5, layers=2):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab)
        self.transformer = SimpleNamespace(wte=_Embedding(vocab, d), h=nn.ModuleList([
            _TinyBlock(d) for _ in range(layers)
        ]))
        self.lm_head = nn.Linear(d, vocab, bias=False)


def test_zero_v1_autonomous_replacement_matches_native_v1_postreplacement():
    # Once every attention write is replaced, the native v1 bus is observationally inert.
    model = _TinyModel()
    tokens = torch.tensor([[0, 1, 2]])
    writes = {(kind, site): torch.full(
        (1, 3, 2), site + (1 if kind == "attn" else 3.0), dtype=torch.float32,
    )
              for site in range(2) for kind in ("attn", "mlp")}

    def legacy_attention(event):
        _, native_v1 = event.block.attn(event.state, event.first_value)
        return writes[("attn", event.site)], native_v1

    def legacy_mlp(event):
        event.block.mlp(event.state)
        return writes[("mlp", event.site)]

    def autonomous_attention(event):
        return writes[("attn", event.site)], torch.zeros((*tokens.shape, 1, 2))

    def autonomous_mlp(event):
        return writes[("mlp", event.site)]

    legacy = run.facade.forward_with_dispatch(
        model, tokens, legacy_attention, legacy_mlp, require_production=False,
    )
    autonomous = run.facade.forward_with_dispatch(
        model, tokens, autonomous_attention, autonomous_mlp, require_production=False,
    )
    assert torch.equal(legacy, autonomous)


def _fill_exact_ledger(monkeypatch):
    monkeypatch.setattr(run, "FIT_OUTER_CALLS", 2)
    monkeypatch.setattr(run, "EVAL_CALLS_PER_ARM", 3)
    monkeypatch.setattr(run, "arm_descriptors", lambda: ({"name": "a"}, {"name": "b"}))
    monkeypatch.setattr(run, "expected_call_schedule", lambda: {"frozen": True})
    ledger = run.PhysicalCallLedger()
    for _ in range(2):
        ledger.record_native_outer("fit")
        for kind in ("attn", "mlp"):
            for site in range(18): ledger.record_native_site("fit", kind, site)
    for _ in range(3):
        ledger.record_native_outer("native_reference")
        for kind in ("attn", "mlp"):
            for site in range(18): ledger.record_native_site("native_reference", kind, site)
        for arm in ("a", "b"):
            ledger.record_compiled_outer(arm)
            for kind in ("attn", "mlp"):
                for site in range(18): ledger.record_compiled_site(arm, kind, site)
    return ledger


def test_call_ledger_accepts_exact_and_rejects_one_missing(monkeypatch):
    ledger = _fill_exact_ledger(monkeypatch)
    receipt = ledger.receipt()
    assert receipt["optimizer_calls"] == 0
    run.semantic_validate_call_ledger(receipt)
    ledger.compiled_site[("a", "mlp")][3] -= 1
    with pytest.raises(RuntimeError, match="compiled"):
        ledger.receipt()


def test_semantic_call_replay_rejects_self_consistent_count_corruption(monkeypatch):
    ledger = _fill_exact_ledger(monkeypatch).receipt()
    ledger["compiled_outer"]["a"] -= 1
    with pytest.raises(RuntimeError, match="header"):
        run.semantic_validate_call_ledger(ledger)


@pytest.mark.parametrize("field,bad", [("elapsed_seconds", float("nan")),
                                        ("maximum_allocated_cuda_bytes", 16 * 1024 ** 3 + 1)])
def test_result_replay_rejects_resource_corruption(field, bad):
    values = {"elapsed_seconds": 1.0, "maximum_allocated_cuda_bytes": 0}
    values[field] = bad
    with pytest.raises(RuntimeError, match="resource"):
        run.semantic_validate_resources(values["elapsed_seconds"],
                                        values["maximum_allocated_cuda_bytes"])


def test_physical_call_replay_rejects_one_hidden_native_call(monkeypatch):
    monkeypatch.setattr(run, "FIT_OUTER_CALLS", 2)
    monkeypatch.setattr(run, "EVAL_CALLS_PER_ARM", 3)
    value = {kind: {str(site): 5 for site in range(18)} for kind in ("attn", "mlp")}
    run.semantic_validate_physical_calls(value)
    value["attn"]["7"] += 1
    with pytest.raises(RuntimeError, match="physical"):
        run.semantic_validate_physical_calls(value)


def test_metric_replay_rejects_nonfinite_and_wrong_denominator():
    metric = {"covered": {"ce": 1.0, "tokens": 2},
              "uncovered": {"ce": 2.0, "tokens": 3},
              "all": {"ce": 1.6, "tokens": 5}}
    run.semantic_validate_metric_ledger(metric, 5)
    metric["covered"]["ce"] = float("inf")
    with pytest.raises(RuntimeError, match="malformed"):
        run.semantic_validate_metric_ledger(metric, 5)


def test_metric_replay_rejects_negative_and_nonweighted_all_ce():
    metric = {"covered": {"ce": 1.0, "tokens": 2},
              "uncovered": {"ce": 2.0, "tokens": 3},
              "all": {"ce": 1.6, "tokens": 5}}
    metric["all"]["ce"] = 1.5
    with pytest.raises(RuntimeError, match="weighted partition"):
        run.semantic_validate_metric_ledger(metric, 5)
    metric["all"]["ce"] = 1.6
    metric["covered"]["ce"] = -0.1
    with pytest.raises(RuntimeError, match="malformed"):
        run.semantic_validate_metric_ledger(metric, 5)


def test_authority_builder_is_outcome_free_and_binds_exact_arm_bank():
    source = {"commit": "c" * 40, "paths": {name: "a" * 64 for name in run.SOURCE_PATHS},
              "sha256": "b" * 64}
    checkpoint = {
        "revision": "r", "snapshot": "/x", "config_sha256": "1" * 64,
        "weights_sha256": "2" * 64, "weights_bytes": 3,
        "tokenizer_vocab": 50257, "logit_vocab": 50304,
    }
    value = run.authority_payload(source, run.FILE_PINS, checkpoint)
    assert value["status"] == "frozen_before_any_row_tensor_or_model_load"
    assert len(value["protocol"]["arms"]) == 24
    assert not any(key in json.dumps(value).lower() for key in ("observed_ce", "gate_pass"))


def test_atomic_create_json_is_create_only(tmp_path):
    path = tmp_path / "x.json"
    run.atomic_create_json({"a": 1}, path)
    with pytest.raises(FileExistsError):
        run.atomic_create_json({"a": 2}, path)
    assert json.loads(path.read_text()) == {"a": 1}


def test_lock_replacement_is_rejected(tmp_path):
    path = tmp_path / "lock"
    claim = run.acquire_lock(path)
    path.unlink()
    path.write_text(json.dumps({"pid": 0, "nonce": "wrong"}))
    try:
        with pytest.raises(RuntimeError, match="replaced"):
            run.require_claim(claim, path)
    finally:
        path.unlink()
        # The original inode is still held but no longer named.
        __import__("os").close(claim.fd)


def _minimal_result(monkeypatch):
    monkeypatch.setattr(run, "ROLE_PATHS", {"r": Path("r")})
    monkeypatch.setattr(run, "ROLE_ROWS", {"r": 1})
    monkeypatch.setattr(run, "SCORED_PER_ROW", 2)
    monkeypatch.setattr(run, "COVERAGE", 1)
    monkeypatch.setattr(run, "COMMON_TABLE_FLOATS", 2)
    monkeypatch.setattr(run, "arm_descriptors", lambda: (
        {"name": "global_q64", "family": "global", "rank": 1},
        {"name": "independent_q64", "family": "independent", "rank": 1},
        {"name": "price_global_q64", "family": "price_independent", "rank": 1,
         "budget_bases": 1},
        {"name": "typed_q481", "family": "typed", "rank": 1},
        {"name": "global_q494", "family": "global", "rank": 1},
        {"name": "legacy_svd_q64", "family": "legacy_svd", "rank": 1},
        {"name": "legacy_svd_q512", "family": "legacy_svd", "rank": 1},
    ))
    monkeypatch.setattr(run, "RANKS", (64,))
    monkeypatch.setattr(run, "FRONTIER_ANCHORS", {
        "legacy_svd_q64": {"r": 1.0}, "legacy_svd_q512": {"r": 1.0},
    })
    def arm(desc, ce=1.0):
        return {"descriptor": desc, "diagnostics": {
            "finite": True, "map_float_count": 8 if desc["family"] == "price_independent" else 1,
            "map_float_bytes": 32 if desc["family"] == "price_independent" else 4,
            "common_table_float_count": 2,
            "full_program_float_count": 10 if desc["family"] == "price_independent" else 3,
            "full_program_float_bytes": 40 if desc["family"] == "price_independent" else 12,
        }, "roles": {"r": {"covered": {"ce": ce, "tokens": 1},
                              "uncovered": {"ce": ce, "tokens": 1},
                              "all": {"ce": ce, "tokens": 2}}}}
    descs = {d["name"]: d for d in run.arm_descriptors()}
    arms = {name: arm(desc) for name, desc in descs.items()}
    return arms


def test_gate_replay_rejects_corruption(monkeypatch):
    arms = _minimal_result(monkeypatch)
    gates = run._result_gates(arms, 1)
    assert gates["covered_identity_control"] is True
    arms["global_q64"]["roles"]["r"]["covered"]["ce"] += 0.1
    assert run._result_gates(arms, 1)["covered_identity_control"] is False


def test_promotive_gates_fail_closed_when_any_common_control_fails(monkeypatch):
    arms = _minimal_result(monkeypatch)
    monkeypatch.setattr(
        run.core, "grouped_map_price",
        lambda *_args, **_kwargs: SimpleNamespace(grouped_float_count=8),
    )
    arms["global_q64"]["roles"]["r"]["uncovered"]["ce"] = 0.9
    arms["global_q64"]["roles"]["r"]["all"]["ce"] = 0.9
    arms["typed_q481"]["roles"]["r"]["uncovered"]["ce"] = 0.9
    arms["typed_q481"]["roles"]["r"]["all"]["ce"] = 0.9
    good = run._result_gates(arms, 1)
    assert good["promotive_controls_all_pass"] is True
    assert good["e2_1_pass"] is True
    assert good["e2_2_pass"] is True
    bad = run._result_gates(arms, 0)
    assert bad["e2_1_ce_qualifying_ranks"] == [64]
    assert bad["e2_2_ce_qualifies"] is True
    assert bad["promotive_controls_all_pass"] is False
    assert bad["e2_1_pass"] is False
    assert bad["e2_2_pass"] is False


def test_failure_after_result_preserves_result_and_never_writes_receipt(tmp_path, monkeypatch):
    authority, results, failure, receipt, lock = (
        tmp_path / "authority.json", tmp_path / "results.json", tmp_path / "failure.json",
        tmp_path / "receipt.json", tmp_path / "lock",
    )
    monkeypatch.setattr(run, "AUTHORITY", authority)
    monkeypatch.setattr(run, "RESULTS", results)
    monkeypatch.setattr(run, "FAILURE", failure)
    monkeypatch.setattr(run, "RECEIPT", receipt)
    monkeypatch.setattr(run, "LOCK", lock)
    run.atomic_create_json({"authority": 1}, authority)
    run.atomic_create_json({"result": 1}, results)
    claim = run.acquire_lock(lock)
    try:
        run.publish_failure(claim, RuntimeError("late"))
    finally:
        run.release_lock(claim, lock)
    assert json.loads(results.read_text()) == {"result": 1}
    assert json.loads(failure.read_text())["results_exists"] is True
    assert not receipt.exists()


def test_preauthority_refusal_cannot_spend_failure_namespace(tmp_path, monkeypatch):
    authority, failure, lock = tmp_path / "authority.json", tmp_path / "failure.json", tmp_path / "lock"
    monkeypatch.setattr(run, "AUTHORITY", authority)
    monkeypatch.setattr(run, "FAILURE", failure)
    monkeypatch.setattr(run, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(run, "LOCK", lock)
    claim = run.acquire_lock(lock)
    try:
        run.publish_failure(claim, RuntimeError("source not committed"))
    finally:
        run.release_lock(claim, lock)
    assert not failure.exists()


def test_receipt_exactly_joins_published_files(tmp_path, monkeypatch):
    authority, results, failure = tmp_path / "a.json", tmp_path / "r.json", tmp_path / "f.json"
    monkeypatch.setattr(run, "AUTHORITY", authority)
    monkeypatch.setattr(run, "RESULTS", results)
    monkeypatch.setattr(run, "FAILURE", failure)
    frozen = {"authority_sha256": "a" * 64, "source_closure": {"sha256": "b" * 64},
              "input_file_sha256s": {"x": "c" * 64}}
    result = {"x": 1}
    run.atomic_create_json(frozen, authority)
    run.atomic_create_json(result, results)
    receipt = run.receipt_payload(frozen, result)
    run.semantic_validate_receipt(receipt, frozen, result)
    assert receipt["authority_file_sha256"] == run.file_sha256(authority)
    assert receipt["results_file_sha256"] == run.file_sha256(results)
    assert receipt["failure_absent"] is True
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == run.logical_sha256(body)


def test_receipt_replay_rejects_result_join_corruption(tmp_path, monkeypatch):
    authority, results, failure = tmp_path / "a.json", tmp_path / "r.json", tmp_path / "f.json"
    monkeypatch.setattr(run, "AUTHORITY", authority)
    monkeypatch.setattr(run, "RESULTS", results)
    monkeypatch.setattr(run, "FAILURE", failure)
    frozen = {"authority_sha256": "a" * 64, "source_closure": {"sha256": "b" * 64},
              "input_file_sha256s": {"x": "c" * 64}}
    result = {"x": 1}
    run.atomic_create_json(frozen, authority)
    run.atomic_create_json(result, results)
    receipt = run.receipt_payload(frozen, result)
    receipt["results_logical_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="joins"):
        run.semantic_validate_receipt(receipt, frozen, result)


def test_receipt_replay_rejects_mutated_published_inputs(tmp_path, monkeypatch):
    authority, results, failure = tmp_path / "a.json", tmp_path / "r.json", tmp_path / "f.json"
    monkeypatch.setattr(run, "AUTHORITY", authority)
    monkeypatch.setattr(run, "RESULTS", results)
    monkeypatch.setattr(run, "FAILURE", failure)
    frozen = {"authority_sha256": "a" * 64, "source_closure": {"sha256": "b" * 64},
              "input_file_sha256s": {"x": "c" * 64}}
    result = {"x": 1}
    run.atomic_create_json(frozen, authority)
    run.atomic_create_json(result, results)
    receipt = run.receipt_payload(frozen, result)
    results.write_text(json.dumps({"x": 2}))
    with pytest.raises(RuntimeError, match="inputs changed"):
        run.semantic_validate_receipt(receipt, frozen, result)
    results.write_text(json.dumps(result))
    authority.write_text(json.dumps({**frozen, "authority_sha256": "d" * 64}))
    with pytest.raises(RuntimeError, match="inputs changed"):
        run.semantic_validate_receipt(receipt, frozen, result)


def test_source_closure_contains_every_direct_runtime_and_test_source():
    assert set(run.SOURCE_PATHS) == {
        "basis_aligned/polynomial_causal/SHARED_OUTPUT_RRR_REAL_V1_PREREGISTRATION.md",
        "basis_aligned/polynomial_causal/run_shared_output_rrr_real_v1.py",
        "basis_aligned/polynomial_causal/test_run_shared_output_rrr_real_v1.py",
        "basis_aligned/polynomial_causal/simultaneous_shared_output_rrr.py",
        "basis_aligned/polynomial_causal/test_simultaneous_shared_output_rrr.py",
        "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
        "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
        "jacclust/tt_model.py", "jacclust/__init__.py",
    }
