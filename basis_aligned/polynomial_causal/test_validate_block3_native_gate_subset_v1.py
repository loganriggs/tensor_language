import json

import pytest
import torch
from torch import nn

import native_gate_subset as subset
import validate_block3_native_gate_subset_v1 as validation


class _Attention(nn.Module):
    def forward(self, value, first):
        return 0.1 * value, value if first is None else first


class _MLP(nn.Module):
    def __init__(self, width, gates):
        super().__init__()
        self.Left = nn.Linear(width, gates, bias=False)
        self.Right = nn.Linear(width, gates, bias=False)
        self.Down = nn.Linear(gates, width, bias=False)
        self.Down_bias = nn.Parameter(torch.randn(width))

    def forward(self, value):
        return self.Down(self.Left(value) * self.Right(value)) + self.Down_bias


class _Block(nn.Module):
    def __init__(self, width, gates):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.attn = _Attention()
        self.mlp = _MLP(width, gates)


class _TinyModel(nn.Module):
    def __init__(self, width=4, gates=6, vocab=11):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, width)
        self.transformer.h = nn.ModuleList([_Block(width, gates) for _ in range(18)])
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def test_validation_contract_preserves_bias_and_intervenes_all_positions():
    text = validation.AMENDMENT.read_text()
    assert "`z_Q=b`" in text
    assert "all positions 0--255" in text
    assert "smallest validation-eligible" in text
    assert "complete 16/15" in text
    v1 = validation.V1_AMENDMENT.read_text()
    assert "before any candidate" in v1
    assert "validation_v1" in v1
    assert "e_\\infty" in v1


def test_v1_namespace_is_distinct_and_binds_terminal_v0_failure():
    assert all("validation_v1" in path.name for path in (
        validation.AUTHORITY, validation.RESULTS, validation.RECEIPT, validation.FAILURE,
    ))
    assert validation.V0_AUTHORITY.exists()
    assert validation.V0_FAILURE.exists()
    assert not validation.V0_RESULTS.exists()
    assert not validation.V0_RECEIPT.exists()
    lineage = validation.v0_failure_lineage()
    assert lineage["candidate_arms_scored"] == 0
    assert lineage["result_exists"] is False
    assert lineage["receipt_exists"] is False


def test_real_row_binding_clusters_rows_by_source_document():
    binding = validation.row_binding()
    mapping = torch.tensor(binding["row_to_document"], dtype=torch.long)
    counts = torch.bincount(mapping)
    assert len(mapping) == 192
    assert len(binding["ordered_document_ids"]) == 79
    assert len(counts) == 79
    assert int((counts > 1).sum()) == 38
    assert int(counts.max()) == 18


def test_prefix_and_suffix_are_exact_autonomous_trajectory(monkeypatch):
    width = 4
    monkeypatch.setattr(validation, "WIDTH", width)
    monkeypatch.setattr(validation, "MODEL_TOKENS", 5)
    model = _TinyModel(width=width)
    tokens = torch.randint(0, 11, (2, 5), generator=torch.Generator().manual_seed(3))
    calls = validation.CallLedger.empty()
    prefix = validation.prefix_to_mlp3(model, tokens, calls)
    assert torch.allclose(prefix.u + prefix.v, prefix.z, atol=1e-6)
    states, logits = validation.suffix_from_write(model, prefix, prefix.native_write, calls)

    x = torch.nn.functional.rms_norm(model.transformer.wte(tokens), (width,))
    x0, first = x, None
    for block in model.transformer.h:
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first = block.attn(torch.nn.functional.rms_norm(x, (width,)), first)
        x = x + attention
        x = x + block.mlp(torch.nn.functional.rms_norm(x, (width,)))
    expected = model.lm_head(torch.nn.functional.rms_norm(x, (width,)))
    expected = 30 * torch.tanh(expected / 30)
    assert torch.allclose(states[17], x, atol=1e-6)
    assert torch.allclose(logits, expected.float(), atol=1e-6)


def test_arm_writes_replay_native_and_bias_only_is_registered_omission(monkeypatch):
    width, gates = 4, 6
    monkeypatch.setattr(validation, "WIDTH", width)
    monkeypatch.setattr(validation, "MODEL_TOKENS", 5)
    model = _TinyModel(width=width, gates=gates)
    tokens = torch.randint(0, 11, (2, 5), generator=torch.Generator().manual_seed(4))
    calls = validation.CallLedger.empty()
    prefix = validation.prefix_to_mlp3(model, tokens, calls)
    mlp = model.transformer.h[3].mlp
    indices = torch.arange(gates)
    program = subset.build_program(
        mlp.Left.weight, mlp.Right.weight, mlp.Down_bias,
        indices, mlp.Down.weight,
    )
    balanced_left, balanced_right, _ = validation.collector.balance_product_gauge(
        mlp.Left.weight, mlp.Right.weight,
    )
    writes, native_terms, candidate_terms, replay = validation.arm_writes(
        prefix, model.transformer.h[3], program, program, program,
        balanced_left, balanced_right, gates, calls,
    )
    assert replay["native"]["relative_max"] < 2e-5
    assert replay["native"]["relative_rms"] < 2e-5
    assert replay["candidate"]["relative_max"] < 2e-5
    assert replay["candidate"]["relative_rms"] < 2e-5
    assert torch.allclose(writes["activation_k6"], prefix.native_write, atol=2e-5)
    assert torch.allclose(
        writes["bias_only"], mlp.Down_bias.reshape(1, 1, -1).expand_as(prefix.native_write),
    )
    for name in subset.TERM_NAMES:
        assert torch.allclose(native_terms[name], candidate_terms[name], atol=2e-5)
        expected = mlp.Down_bias + sum(
            native_terms[other] for other in subset.TERM_NAMES if other != name
        )
        assert torch.allclose(writes[f"omit_{name}"], expected, atol=2e-5)


def test_summed_local_nrmse_uses_bias_free_native_energy(monkeypatch):
    monkeypatch.setattr(validation, "POSITION_START", 0)
    monkeypatch.setattr(validation, "POSITION_STOP", 2)
    ledger = validation.empty_ledger(1)
    bias = torch.tensor([7.0, -3.0])
    signal = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    native = signal + bias
    arm = bias.reshape(1, 1, -1).expand_as(native)
    validation.accumulate_local(ledger, native, arm, bias, torch.tensor([0]))
    assert torch.equal(ledger["local_sse"], ledger["local_energy"])


def test_replay_guard_is_scale_relative_and_checks_both_norms():
    reference = torch.tensor([5493.6538, -600.0, 1.0], dtype=torch.float32)
    numerically_equivalent = reference + torch.tensor(
        [0.009765625, 0.0005, 0.0], dtype=torch.float32,
    )
    diagnostics = validation.replay_diagnostics(reference, numerically_equivalent)
    assert diagnostics["absolute_max"] > 3e-4
    validation.require_replay(diagnostics, label="known float32 reorder")

    wrong = validation.replay_diagnostics(reference, reference * 1.001)
    with pytest.raises(RuntimeError, match="relative replay failed"):
        validation.require_replay(wrong, label="algebraic mismatch")


def test_replay_guard_rejects_nonfinite_and_independent_max_or_rms_failure():
    with pytest.raises(RuntimeError, match="nonfinite tensor"):
        validation.replay_diagnostics(
            torch.tensor([1.0]), torch.tensor([float("inf")]),
        )

    reference = torch.full((10_000,), 1_000.0)
    max_only = reference.clone()
    max_only[0] += 0.1
    max_diagnostics = validation.replay_diagnostics(reference, max_only)
    assert max_diagnostics["relative_max"] > 2e-5
    assert max_diagnostics["relative_rms"] < 2e-5
    with pytest.raises(RuntimeError, match="relative replay failed"):
        validation.require_replay(max_diagnostics, label="max-only failure")

    sparse_reference = torch.zeros(10_000)
    sparse_reference[0] = 1_000.0
    rms_only = sparse_reference + 0.01
    rms_diagnostics = validation.replay_diagnostics(sparse_reference, rms_only)
    assert rms_diagnostics["relative_max"] < 2e-5
    assert rms_diagnostics["relative_rms"] > 2e-5
    with pytest.raises(RuntimeError, match="relative replay failed"):
        validation.require_replay(rms_diagnostics, label="rms-only failure")


def _filled_ledger(documents, *, local=0.1, kl=1.0):
    ledger = validation.empty_ledger(documents)
    ledger["token_count"].fill_(1)
    ledger["local_energy"].fill_(1)
    ledger["local_sse"].fill_(local * local)
    ledger["kl_sum"].fill_(kl)
    ledger["centered_stake_energy"].fill_(1)
    ledger["centered_error"].fill_(0.01)
    ledger["response_dot"].fill_(1)
    ledger["response_native_energy"].fill_(1)
    ledger["response_arm_energy"].fill_(1)
    ledger["top1_agree"].fill_(1)
    ledger["target_top1"].fill_(1)
    for values in ledger["cuts"].values():
        values["error"].fill_(local * local)
        values["native_energy"].fill_(1)
    return ledger


def _fake_wave(*, budget, ledgers, term_ledgers, include_omissions, **kwargs):
    documents = validation.ROW_COUNT
    if include_omissions:
        ledgers["bias_only"] = _filled_ledger(documents, local=1, kl=10)
        for name in subset.TERM_NAMES:
            ledgers[f"omit_{name}"] = _filled_ledger(documents, kl=2)
    for name, kl in (
        (f"activation_k{budget}", 1), (f"random_k{budget}", 2),
        (f"permutation_k{budget}", 3), (f"mirror_k{budget}", 1),
    ):
        ledgers[name] = _filled_ledger(documents, kl=kl)
    for name in subset.TERM_NAMES:
        ledgers[f"activation_k{budget}_{name}"] = _filled_ledger(documents, kl=1)
    term_ledgers[f"activation_k{budget}"] = validation.empty_term_ledger(documents)
    for values in term_ledgers[f"activation_k{budget}"].values():
        values["sse"].fill_(0.01)
        values["energy"].fill_(1)
    return 0.0


def _configure_transaction(monkeypatch, tmp_path, *, fail_terminal=False):
    for name in ("AUTHORITY", "RESULTS", "RECEIPT", "FAILURE", "LOCK"):
        monkeypatch.setattr(validation, name, tmp_path / name.lower())
    monkeypatch.setattr(validation, "ROW_COUNT", 2)
    monkeypatch.setattr(validation, "BOOTSTRAP_DRAWS", 20)
    source = {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64}
    fit = {"file_sha256s": {}, "fit_authority_sha256": "c" * 64}
    rows_binding = {
        "ordered_document_ids": ["d0", "d1"], "row_file_sha256": "d" * 64,
        "row_to_document": [0, 1],
    }
    checkpoint = validation.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="e" * 64,
        weights_sha256="f" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=11,
    )
    monkeypatch.setattr(validation, "source_closure", lambda: source)
    monkeypatch.setattr(validation, "validate_fit_artifacts", lambda: ({"programs": {}}, fit))
    monkeypatch.setattr(validation, "row_binding", lambda: rows_binding)
    checks = 0

    def verify_inputs(*args):
        nonlocal checks
        checks += 1
        if fail_terminal and checks == 3:
            raise RuntimeError("injected validation terminal drift")

    monkeypatch.setattr(validation, "verify_inputs", verify_inputs)
    monkeypatch.setattr(validation, "load_rows", lambda binding: torch.zeros(2, 513, dtype=torch.long))
    monkeypatch.setattr(validation.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    model = nn.Linear(2, 2, bias=False).eval()
    monkeypatch.setattr(validation.facade, "load_bilin18", lambda **kwargs: (model, checkpoint))
    monkeypatch.setattr(validation, "materialize_program", lambda value: object())
    monkeypatch.setattr(validation, "run_wave", _fake_wave)
    monkeypatch.setattr(validation, "validate_call_ledger", lambda *args, **kwargs: None)
    return checkpoint


@pytest.mark.parametrize("opened_512,suffix_per_batch", ((False, 14), (True, 24)))
def test_measured_call_census_matches_adaptive_wave(opened_512, suffix_per_batch, monkeypatch):
    monkeypatch.setattr(validation, "ROW_COUNT", 8)
    monkeypatch.setattr(validation, "BATCH_SIZE", 4)
    batches = 2
    passes = 2 if opened_512 else 1
    calls = validation.CallLedger.empty()
    for site in range(4):
        calls.attention[site] = batches * passes
        calls.mlp[site] = batches * passes
    for site in range(4, 18):
        calls.attention[site] = batches * suffix_per_batch
        calls.mlp[site] = batches * suffix_per_batch
    waves = ((256, True),) + (((512, False),) if opened_512 else ())
    for budget, include_omissions in waves:
        wave = f"k{budget}"
        calls.prefix_batches_by_wave[wave] = batches
        calls.teacher_mlp3_calls_by_wave[wave] = batches
        for arm in validation._wave_arms(budget, include_omissions=include_omissions):
            calls.suffix_calls_by_wave_arm[f"{wave}/{arm}"] = batches
            validation._increment(
                calls.suffix_calls_by_family, validation._suffix_family(arm), batches,
            )
        for term in subset.TERM_NAMES:
            calls.native_typed_down_calls_by_wave_term[f"{wave}/{term}"] = batches
            calls.candidate_typed_decoder_calls_by_wave_term[f"{wave}/{term}"] = batches
        for arm in (
            f"activation_k{budget}", f"random_k{budget}", f"permutation_k{budget}",
        ):
            calls.direct_program_calls_by_wave_arm[f"{wave}/{arm}"] = batches
    validation.validate_call_ledger(calls, opened_512=opened_512)


def test_call_census_rejects_cross_family_substitution(monkeypatch):
    monkeypatch.setattr(validation, "ROW_COUNT", 4)
    monkeypatch.setattr(validation, "BATCH_SIZE", 4)
    calls = validation.CallLedger.empty()
    for site in range(4):
        calls.attention[site] = calls.mlp[site] = 1
    for site in range(4, 18):
        calls.attention[site] = calls.mlp[site] = 14
    calls.prefix_batches_by_wave["k256"] = 1
    calls.teacher_mlp3_calls_by_wave["k256"] = 1
    for arm in validation._wave_arms(256, include_omissions=True):
        calls.suffix_calls_by_wave_arm[f"k256/{arm}"] = 1
        validation._increment(
            calls.suffix_calls_by_family, validation._suffix_family(arm), 1,
        )
    for term in subset.TERM_NAMES:
        calls.native_typed_down_calls_by_wave_term[f"k256/{term}"] = 1
        calls.candidate_typed_decoder_calls_by_wave_term[f"k256/{term}"] = 1
    for arm in ("activation_k256", "random_k256", "permutation_k256"):
        calls.direct_program_calls_by_wave_arm[f"k256/{arm}"] = 1
    calls.suffix_calls_by_family["mirror"] -= 1
    calls.suffix_calls_by_family["random_control"] += 1
    with pytest.raises(RuntimeError, match="call census"):
        validation.validate_call_ledger(calls, opened_512=False)


def test_validation_transaction_is_authority_first_and_receipt_last(monkeypatch, tmp_path):
    _configure_transaction(monkeypatch, tmp_path)

    def load_rows(binding):
        assert validation.AUTHORITY.exists() and not validation.RESULTS.exists()
        return torch.zeros(2, 513, dtype=torch.long)

    monkeypatch.setattr(validation, "load_rows", load_rows)
    result = validation.run()
    assert result["validation_eligible_budget"] == 256
    assert validation.RECEIPT.exists() and not validation.FAILURE.exists()
    assert not validation.LOCK.exists()
    receipt = json.loads(validation.RECEIPT.read_text())
    assert receipt["status"] == "validation_v1_complete_receipt_last"


def test_validation_terminal_drift_publishes_failure_without_receipt(monkeypatch, tmp_path):
    _configure_transaction(monkeypatch, tmp_path, fail_terminal=True)
    with pytest.raises(RuntimeError, match="terminal drift"):
        validation.run()
    assert validation.AUTHORITY.exists() and validation.RESULTS.exists()
    assert validation.FAILURE.exists() and not validation.RECEIPT.exists()
    assert not validation.LOCK.exists()
