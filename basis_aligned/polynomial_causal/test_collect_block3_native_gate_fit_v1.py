import torch
from torch import nn
import pytest

import collect_block3_native_gate_fit_v1 as collector
import native_gate_subset as subset


def _batch(seed: int = 0, *, n: int = 9):
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(n, collector.WIDTH, generator=generator, dtype=torch.float64)
    v = torch.randn(n, collector.WIDTH, generator=generator, dtype=torch.float64)
    # Keep this unit test small in gate count by testing the generic subset helpers
    # directly; collector shape constants are checked separately.
    left = torch.randn(13, collector.WIDTH, generator=generator, dtype=torch.float64)
    right = torch.randn(13, collector.WIDTH, generator=generator, dtype=torch.float64)
    down = torch.randn(collector.WIDTH, 13, generator=generator, dtype=torch.float64)
    features = subset.typed_gate_features(left, right, u, v)
    return u, v, down, features


def test_first_pass_statistics_match_direct_second_moments_and_energy(monkeypatch):
    u, v, down, features = _batch()
    monkeypatch.setattr(collector, "GATES", 13)
    observed = collector.first_pass_statistics(u, v, features, down)
    assert observed["count"] == len(u)
    assert torch.equal(observed["u_second"], u.T @ u)
    assert torch.equal(observed["v_second"], v.T @ v)
    assert torch.equal(observed["z_second"], (u + v).T @ (u + v))
    assert torch.equal(observed["energy"], subset.contribution_energy(features, down))


def test_second_pass_statistics_match_direct_stacked_regression(monkeypatch):
    _u, _v, down, features = _batch(seed=1)
    monkeypatch.setattr(collector, "GATES", 13)
    monkeypatch.setattr(collector, "PREFILTER", 5)
    indices = torch.tensor([0, 3, 5, 8, 12])
    gram, cross, permuted, write_energy, count = collector.second_pass_statistics(
        features, down, indices,
    )
    full, writes = subset.stack_features_and_writes(features, down)
    expected_gram, expected_cross = subset.sufficient_statistics(full[:, indices], writes)
    assert torch.equal(gram, expected_gram)
    # The collector concatenates the four write banks before the cross product;
    # the direct helper may choose a different GEMM accumulation order.
    assert torch.allclose(cross, expected_cross, rtol=1e-12, atol=1e-8)
    _, expected_permuted = subset.sufficient_statistics(
        full[:, indices], writes.flip(0),
    )
    assert torch.allclose(permuted, expected_permuted, rtol=1e-12, atol=1e-8)
    assert torch.equal(write_energy, writes.square().sum())
    assert count == len(full)


def test_collector_contract_prices_two_pass_fit_without_native_mlp3():
    assert collector.PASSES == 2
    assert collector.PREFILTER == 1024
    assert collector.POSITION_STOP - collector.POSITION_START == 192
    assert collector.ROW_COUNT * collector.TOKENS_PER_ROW == 92_160
    # Before execution both were absent; after the authoritative transaction the
    # payload is valid only under its terminal receipt and without a failure file.
    assert collector.PAYLOAD.exists() == collector.RECEIPT.exists()
    if collector.PAYLOAD.exists():
        assert not collector.FAILURE.exists()


def test_create_json_is_exclusive_and_never_overwrites(tmp_path):
    path = tmp_path / "artifact.json"
    collector.create_json(path, {"first": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        collector.create_json(path, {"second": 2})
    assert path.read_bytes() == before


def test_run_claim_detects_replacement_before_release(tmp_path):
    path = tmp_path / "claim.lock"
    claim = collector.acquire_claim(path)
    claim.verify()
    path.unlink()
    path.write_text("attacker\n")
    with pytest.raises(RuntimeError, match="replaced or altered"):
        claim.verify()
    # Close only the still-owned descriptor; the foreign pathname must survive.
    import os
    os.close(claim.descriptor)
    assert path.read_text() == "attacker\n"


class _Attention(nn.Module):
    def forward(self, value, first):
        return torch.zeros_like(value), value if first is None else first


class _MLP(nn.Module):
    def __init__(self, width, gates):
        super().__init__()
        self.Left = nn.Linear(width, gates, bias=False)
        self.Right = nn.Linear(width, gates, bias=False)
        self.Down = nn.Linear(gates, width, bias=False)
        self.Down_bias = nn.Parameter(torch.zeros(width))

    def forward(self, value):
        return self.Down(self.Left(value) * self.Right(value)) + self.Down_bias


class _Block(nn.Module):
    def __init__(self, width, gates):
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.attn = _Attention()
        self.mlp = _MLP(width, gates)


class _TinyModel(nn.Module):
    def __init__(self, width, gates):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(10, width)
        self.transformer.h = nn.ModuleList([_Block(width, gates) for _ in range(4)])
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def test_trajectory_content_hash_detects_upstream_parameter_mutation():
    model = _TinyModel(4, 6)
    before = collector.block0_through_3_state_sha256(model)
    with torch.no_grad():
        model.transformer.h[1].mlp.Left.weight[0, 0].add_(1)
    after = collector.block0_through_3_state_sha256(model)
    assert after != before


def _configure_tiny_run(monkeypatch, tmp_path, *, terminal_drift=None):
    width, gates = 4, 6
    for name, value in {
        "WIDTH": width, "GATES": gates, "ROW_COUNT": 2, "ROW_WIDTH": 5,
        "MODEL_TOKEN_COUNT": 4, "POSITION_START": 1, "POSITION_STOP": 3,
        "TOKENS_PER_ROW": 2, "PREFILTER": 3, "BATCH_SIZE": 1, "PASSES": 2,
        "DEVICE": "cpu",
    }.items():
        monkeypatch.setattr(collector, name, value)
    paths = {
        name: tmp_path / f"{name.lower()}.artifact"
        for name in ("AUTHORITY", "PAYLOAD", "RECEIPT", "FAILURE", "LOCK")
    }
    for name, path in paths.items():
        monkeypatch.setattr(collector, name, path)
    source = {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64}
    rows = torch.randint(0, 10, (2, 5), generator=torch.Generator().manual_seed(8))
    row_binding = {"disjointness_sha256": "c" * 64}
    checkpoint = collector.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="d" * 64,
        weights_sha256="e" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=10,
    )
    model = _TinyModel(width, gates)
    monkeypatch.setattr(collector, "source_closure", lambda: source)
    monkeypatch.setattr(collector, "validate_rows", lambda: (rows, row_binding))
    source_checks = 0
    row_checks = 0
    snapshot_checks = 0

    def verify_source(value):
        nonlocal source_checks
        source_checks += 1
        if terminal_drift == "source" and source_checks == 3:
            raise RuntimeError("injected terminal source drift")

    def verify_rows(binding, value):
        nonlocal row_checks
        row_checks += 1
        if row_checks == 3 and terminal_drift == "row":
            raise RuntimeError("injected terminal row drift")
        if row_checks == 3 and terminal_drift == "payload":
            with paths["PAYLOAD"].open("ab") as handle:
                handle.write(b"injected-drift")

    def validate_snapshot(**kwargs):
        nonlocal snapshot_checks
        snapshot_checks += 1
        if terminal_drift == "checkpoint" and snapshot_checks == 3:
            return collector.facade.CheckpointReceipt(
                revision="changed", snapshot="s", config_sha256="d" * 64,
                weights_sha256="f" * 64, weights_bytes=1,
                tokenizer_vocab=10, logit_vocab=10,
            )
        return checkpoint

    monkeypatch.setattr(collector, "verify_source_closure", verify_source)
    monkeypatch.setattr(collector, "verify_rows_unchanged", verify_rows)
    monkeypatch.setattr(collector.facade, "validate_snapshot", validate_snapshot)

    def load(**kwargs):
        assert paths["AUTHORITY"].exists()
        assert not paths["PAYLOAD"].exists() and not paths["RECEIPT"].exists()
        return model, checkpoint

    monkeypatch.setattr(collector.facade, "load_bilin18", load)
    return paths


def test_run_publishes_authority_before_forward_measures_calls_and_receipt_last(
    monkeypatch, tmp_path,
):
    paths = _configure_tiny_run(monkeypatch, tmp_path)
    result = collector.run()
    assert result["status"] == "fit_sufficient_statistics_complete_no_evaluation_opened"
    assert paths["RECEIPT"].exists() and not paths["FAILURE"].exists()
    assert not paths["LOCK"].exists()
    payload = torch.load(paths["PAYLOAD"], map_location="cpu", weights_only=True)
    ledger = payload["fit_call_ledger"]
    assert ledger["outer_returned"] == 4
    assert [ledger["attention_calls_by_site"][str(i)] for i in range(4)] == [4] * 4
    assert [ledger["mlp_calls_by_site"][str(i)] for i in range(4)] == [4, 4, 4, 0]
    assert ledger["explicit_typed_down_banks"] == 8


@pytest.mark.parametrize("drift", ("source", "row", "checkpoint", "payload"))
def test_terminal_drift_publishes_failure_and_never_receipt(monkeypatch, tmp_path, drift):
    paths = _configure_tiny_run(monkeypatch, tmp_path, terminal_drift=drift)
    with pytest.raises(RuntimeError):
        collector.run()
    assert paths["AUTHORITY"].exists()
    assert paths["PAYLOAD"].exists()
    assert paths["FAILURE"].exists()
    assert not paths["RECEIPT"].exists()
    assert not paths["LOCK"].exists()
