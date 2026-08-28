from __future__ import annotations

import json

import pytest
import torch

import compilation_mask_cut_rank_v1_gpu_adapter as inherited
import early_mlp_context_cross_v1_bilin18_backend as backend_module
import early_mlp_context_cross_v1_lifecycle as lifecycle
import early_mlp_context_cross_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as statistics
import run_early_mlp_context_cross_v1 as runner


CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
IMPLEMENTATION_SHA256 = "94b927b7d0576b29f2bfd4dbee851462598ae944d3c2f6d9405e702163fdbc4d"


def _source() -> inherited.SourceClosure:
    return inherited.SourceClosure(
        source_commit="a" * 40,
        path_sha256s=(("synthetic_launch.py", "b" * 64),),
    )


def _model_binding() -> inherited.ModelBinding:
    return inherited.ModelBinding(
        config_sha256=CONFIG_SHA256,
        weights_sha256=WEIGHTS_SHA256,
        implementation_sha256=IMPLEMENTATION_SHA256,
        model_realization_sha256=measurement.MODEL_REALIZATION_SHA256,
        component_tree_sha256=measurement.COMPONENT_TREE_SHA256,
    )


class FakeBackend:
    batch_size = 8
    source_paths = backend_module.SOURCE_PATHS

    def __init__(self, *, fail_at=None, after_prepare=None, after_verify=None):
        self.fail_at = fail_at
        self.after_prepare = after_prepare
        self.after_verify = after_verify
        self.calls = []
        self.closed = False
        self.bank = None

    def prepare(self, role_rows, requests):
        assert tuple(role_rows) == statistics.ROLE_NAMES
        programs = tuple(
            backend_module.ProgramDescriptor(
                ordinal=request.ordinal, request_sha256=request.sha256,
                installed_compiled_sites=backend_module._canonical_sites(request.sites),
                shared_program_sha256=measurement.SHARED_PROGRAM_SHA256,
            )
            for request in requests
        )
        self.bank = backend_module.PreparedBank(
            model=_model_binding(), programs=programs,
            shared_program_sha256=measurement.SHARED_PROGRAM_SHA256,
            evaluation_role_row_sha256s=tuple(
                (role, statistics.tensor_sha256(role_rows[role]))
                for role in statistics.ROLE_NAMES
            ),
        )
        if self.after_prepare is not None:
            self.after_prepare()
        return self.bank

    def verify_pre_outcome(self, bank):
        assert bank is self.bank
        if self.after_verify is not None:
            self.after_verify()
        return measurement.COMPONENT_TREE_SHA256, measurement.SHARED_PROGRAM_SHA256

    def execute_cell(self, role, request, rows, descriptor):
        self.calls.append((role, request.ordinal))
        if self.fail_at == (role, request.ordinal):
            raise RuntimeError("synthetic backend failure")
        batch_count = (len(rows) + self.batch_size - 1) // self.batch_size
        token_count = torch.full(
            (len(rows),), measurement.SCORED_TOKENS_PER_ROW, dtype=torch.long,
        )
        values = measurement.RowCellStatistics(
            top1_correct=torch.full(
                (len(rows),), request.ordinal % 193, dtype=torch.long,
            ),
            ce_sum=torch.full(
                (len(rows),), float(request.ordinal + 1), dtype=torch.float64,
            ),
            row_token_count=token_count,
        )
        ledger = backend_module.CellCallLedger(
            ordinal=request.ordinal, request_sha256=request.sha256,
            program_sha256=descriptor.sha256, row_count=len(rows),
            scored_token_count=len(rows) * measurement.SCORED_TOKENS_PER_ROW,
            batch_count=batch_count, outer_forward_count=batch_count,
            outer_returned_count=batch_count,
            native_module_calls=tuple(
                (site, batch_count) for site in inherited.ALL_NATIVE_SITES
            ),
            substitution_calls=tuple(
                (site, batch_count) for site in descriptor.installed_compiled_sites
            ),
        )
        return backend_module.BackendCellResult(
            statistics=values, call_ledger=ledger,
            component_tree_before_sha256=measurement.COMPONENT_TREE_SHA256,
            component_tree_after_sha256=measurement.COMPONENT_TREE_SHA256,
            shared_program_before_sha256=measurement.SHARED_PROGRAM_SHA256,
            shared_program_after_sha256=measurement.SHARED_PROGRAM_SHA256,
        )

    def close(self):
        self.closed = True
        return measurement.COMPONENT_TREE_SHA256


@pytest.fixture
def frozen_inputs(monkeypatch):
    monkeypatch.setattr(lifecycle, "committed_source_closure", _source)
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda _source: None)


def test_transaction_orders_both_roles_and_publishes_receipt_last(
    tmp_path, monkeypatch, frozen_inputs,
):
    paths = lifecycle.output_paths(tmp_path, "transaction_success")
    backend = FakeBackend()
    publication_order = []
    real_json = lifecycle.publish_json_create_only
    real_torch = lifecycle.publish_torch_create_only

    def record_json(path, value, lock):
        publication_order.append(path.name)
        return real_json(path, value, lock)

    def record_torch(path, value, lock):
        publication_order.append(path.name)
        return real_torch(path, value, lock)

    monkeypatch.setattr(lifecycle, "publish_json_create_only", record_json)
    monkeypatch.setattr(lifecycle, "publish_torch_create_only", record_torch)
    receipt = runner.run_transaction(backend=backend, paths=paths)
    assert backend.calls == [
        (role, ordinal) for role in statistics.ROLE_NAMES for ordinal in range(64)
    ]
    assert backend.closed
    assert publication_order == [
        paths.authority.name, paths.payload.name, paths.manifest.name,
        paths.receipt.name,
    ]
    assert receipt["status"] == "test_only_non_authoritative_receipt_last"
    assert receipt["authoritative_measurement"] is False
    assert paths.receipt.exists() and not paths.failure.exists() and not paths.lock.exists()
    payload = torch.load(paths.payload, map_location="cpu", weights_only=True)
    assert tuple(payload["roles"]) == statistics.ROLE_NAMES
    for role in statistics.ROLE_NAMES:
        assert set(payload["roles"][role]["stages"]) == {
            "discovery", "validation", "heldout",
        }


def test_mid_second_role_failure_publishes_no_partial_outcome(
    tmp_path, frozen_inputs,
):
    paths = lifecycle.output_paths(tmp_path, "transaction_failure")
    backend = FakeBackend(fail_at=("skip11000", 3))
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        runner.run_transaction(backend=backend, paths=paths)
    assert backend.closed
    assert paths.authority.exists() and paths.failure.exists()
    assert not paths.payload.exists() and not paths.manifest.exists() and not paths.receipt.exists()
    failure = json.loads(paths.failure.read_text())
    assert failure["phase"] == "measure_both_roles"
    assert failure["role"] == "skip11000" and failure["ordinal"] == 3
    assert not paths.lock.exists()


def test_payload_mutation_before_receipt_fails_closed(
    tmp_path, monkeypatch, frozen_inputs,
):
    paths = lifecycle.output_paths(tmp_path, "transaction_mutation")
    backend = FakeBackend()
    real_json = lifecycle.publish_json_create_only

    def mutate_after_manifest(path, value, lock):
        result = real_json(path, value, lock)
        if path == paths.manifest:
            paths.payload.write_bytes(paths.payload.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(lifecycle, "publish_json_create_only", mutate_after_manifest)
    with pytest.raises(Exception):
        runner.run_transaction(backend=backend, paths=paths)
    assert paths.authority.exists() and paths.payload.exists() and paths.manifest.exists()
    assert paths.failure.exists() and not paths.receipt.exists()


def test_existing_namespace_fails_before_lock_or_backend_use(tmp_path):
    paths = lifecycle.output_paths(tmp_path, "transaction_collision")
    paths.authority.write_text("spent\n")
    backend = FakeBackend()
    with pytest.raises(RuntimeError, match="spent"):
        runner.run_transaction(backend=backend, paths=paths)
    assert backend.calls == [] and not backend.closed and not paths.lock.exists()


def test_canonical_publication_rejects_a_source_path_spoofing_backend():
    with pytest.raises(RuntimeError, match="exact production backend"):
        runner._verify_backend_surface(FakeBackend(), require_production=True)


def test_output_created_during_prepare_prevents_false_pre_outcome_authority(
    tmp_path, frozen_inputs,
):
    paths = lifecycle.output_paths(tmp_path, "transaction_prepare_race")
    backend = FakeBackend(after_prepare=lambda: paths.payload.write_bytes(b"raced"))
    with pytest.raises(RuntimeError, match="appeared during pre-outcome"):
        runner.run_transaction(backend=backend, paths=paths)
    assert not paths.authority.exists() and not paths.receipt.exists()
    assert paths.payload.read_bytes() == b"raced"
    assert paths.failure.exists() and backend.closed and not paths.lock.exists()


def test_output_created_during_final_rehash_is_caught_by_immediate_guard(
    tmp_path, frozen_inputs,
):
    paths = lifecycle.output_paths(tmp_path, "transaction_verify_race")
    backend = FakeBackend(after_verify=lambda: paths.receipt.write_text("raced\n"))
    with pytest.raises(RuntimeError, match="immediately before authority"):
        runner.run_transaction(backend=backend, paths=paths)
    assert not paths.authority.exists() and not paths.payload.exists()
    assert paths.receipt.read_text() == "raced\n"
    assert paths.failure.exists() and backend.closed and not paths.lock.exists()
