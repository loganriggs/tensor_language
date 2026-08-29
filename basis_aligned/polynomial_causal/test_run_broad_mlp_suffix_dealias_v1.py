from __future__ import annotations

import json
from pathlib import Path
import re

import pytest
import torch

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_bilin18_backend as backend_module
import broad_mlp_suffix_dealias_v1_lifecycle as lifecycle
import broad_mlp_suffix_dealias_v1_measurements as measurement
import compilation_mask_cut_rank_v1_gpu_adapter as inherited
import early_mlp_context_cross_v1_statistics as parent_statistics
import run_broad_mlp_suffix_dealias_v1 as runner


CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
IMPLEMENTATION_SHA256 = "94b927b7d0576b29f2bfd4dbee851462598ae944d3c2f6d9405e702163fdbc4d"


def _source() -> inherited.SourceClosure:
    return inherited.SourceClosure(
        source_commit="a" * 40,
        path_sha256s=(("synthetic_broad_mlp_launch.py", "b" * 64),),
    )


class FakeBackend:
    batch_size = 8
    source_paths = backend_module.SOURCE_PATHS

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.calls = []
        self.closed = False
        self.bank = None

    def prepare(self, role_rows, requests):
        programs = tuple(
            backend_module.ProgramDescriptor(
                ordinal=request.ordinal,
                request_sha256=request.sha256,
                installed_compiled_sites=backend_module._canonical_sites(request.sites),
                shared_program_sha256=measurement.SHARED_PROGRAM_SHA256,
            ) for request in requests
        )
        binding = inherited.ModelBinding(
            config_sha256=CONFIG_SHA256,
            weights_sha256=WEIGHTS_SHA256,
            implementation_sha256=IMPLEMENTATION_SHA256,
            model_realization_sha256=measurement.MODEL_REALIZATION_SHA256,
            component_tree_sha256=measurement.COMPONENT_TREE_SHA256,
        )
        self.bank = backend_module.PreparedBank(
            model=binding,
            programs=programs,
            shared_program_sha256=measurement.SHARED_PROGRAM_SHA256,
            evaluation_role_row_sha256s=tuple(
                (role, parent_statistics.tensor_sha256(role_rows[role]))
                for role in assay.ROLE_NAMES
            ),
        )
        return self.bank

    def verify_pre_outcome(self, bank):
        assert bank is self.bank
        return measurement.COMPONENT_TREE_SHA256, measurement.SHARED_PROGRAM_SHA256

    def execute_cell(self, role, request, rows, descriptor):
        self.calls.append((role, request.ordinal))
        if self.fail_at == (role, request.ordinal):
            raise RuntimeError("synthetic broad-MLP failure")
        batch_count = (len(rows) + self.batch_size - 1) // self.batch_size
        values = measurement.RowCellStatistics(
            top1_correct=torch.full((len(rows),), request.ordinal, dtype=torch.long),
            ce_sum=torch.full((len(rows),), float(request.ordinal + 1), dtype=torch.float64),
            row_token_count=torch.full(
                (len(rows),), measurement.SCORED_TOKENS_PER_ROW, dtype=torch.long,
            ),
        )
        ledger = backend_module.CellCallLedger(
            ordinal=request.ordinal,
            request_sha256=request.sha256,
            program_sha256=descriptor.sha256,
            row_count=len(rows),
            scored_token_count=len(rows) * measurement.SCORED_TOKENS_PER_ROW,
            batch_count=batch_count,
            outer_forward_count=batch_count,
            outer_returned_count=batch_count,
            native_module_calls=tuple(
                (site, batch_count) for site in inherited.ALL_NATIVE_SITES
            ),
            substitution_calls=tuple(
                (site, batch_count) for site in descriptor.installed_compiled_sites
            ),
        )
        return backend_module.BackendCellResult(
            statistics=values,
            call_ledger=ledger,
            component_tree_before_sha256=measurement.COMPONENT_TREE_SHA256,
            component_tree_after_sha256=measurement.COMPONENT_TREE_SHA256,
            shared_program_before_sha256=measurement.SHARED_PROGRAM_SHA256,
            shared_program_after_sha256=measurement.SHARED_PROGRAM_SHA256,
        )

    def close(self):
        self.closed = True
        return measurement.COMPONENT_TREE_SHA256


@pytest.fixture
def frozen_source(monkeypatch):
    monkeypatch.setattr(lifecycle, "committed_source_closure", _source)
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda _source: None)


def test_transaction_runs_exact_sixteen_role_cells_and_receipt_last(
    tmp_path, frozen_source,
):
    paths = lifecycle.output_paths(tmp_path, "broad_mlp_test_success")
    backend = FakeBackend()
    receipt = runner.run_transaction(backend=backend, paths=paths)
    assert backend.calls == [
        (role, ordinal) for role in assay.ROLE_NAMES
        for ordinal in range(assay.CELL_COUNT)
    ]
    assert backend.closed and receipt["status"] == (
        "test_only_non_authoritative_receipt_last"
    )
    assert paths.receipt.exists() and not paths.failure.exists() and not paths.lock.exists()
    payload = torch.load(paths.payload, map_location="cpu", weights_only=True)
    for role in assay.ROLE_NAMES:
        assert payload["roles"][role]["statistics"]["ce_sum"].shape == (
            measurement.ROLE_DOCUMENT_COUNTS[role], assay.CELL_COUNT,
        )


def test_failure_has_no_terminal_payload_or_receipt(tmp_path, frozen_source):
    paths = lifecycle.output_paths(tmp_path, "broad_mlp_test_failure")
    backend = FakeBackend(fail_at=("skip11000", 2))
    with pytest.raises(RuntimeError, match="synthetic broad-MLP failure"):
        runner.run_transaction(backend=backend, paths=paths)
    failure = json.loads(paths.failure.read_text())
    assert failure["phase"] == "measure_both_roles"
    assert failure["role"] == "skip11000" and failure["ordinal"] == 2
    assert paths.authority.exists() and not paths.payload.exists()
    assert not paths.manifest.exists() and not paths.receipt.exists() and not paths.lock.exists()


def test_source_mutation_after_first_role_fails_before_second_role_or_payload(
    tmp_path, frozen_source, monkeypatch,
):
    paths = lifecycle.output_paths(tmp_path, "broad_mlp_source_mutation")
    backend = FakeBackend()
    calls = 0

    def verify(_source):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic source mutation after role")

    monkeypatch.setattr(lifecycle, "verify_source_closure", verify)
    with pytest.raises(RuntimeError, match="synthetic source mutation"):
        runner.run_transaction(backend=backend, paths=paths)
    assert backend.calls == [
        ("skip7000", ordinal) for ordinal in range(assay.CELL_COUNT)
    ]
    assert paths.authority.exists() and paths.failure.exists()
    assert not paths.payload.exists() and not paths.manifest.exists() and not paths.receipt.exists()


def test_new_executables_have_no_inherited_predicate_surface():
    """The LESSONS-63 stale predicate/docstring hazard is absent by construction."""

    root = Path(__file__).resolve().parent
    for name in (
        "broad_mlp_suffix_dealias_v1_bilin18_backend.py",
        "run_broad_mlp_suffix_dealias_v1.py",
        "score_broad_mlp_suffix_dealias_v1.py",
    ):
        source = (root / name).read_text(encoding="utf-8")
        assert re.search(r"#\s+pred_[a-d]\b", source) is None
        assert re.search(r"['\"]pred_[a-d]_[a-z0-9_]+['\"]\s*:", source) is None
