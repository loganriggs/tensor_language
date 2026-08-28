from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import compilation_mask_cut_rank_v1_measurements as measurement


def _hash(label: str) -> str:
    return adapter._logical_sha256(label)


def test_path_cli_binds_backend_protocol_classes_to_one_module_identity(tmp_path):
    probe = tmp_path / "script_identity_probe.py"
    probe.write_text(
        """import sys
import compilation_mask_cut_rank_v1_gpu_adapter as canonical

def create_backend():
    script = sys.modules[\"__main__\"]
    if canonical is not script:
        raise RuntimeError(\"adapter module identity duplicated\")
    if canonical.PreparedProgramBank is not script.PreparedProgramBank:
        raise RuntimeError(\"prepared-bank class identity duplicated\")
    print(\"SCRIPT_PROTOCOL_IDENTITY_OK\", flush=True)
    raise RuntimeError(\"intentional stop before source, row, model, or CUDA work\")
""",
        encoding="utf-8",
    )
    module_directory = str(Path(adapter.__file__).resolve().parent)
    python_path = os.pathsep.join(filter(None, (
        str(tmp_path), module_directory, os.environ.get("PYTHONPATH", ""),
    )))
    completed = subprocess.run(
        (
            sys.executable, str(Path(adapter.__file__).resolve()),
            "--backend-module", "script_identity_probe",
        ),
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "SCRIPT_PROTOCOL_IDENTITY_OK" in completed.stdout
    assert "intentional stop before source, row, model, or CUDA work" in completed.stderr
    assert "identity duplicated" not in completed.stderr


def _make_rows(tmp_path: Path, *, authorized: bool = True):
    rows = torch.arange(6 * 513, dtype=torch.long).reshape(6, 513) % 50_257
    cache = tmp_path / "rows.pt"
    torch.save(rows, cache)
    provenance = [
        {"document_id": document, "dataset_document_index": index,
         "chunk_id": chunk, "token_start": chunk * 513}
        for index, (document, chunk) in enumerate((
            ("doc-a", 0), ("doc-a", 1), ("doc-b", 0),
            ("doc-c", 0), ("doc-c", 1), ("doc-c", 2),
        ))
    ]
    receipt = {
        "authority": "pinned_local_ordered_manifest",
        "authorized_for_scored_experiments": authorized,
        "ordered_manifest_local_parquet_identity_gate": {"passed": True},
        "entries": {
            "synthetic": {
                "n": 6, "skip": 0, "shape": [6, 513], "dtype": "torch.int64",
                "tensor_raw_sha256": adapter.raw_tensor_sha256(rows),
                "cache_path": str(cache.resolve()),
            },
        },
        "document_provenance": {"schema_version": 1, "sets": {"synthetic": provenance}},
    }
    receipt_path = tmp_path / "row_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, rows, provenance


def _model_binding() -> adapter.ModelBinding:
    config = _hash("config")
    weights = _hash("weights")
    implementation = _hash("implementation")
    return adapter.ModelBinding(
        config_sha256=config, weights_sha256=weights,
        implementation_sha256=implementation,
        model_realization_sha256=adapter._logical_sha256({
            "config_sha256": config, "weights_sha256": weights,
            "implementation_sha256": implementation,
        }),
        component_tree_sha256=_hash("component tree"),
    )


def _program(ordinal: int, *, gains=()):
    request = measurement.REQUESTS[ordinal]
    return adapter.ProgramDescriptor(
        ordinal=ordinal,
        request_sha256=request.sha256,
        installed_compiled_sites=adapter._canonical_sites((
            *request.always_compiled_sites, *request.additional_sites,
        )),
        live_attention_gain_sites=tuple(gains),
        shared_program_state_sha256=_hash("shared program"),
        cell_program_state_sha256=_hash(f"cell program {ordinal}"),
        program_source_sha256=_hash("program source"),
    )


class FakeBackend:
    batch_size = 4
    source_paths = ("fake_backend.py",)

    def __init__(self, *, fail_at=None, mutate_at=None, close_hash=None):
        self.fail_at = fail_at
        self.mutate_at = mutate_at
        self.close_hash = close_hash
        self.executed = []
        self.close_calls = 0
        self.model = _model_binding()

    def prepare(self, rows, requests):
        assert tuple(requests) == measurement.REQUESTS
        assert tuple(rows.shape) == (6, 257)
        return adapter.PreparedProgramBank(
            model=self.model,
            programs=tuple(_program(ordinal) for ordinal in range(64)),
        )

    def execute_cell(self, request, rows, program):
        if request.ordinal == self.fail_at:
            raise RuntimeError("synthetic backend failure with no outcome value")
        self.executed.append(request.ordinal)
        row_count = len(rows)
        batch_count = 2
        token_count = torch.full(
            (row_count,), adapter.SCORE_STOP - adapter.SCORE_START, dtype=torch.long,
        )
        correct = token_count - (request.ordinal % 3)
        ce = token_count.to(torch.float64) * (0.25 + request.ordinal / 1000.0)
        statistics = measurement.RowCellSufficientStatistics(
            top1_correct=correct, ce_sum=ce, row_token_count=token_count,
        )
        ledger = adapter.CellCallLedger(
            ordinal=request.ordinal,
            request_sha256=request.sha256,
            program_realization_sha256=program.sha256,
            execution_mode=adapter.EXECUTION_MODE,
            row_count=row_count,
            scored_token_count=row_count * (adapter.SCORE_STOP - adapter.SCORE_START),
            batch_count=batch_count,
            outer_forward_count=batch_count,
            outer_returned_count=batch_count,
            native_module_calls=tuple(
                (site, batch_count) for site in adapter.ALL_NATIVE_SITES
            ),
            substitution_calls=tuple(
                (site, batch_count) for site in program.installed_compiled_sites
            ),
            live_attention_gain_calls=tuple(
                (site, batch_count) for site in program.live_attention_gain_sites
            ),
            fitter_calls=0,
            retained_logits=0,
        )
        after = (
            _hash("mutated component tree")
            if request.ordinal == self.mutate_at else self.model.component_tree_sha256
        )
        return adapter.BackendCellResult(
            statistics=statistics, call_ledger=ledger,
            component_tree_before_sha256=self.model.component_tree_sha256,
            component_tree_after_sha256=after,
        )

    def close(self):
        self.close_calls += 1
        return self.close_hash or self.model.component_tree_sha256


def _fake_source_closure():
    return adapter.SourceClosure(
        source_commit="1" * 40,
        path_sha256s=(("fake.py", _hash("fake source")),),
    )


def test_row_wave_binds_exact_scored_support_and_document_clusters(tmp_path):
    receipt, rows, provenance = _make_rows(tmp_path)
    wave = adapter.load_row_wave(receipt, "synthetic")
    assert wave.row_count == 6 and wave.document_count == 3
    assert tuple(wave.clone_rows().shape) == (6, 257)
    mapping, counts = wave.clone_mapping_and_counts()
    assert torch.equal(mapping, torch.tensor([0, 0, 1, 2, 2, 2]))
    assert torch.equal(counts, torch.full((6,), 192, dtype=torch.long))
    expected_targets = rows[:, 65:257].contiguous()
    expected_support = adapter._logical_sha256({
        "ordered_row_identity_sha256": wave.ordered_row_identity_sha256,
        "input_position_half_open": [0, 256],
        "scored_logit_position_half_open": [64, 256],
        "target_position_half_open": [65, 257],
        "target_tensor_sha256": measurement.tensor_sha256(expected_targets),
        "row_token_count_sha256": measurement.tensor_sha256(counts),
    })
    assert wave.common_support_sha256 == expected_support
    rows.zero_()
    provenance[0]["document_id"] = "changed"
    assert wave.sha256 == wave.sha256
    escaped = wave.clone_rows()
    escaped.zero_()
    assert bool((wave.clone_rows() != 0).any())


def test_row_loader_rejects_unlicensed_changed_or_ambiguous_receipts(tmp_path):
    receipt, _, _ = _make_rows(tmp_path, authorized=False)
    with pytest.raises(RuntimeError, match="not licensed"):
        adapter.load_row_wave(receipt, "synthetic")
    receipt, _, _ = _make_rows(tmp_path)
    content = json.loads(receipt.read_text())
    content["entries"]["synthetic"]["tensor_raw_sha256"] = _hash("wrong rows")
    receipt.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match="differ"):
        adapter.load_row_wave(receipt, "synthetic")
    receipt, _, _ = _make_rows(tmp_path)
    content = json.loads(receipt.read_text())
    content["document_provenance"]["sets"]["synthetic"][0]["extra"] = 1
    receipt.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match="exact ordered schema"):
        adapter.load_row_wave(receipt, "synthetic")


def test_program_bank_rejects_wrong_masks_gains_reordering_and_aliases():
    request = measurement.REQUESTS[1]
    with pytest.raises(ValueError, match="canonical request"):
        replace(_program(1), installed_compiled_sites=())
    compiled_attention = next(
        site for site in _program(1).installed_compiled_sites if site[0] == "attn"
    )
    with pytest.raises(ValueError, match="canonical request"):
        _program(1, gains=(compiled_attention,))
    programs = tuple(_program(ordinal) for ordinal in range(64))
    bank = adapter.PreparedProgramBank(model=_model_binding(), programs=programs)
    assert bank.sha256 == measurement.program_bank_sha256(
        bank.program_realization_sha256s
    )
    with pytest.raises(ValueError, match="reordered"):
        adapter.PreparedProgramBank(
            model=_model_binding(), programs=(programs[1], programs[0], *programs[2:]),
        )
    with pytest.raises(ValueError, match="aliased"):
        adapter.PreparedProgramBank(
            model=_model_binding(), programs=tuple(_program(0) for _ in range(64)),
        )
    assert request.sha256 == programs[1].request_sha256


def test_call_ledger_requires_exact_native_substitution_gain_counts_and_no_fitter():
    backend = FakeBackend()
    rows = torch.zeros((6, 257), dtype=torch.long)
    request = measurement.REQUESTS[0]
    program = _program(0)
    result = backend.execute_cell(request, rows, program)
    result.call_ledger.validate(program, row_count=6, batch_count=2)
    with pytest.raises(RuntimeError, match="forbidden"):
        replace(result.call_ledger, fitter_calls=1).validate(
            program, row_count=6, batch_count=2,
        )
    with pytest.raises(ValueError, match="site counts"):
        replace(
            result.call_ledger,
            substitution_calls=result.call_ledger.substitution_calls[:-1],
        ).validate(program, row_count=6, batch_count=2)
    with pytest.raises(ValueError, match="site counts"):
        replace(
            result.call_ledger,
            native_module_calls=tuple(
                (site, 1) for site in adapter.ALL_NATIVE_SITES
            ),
        ).validate(program, row_count=6, batch_count=2)


def test_committed_source_closure_rejects_dirty_bytes_and_path_escape(tmp_path):
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "config", "user.email", "x@y"), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "config", "user.name", "x"), check=True)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    subprocess.run(("git", "-C", str(tmp_path), "add", "source.py"), check=True)
    subprocess.run(("git", "-C", str(tmp_path), "commit", "-qm", "source"), check=True)
    closure = adapter.committed_source_closure(tmp_path, ("source.py",))
    assert closure.path_sha256s == (("source.py", adapter.file_sha256(source)),)
    source.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="differ"):
        adapter.committed_source_closure(tmp_path, ("source.py",))
    with pytest.raises(ValueError, match="escapes"):
        adapter.committed_source_closure(tmp_path, ("../source.py",))


def _run(tmp_path, monkeypatch, backend):
    receipt, _, _ = _make_rows(tmp_path)
    monkeypatch.setattr(adapter, "committed_source_closure", lambda repo, paths: _fake_source_closure())
    paths = adapter.output_paths(tmp_path, "assay")
    result = adapter.run_transaction(
        backend=backend, row_receipt=receipt, row_role="synthetic",
        repo=tmp_path, paths=paths,
    )
    return paths, result


def test_complete_transaction_freezes_authority_then_publishes_sealed_payload_receipt(
    tmp_path, monkeypatch,
):
    backend = FakeBackend()
    paths, result = _run(tmp_path, monkeypatch, backend)
    assert backend.executed == list(range(64))
    assert backend.close_calls == 1
    assert paths.authority.is_file() and paths.payload.is_file() and paths.receipt.is_file()
    assert not paths.failure.exists() and not paths.lock.exists()
    authority = json.loads(paths.authority.read_text())
    assert authority["status"] == "frozen_before_any_measurement_cell"
    assert authority["authorized_for_final_role"] is False
    assert len(authority["program_descriptors"]) == 64
    assert result == json.loads(paths.receipt.read_text())
    assert result["authorized_for_final_role"] is False
    assert result["payload_file_sha256"] == adapter.file_sha256(paths.payload)
    payload = torch.load(paths.payload, map_location="cpu", weights_only=True)
    assert tuple(payload["top1_correct"].shape) == (3, 64)
    assert tuple(payload["ce_sum"].shape) == (3, 64)
    assert payload["ce_sum"].dtype == torch.float64
    assert torch.equal(payload["document_row_count"], torch.tensor([2, 1, 3]))
    with pytest.raises(RuntimeError, match="not pristine"):
        adapter.run_transaction(
            backend=FakeBackend(), row_receipt=tmp_path / "row_receipt.json",
            row_role="synthetic", repo=tmp_path, paths=paths,
        )


@pytest.mark.parametrize("failure_kind", ("cell", "component", "close"))
def test_failures_after_authority_publish_no_receipt_or_payload(
    tmp_path, monkeypatch, failure_kind,
):
    receipt, _, _ = _make_rows(tmp_path)
    monkeypatch.setattr(adapter, "committed_source_closure", lambda repo, paths: _fake_source_closure())
    if failure_kind == "cell":
        backend = FakeBackend(fail_at=3)
    elif failure_kind == "component":
        backend = FakeBackend(mutate_at=3)
    else:
        backend = FakeBackend(close_hash=_hash("wrong close tree"))
    paths = adapter.output_paths(tmp_path, "failure_assay")
    with pytest.raises(RuntimeError):
        adapter.run_transaction(
            backend=backend, row_receipt=receipt, row_role="synthetic",
            repo=tmp_path, paths=paths,
        )
    assert paths.authority.is_file() and paths.failure.is_file()
    assert not paths.payload.exists() and not paths.receipt.exists() and not paths.lock.exists()
    failure = json.loads(paths.failure.read_text())
    assert failure["authorized_for_final_role"] is False
    assert set(failure) == {
        "schema_version", "status", "authorized_for_final_role", "phase", "ordinal",
        "exception_type", "authority_file_sha256",
    }
    assert failure["authority_file_sha256"] == adapter.file_sha256(paths.authority)


def test_lock_conflict_and_lock_loss_never_overwrite_or_publish(tmp_path, monkeypatch):
    receipt, _, _ = _make_rows(tmp_path)
    paths = adapter.output_paths(tmp_path, "locked")
    paths.lock.write_text("someone-else\n")
    with pytest.raises(RuntimeError, match="not pristine"):
        adapter.run_transaction(
            backend=FakeBackend(), row_receipt=receipt, row_role="synthetic",
            repo=tmp_path, paths=paths,
        )
    assert paths.lock.read_text() == "someone-else\n"
    assert not paths.authority.exists() and not paths.payload.exists() and not paths.receipt.exists()

    lock = adapter.RunLock(tmp_path / ".ownership.lock")
    lock.acquire()
    lock.path.unlink()
    lock.path.write_text("replacement\n")
    with pytest.raises(RuntimeError, match="ownership"):
        adapter._publish_bytes_create_only(tmp_path / "forbidden", b"x", lock)
    assert not (tmp_path / "forbidden").exists()


def test_corrupted_installed_payload_cannot_receive_last_written_receipt(tmp_path, monkeypatch):
    receipt, _, _ = _make_rows(tmp_path)
    monkeypatch.setattr(adapter, "committed_source_closure", lambda repo, paths: _fake_source_closure())
    paths = adapter.output_paths(tmp_path, "corrupt_payload")
    original_publish = adapter._publish_bytes_create_only

    def corrupt_payload(path, content, lock):
        original_publish(path, content, lock)
        if path == paths.payload:
            value = torch.load(path, map_location="cpu", weights_only=True)
            value["ce_sum"][0, 0] += 1.0
            torch.save(value, path)

    monkeypatch.setattr(adapter, "_publish_bytes_create_only", corrupt_payload)
    with pytest.raises(RuntimeError, match="payload tensor changed"):
        adapter.run_transaction(
            backend=FakeBackend(), row_receipt=receipt, row_role="synthetic",
            repo=tmp_path, paths=paths,
        )
    assert paths.authority.is_file() and paths.payload.is_file() and paths.failure.is_file()
    assert not paths.receipt.exists() and not paths.lock.exists()
