#!/usr/bin/env python3
"""Thin, fitter-free execution transaction for the cut-rank measurement grid.

The module owns source/row/model/program currency, the canonical 64-cell lifecycle,
typed call-ledger validation, and receipt-last publication.  A separately committed
backend owns the actual model and compiled program.  No rank fitting or scientific
decision code is imported or called here.

Importing this module performs no file, row, model, or CUDA work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any, Protocol, Sequence

import torch

import compilation_mask_cut_rank_v1_measurements as measurement


SCHEMA_VERSION = 1
SCORE_START = 64
SCORE_STOP = 256
INPUT_STOP = 256
TARGET_START = 65
TARGET_STOP = 257
ROW_ROLE = "n192_skip7000"
EXECUTION_MODE = "native_module_executes_then_exact_output_substitution"
GAIN_POLICY = "identity_gains_no_mask_specific_refitting"
ALL_NATIVE_SITES = tuple(
    (kind, layer) for layer in range(18) for kind in ("attn", "mlp")
)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROW_RECEIPT = (
    HERE.parent / "bilinear_quotient" / ".rowcache" / "fineweb_oracle_v2_receipt.json"
)
DEFAULT_OUTPUT_DIRECTORY = HERE
DEFAULT_NAMESPACE = "compilation_mask_cut_rank_v1_measurement_wave_v1"
CORE_SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/COMPILATION_MASK_CUT_RANK_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1.py",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_measurements.py",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_gpu_adapter.py",
)


def _bind_script_module_to_canonical_name() -> None:
    """Make path execution and backend imports share every protocol class.

    ``python path/to/this_file.py`` initially installs this module only as
    ``__main__``.  A dynamically loaded backend imports the filename stem, and
    without this alias Python executes a second copy whose dataclass identities
    are distinct even though their source bytes are identical.
    """

    canonical_name = Path(__file__).stem
    current = sys.modules.get(__name__)
    canonical = sys.modules.get(canonical_name)
    if current is None:
        raise RuntimeError("adapter script module is absent from sys.modules")
    if canonical is not None and canonical is not current:
        raise RuntimeError("adapter was loaded under conflicting module identities")
    sys.modules[canonical_name] = current


if __name__ == "__main__":
    _bind_script_module_to_canonical_name()


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("raw tensor hash requires a tensor")
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _git(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repo), *arguments), check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError("git source-closure operation failed")
    return completed.stdout


@dataclass(frozen=True, slots=True)
class SourceClosure:
    source_commit: str
    path_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ) or not self.path_sha256s or tuple(path for path, _ in self.path_sha256s) != tuple(
            sorted(path for path, _ in self.path_sha256s)
        ) or len({path for path, _ in self.path_sha256s}) != len(self.path_sha256s) or any(
            not path or Path(path).is_absolute() or ".." in Path(path).parts or not _sha256_text(digest)
            for path, digest in self.path_sha256s
        ):
            raise ValueError("source closure is malformed")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


def committed_source_closure(repo: Path, paths: Sequence[str]) -> SourceClosure:
    """Bind exact working bytes to the current commit for a closed path set."""

    root = repo.resolve()
    normalized = tuple(sorted(set(paths)))
    if not normalized or len(normalized) != len(tuple(paths)):
        raise ValueError("source closure paths must be a nonempty unique sequence")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    bindings = []
    for relative in normalized:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative:
            raise ValueError("source closure path escapes the repository")
        disk = root / path
        if not disk.is_file():
            raise RuntimeError("source closure file is absent")
        committed = _git(root, "show", f"{commit}:{relative}")
        current = disk.read_bytes()
        if current != committed:
            raise RuntimeError("source closure working bytes differ from the launch commit")
        bindings.append((relative, hashlib.sha256(committed).hexdigest()))
    return SourceClosure(source_commit=commit, path_sha256s=tuple(bindings))


def _exact_provenance_record(value: Any) -> dict[str, Any]:
    keys = {"document_id", "dataset_document_index", "chunk_id", "token_start"}
    if not isinstance(value, dict) or set(value) != keys or not isinstance(
        value["document_id"], str
    ) or not value["document_id"] or any(
        type(value[key]) is not int or value[key] < 0
        for key in ("dataset_document_index", "chunk_id", "token_start")
    ):
        raise RuntimeError("row provenance record is not the exact ordered schema")
    return dict(value)


class RowWave:
    """Sealed canonical rows plus every identity needed by measurement authority."""

    __slots__ = (
        "_expected_sha256", "_ordered_document_ids", "_provenance", "_row_to_document",
        "_rows", "_sealed", "_source_receipt_sha256", "_token_count",
    )

    def __init__(
        self, *, rows: torch.Tensor, provenance: Sequence[dict[str, Any]],
        source_receipt_sha256: str,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not torch.is_tensor(rows) or tuple(rows.shape[1:]) != (TARGET_STOP,) or (
            rows.ndim != 2 or rows.dtype != torch.long or rows.device.type != "cpu"
        ) or not rows.is_contiguous() or rows.requires_grad or len(rows) == 0 or not (
            _sha256_text(source_receipt_sha256)
        ) or int(rows.min()) < 0 or int(rows.max()) >= 50_257 or len(provenance) != len(rows):
            raise RuntimeError("row wave tensor/source schema is malformed")
        checked = tuple(_exact_provenance_record(value) for value in provenance)
        documents: list[str] = []
        document_index: dict[str, int] = {}
        mapping = []
        for record in checked:
            document_id = record["document_id"]
            if document_id not in document_index:
                document_index[document_id] = len(documents)
                documents.append(document_id)
            mapping.append(document_index[document_id])
        self._rows = rows.detach().clone().contiguous()
        self._provenance = checked
        self._ordered_document_ids = tuple(documents)
        self._row_to_document = torch.tensor(mapping, dtype=torch.long)
        self._token_count = torch.full(
            (len(rows),), SCORE_STOP - SCORE_START, dtype=torch.long,
        )
        self._source_receipt_sha256 = source_receipt_sha256
        self._expected_sha256 = self._compute_sha256()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("row wave is sealed")
        object.__setattr__(self, name, value)

    def _compute_sha256(self) -> str:
        return _logical_sha256({
            "rows_sha256": measurement.tensor_sha256(self._rows),
            "provenance_sha256": _logical_sha256(self._provenance),
            "row_to_document_sha256": measurement.tensor_sha256(self._row_to_document),
            "ordered_document_ids_sha256": _logical_sha256(self._ordered_document_ids),
            "token_count_sha256": measurement.tensor_sha256(self._token_count),
            "source_receipt_sha256": self._source_receipt_sha256,
        })

    def _require_pristine(self) -> None:
        if self._compute_sha256() != self._expected_sha256:
            raise RuntimeError("row wave mutated after validation")

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def document_count(self) -> int:
        return len(self._ordered_document_ids)

    @property
    def source_receipt_sha256(self) -> str:
        return self._source_receipt_sha256

    @property
    def row_tensor_sha256(self) -> str:
        self._require_pristine()
        return measurement.tensor_sha256(self._rows)

    @property
    def row_provenance_sha256(self) -> str:
        return _logical_sha256(self._provenance)

    @property
    def ordered_row_identity_sha256(self) -> str:
        return _logical_sha256(tuple(
            {"ordinal": ordinal, **record} for ordinal, record in enumerate(self._provenance)
        ))

    @property
    def ordered_document_ids_sha256(self) -> str:
        return _logical_sha256(self._ordered_document_ids)

    @property
    def common_support_sha256(self) -> str:
        self._require_pristine()
        targets = self._rows[:, TARGET_START:TARGET_STOP].contiguous()
        return _logical_sha256({
            "ordered_row_identity_sha256": self.ordered_row_identity_sha256,
            "input_position_half_open": [0, INPUT_STOP],
            "scored_logit_position_half_open": [SCORE_START, SCORE_STOP],
            "target_position_half_open": [TARGET_START, TARGET_STOP],
            "target_tensor_sha256": measurement.tensor_sha256(targets),
            "row_token_count_sha256": measurement.tensor_sha256(self._token_count),
        })

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return self._expected_sha256

    def clone_rows(self) -> torch.Tensor:
        self._require_pristine()
        return self._rows.clone()

    def clone_mapping_and_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_pristine()
        return self._row_to_document.clone(), self._token_count.clone()

    def receipt(self) -> dict[str, Any]:
        self._require_pristine()
        return {
            "row_wave_sha256": self.sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "row_tensor_sha256": self.row_tensor_sha256,
            "row_provenance_sha256": self.row_provenance_sha256,
            "ordered_row_identity_sha256": self.ordered_row_identity_sha256,
            "ordered_row_to_document_sha256": measurement.tensor_sha256(
                self._row_to_document
            ),
            "ordered_document_ids_sha256": self.ordered_document_ids_sha256,
            "row_token_count_sha256": measurement.tensor_sha256(self._token_count),
            "common_support_sha256": self.common_support_sha256,
            "row_count": self.row_count,
            "document_count": self.document_count,
            "total_scored_token_count": int(self._token_count.sum()),
            "support": {
                "input_position_half_open": [0, INPUT_STOP],
                "scored_logit_position_half_open": [SCORE_START, SCORE_STOP],
                "target_position_half_open": [TARGET_START, TARGET_STOP],
            },
        }


def load_row_wave(receipt_path: Path, role: str = ROW_ROLE) -> RowWave:
    """Load only one licensed role and bind its exact 257-token scored prefix."""

    path = receipt_path.resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("authority") != "pinned_local_ordered_manifest" or receipt.get(
        "authorized_for_scored_experiments"
    ) is not True or receipt.get("ordered_manifest_local_parquet_identity_gate", {}).get(
        "passed"
    ) is not True or receipt.get("document_provenance", {}).get("schema_version") != 1:
        raise RuntimeError("row receipt is not licensed ordered-manifest authority")
    entry = receipt.get("entries", {}).get(role)
    provenance = receipt.get("document_provenance", {}).get("sets", {}).get(role)
    if not isinstance(entry, dict) or not isinstance(provenance, list) or set(entry) != {
        "n", "skip", "shape", "dtype", "tensor_raw_sha256", "cache_path",
    } or type(entry["n"]) is not int or entry["n"] <= 0 or entry["dtype"] != (
        "torch.int64"
    ) or entry["shape"] != [entry["n"], 513] or not _sha256_text(
        entry["tensor_raw_sha256"]
    ) or len(provenance) != entry["n"]:
        raise RuntimeError("row receipt role schema is incomplete or changed")
    cache_path = Path(entry["cache_path"]).resolve()
    if not cache_path.is_file():
        raise RuntimeError("row cache named by receipt is absent")
    loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
    rows = loaded["rows"] if isinstance(loaded, dict) and set(loaded) == {"rows"} else loaded
    if not torch.is_tensor(rows) or tuple(rows.shape) != tuple(entry["shape"]) or (
        rows.dtype != torch.long
    ) or raw_tensor_sha256(rows) != entry["tensor_raw_sha256"]:
        raise RuntimeError("row cache bytes differ from the licensed receipt")
    return RowWave(
        rows=rows[:, :TARGET_STOP].contiguous(), provenance=provenance,
        source_receipt_sha256=file_sha256(path),
    )


@dataclass(frozen=True, slots=True)
class ModelBinding:
    config_sha256: str
    weights_sha256: str
    implementation_sha256: str
    model_realization_sha256: str
    component_tree_sha256: str

    def __post_init__(self) -> None:
        values = (
            self.config_sha256, self.weights_sha256, self.implementation_sha256,
            self.model_realization_sha256, self.component_tree_sha256,
        )
        if any(not _sha256_text(value) for value in values) or self.model_realization_sha256 != (
            _logical_sha256({
                "config_sha256": self.config_sha256,
                "weights_sha256": self.weights_sha256,
                "implementation_sha256": self.implementation_sha256,
            })
        ):
            raise ValueError("model binding is malformed or not content-derived")


def _canonical_sites(sites: Sequence[measurement.cut.Site]) -> tuple[measurement.cut.Site, ...]:
    selected = set(sites)
    return tuple(site for site in ALL_NATIVE_SITES if site in selected)


@dataclass(frozen=True, slots=True)
class ProgramDescriptor:
    ordinal: int
    request_sha256: str
    installed_compiled_sites: tuple[measurement.cut.Site, ...]
    live_attention_gain_sites: tuple[measurement.cut.Site, ...]
    shared_program_state_sha256: str
    cell_program_state_sha256: str
    program_source_sha256: str
    execution_mode: str = EXECUTION_MODE
    gain_policy: str = GAIN_POLICY

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or not 0 <= self.ordinal < 64:
            raise ValueError("program descriptor ordinal is malformed")
        request = measurement.REQUESTS[self.ordinal]
        expected_sites = _canonical_sites((
            *request.always_compiled_sites, *request.additional_sites,
        ))
        gains = self.live_attention_gain_sites
        if self.request_sha256 != request.sha256 or self.installed_compiled_sites != (
            expected_sites
        ) or gains != () or self.execution_mode != EXECUTION_MODE or self.gain_policy != (
            GAIN_POLICY
        ) or any(
            not _sha256_text(value) for value in (
                self.shared_program_state_sha256, self.cell_program_state_sha256,
                self.program_source_sha256,
            )
        ):
            raise ValueError("program descriptor differs from its canonical request")

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class PreparedProgramBank:
    model: ModelBinding
    programs: tuple[ProgramDescriptor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelBinding) or not isinstance(
            self.programs, tuple
        ) or len(self.programs) != 64 or tuple(
            value.ordinal for value in self.programs
        ) != tuple(range(64)) or len({value.sha256 for value in self.programs}) != 64:
            raise ValueError("prepared program bank is incomplete, reordered, or aliased")

    @property
    def program_realization_sha256s(self) -> tuple[str, ...]:
        return tuple(value.sha256 for value in self.programs)

    @property
    def sha256(self) -> str:
        return measurement.program_bank_sha256(self.program_realization_sha256s)


def _site_count_vector(
    value: Any, expected_sites: Sequence[measurement.cut.Site], expected_count: int,
) -> tuple[tuple[measurement.cut.Site, int], ...]:
    expected = tuple((site, expected_count) for site in expected_sites)
    if not isinstance(value, tuple) or value != expected or any(
        type(count) is not int or count < 0 for _, count in value
    ):
        raise ValueError("call-ledger site counts differ from exact execution algebra")
    return value


@dataclass(frozen=True, slots=True)
class CellCallLedger:
    ordinal: int
    request_sha256: str
    program_realization_sha256: str
    execution_mode: str
    row_count: int
    scored_token_count: int
    batch_count: int
    outer_forward_count: int
    outer_returned_count: int
    native_module_calls: tuple[tuple[measurement.cut.Site, int], ...]
    substitution_calls: tuple[tuple[measurement.cut.Site, int], ...]
    live_attention_gain_calls: tuple[tuple[measurement.cut.Site, int], ...]
    fitter_calls: int
    retained_logits: int

    def validate(self, descriptor: ProgramDescriptor, *, row_count: int, batch_count: int) -> None:
        if self.ordinal != descriptor.ordinal or self.request_sha256 != (
            descriptor.request_sha256
        ) or self.program_realization_sha256 != descriptor.sha256 or self.execution_mode != (
            EXECUTION_MODE
        ) or self.row_count != row_count or self.scored_token_count != row_count * (
            SCORE_STOP - SCORE_START
        ) or self.batch_count != batch_count or self.outer_forward_count != batch_count or (
            self.outer_returned_count != batch_count
        ) or self.fitter_calls != 0 or self.retained_logits != 0 or any(
            type(value) is not int for value in (
                self.row_count, self.scored_token_count, self.batch_count,
                self.outer_forward_count, self.outer_returned_count,
                self.fitter_calls, self.retained_logits,
            )
        ):
            raise RuntimeError("cell call-ledger scalar/forbidden contract changed")
        _site_count_vector(self.native_module_calls, ALL_NATIVE_SITES, batch_count)
        _site_count_vector(
            self.substitution_calls, descriptor.installed_compiled_sites, batch_count,
        )
        _site_count_vector(
            self.live_attention_gain_calls, descriptor.live_attention_gain_sites, batch_count,
        )

    @property
    def sha256(self) -> str:
        return _logical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class BackendCellResult:
    statistics: measurement.RowCellSufficientStatistics
    call_ledger: CellCallLedger
    component_tree_before_sha256: str
    component_tree_after_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.statistics, measurement.RowCellSufficientStatistics) or not (
            isinstance(self.call_ledger, CellCallLedger)
        ) or not _sha256_text(self.component_tree_before_sha256) or not _sha256_text(
            self.component_tree_after_sha256
        ):
            raise ValueError("backend cell result schema is malformed")


class MeasurementBackend(Protocol):
    """Capability boundary; implementations must be committed and source-closed."""

    batch_size: int
    source_paths: tuple[str, ...]

    def prepare(
        self, rows: torch.Tensor,
        requests: tuple[measurement.MeasurementRequest, ...],
    ) -> PreparedProgramBank: ...

    def execute_cell(
        self, request: measurement.MeasurementRequest, rows: torch.Tensor,
        program: ProgramDescriptor,
    ) -> BackendCellResult: ...

    def close(self) -> str: ...


@dataclass(frozen=True, slots=True)
class OutputPaths:
    authority: Path
    payload: Path
    receipt: Path
    failure: Path
    lock: Path


def output_paths(directory: Path, namespace: str) -> OutputPaths:
    if not namespace or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in namespace):
        raise ValueError("output namespace is not a safe lowercase identifier")
    root = directory.resolve()
    return OutputPaths(
        authority=root / f"{namespace}_authority.json",
        payload=root / f"{namespace}_payload.pt",
        receipt=root / f"{namespace}_receipt.json",
        failure=root / f"{namespace}_failure.json",
        lock=root / f".{namespace}.lock",
    )


class RunLock:
    """Create-only nonce/inode lock; only its owner can publish or release it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.nonce = secrets.token_hex(32)
        self.inode: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, (self.nonce + "\n").encode("ascii"))
            os.fsync(descriptor)
            self.inode = os.fstat(descriptor).st_ino
        finally:
            os.close(descriptor)

    def require_owned(self) -> None:
        if self.inode is None:
            raise RuntimeError("run lock is not held")
        stat = self.path.stat()
        if stat.st_ino != self.inode or self.path.read_text(encoding="ascii") != self.nonce + "\n":
            raise RuntimeError("run lock ownership was lost")

    def release(self) -> None:
        self.require_owned()
        self.path.unlink()
        self.inode = None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_create_only(path: Path, content: bytes, lock: RunLock) -> None:
    lock.require_owned()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        lock.require_owned()
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _torch_bytes(value: Any) -> bytes:
    import io
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _validate_installed_payload(path: Path, expected: dict[str, Any]) -> None:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or set(loaded) != set(expected):
        raise RuntimeError("installed per-document payload schema changed")
    for key, expected_value in expected.items():
        observed = loaded[key]
        if torch.is_tensor(expected_value):
            if not torch.is_tensor(observed) or not torch.equal(observed, expected_value):
                raise RuntimeError("installed per-document payload tensor changed")
        elif observed != expected_value or type(observed) is not type(expected_value):
            raise RuntimeError("installed per-document payload metadata changed")


def _authority_payload(
    authority: measurement.MeasurementWaveAuthority, source: SourceClosure,
    row_wave: RowWave, bank: PreparedProgramBank,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_any_measurement_cell",
        "authorized_for_final_role": False,
        "measurement_authority": asdict(authority),
        "measurement_authority_sha256": authority.sha256,
        "source_closure": asdict(source),
        "source_closure_sha256": source.sha256,
        "row_wave": row_wave.receipt(),
        "model_binding": asdict(bank.model),
        "program_descriptors": [asdict(value) for value in bank.programs],
        "program_realization_sha256s": list(bank.program_realization_sha256s),
        "program_bank_sha256": bank.sha256,
        "request_plan_sha256": measurement.REQUEST_PLAN_SHA256,
    }


def _validate_backend_surface(backend: MeasurementBackend) -> None:
    if type(backend.batch_size) is not int or backend.batch_size <= 0 or not isinstance(
        backend.source_paths, tuple
    ) or not backend.source_paths or len(set(backend.source_paths)) != len(
        backend.source_paths
    ) or any(not isinstance(value, str) or not value for value in backend.source_paths) or not (
        callable(backend.prepare) and callable(backend.execute_cell) and callable(backend.close)
    ):
        raise RuntimeError("measurement backend capability surface is malformed")


def run_transaction(
    *, backend: MeasurementBackend, row_receipt: Path, row_role: str,
    repo: Path, paths: OutputPaths,
) -> dict[str, Any]:
    """Run exactly one create-only authority -> 64 cells -> payload -> receipt transaction."""

    _validate_backend_surface(backend)
    if any(path.exists() for path in (
        paths.authority, paths.payload, paths.receipt, paths.failure, paths.lock,
    )):
        raise RuntimeError("cut-rank namespace is not pristine")
    lock = RunLock(paths.lock)
    lock.acquire()
    phase = "preflight"
    ordinal: int | None = None
    authority_written = False
    backend_closed = False
    try:
        source = committed_source_closure(
            repo, (*CORE_SOURCE_PATHS, *backend.source_paths),
        )
        row_wave = load_row_wave(row_receipt, row_role)
        rows = row_wave.clone_rows()
        phase = "prepare_program_bank"
        bank = backend.prepare(rows.clone(), measurement.REQUESTS)
        if not isinstance(bank, PreparedProgramBank):
            raise RuntimeError("backend did not return a typed prepared program bank")
        batch_count = math.ceil(row_wave.row_count / backend.batch_size)
        row_to_document, token_count = row_wave.clone_mapping_and_counts()
        authority = measurement.MeasurementWaveAuthority(
            source_commit=source.source_commit,
            source_receipt_sha256=row_wave.source_receipt_sha256,
            row_tensor_sha256=row_wave.row_tensor_sha256,
            row_provenance_sha256=row_wave.row_provenance_sha256,
            ordered_row_identity_sha256=row_wave.ordered_row_identity_sha256,
            ordered_row_to_document_sha256=measurement.tensor_sha256(row_to_document),
            ordered_document_ids_sha256=row_wave.ordered_document_ids_sha256,
            row_token_count_sha256=measurement.tensor_sha256(token_count),
            common_support_sha256=row_wave.common_support_sha256,
            model_realization_sha256=bank.model.model_realization_sha256,
            component_tree_sha256=bank.model.component_tree_sha256,
            program_bank_sha256=bank.sha256,
            source_closure_sha256=source.sha256,
            wave_nonce_sha256=_logical_sha256({
                "nonce": secrets.token_hex(32), "source_commit": source.source_commit,
                "row_wave_sha256": row_wave.sha256, "program_bank_sha256": bank.sha256,
            }),
            program_realization_sha256s=bank.program_realization_sha256s,
            row_count=row_wave.row_count,
            document_count=row_wave.document_count,
            total_scored_token_count=int(token_count.sum()),
            batch_count=batch_count,
        )
        authority_payload = _authority_payload(authority, source, row_wave, bank)
        authority_bytes = _json_bytes(authority_payload)
        phase = "publish_pre_outcome_authority"
        _publish_bytes_create_only(paths.authority, authority_bytes, lock)
        authority_written = True
        if paths.authority.read_bytes() != authority_bytes:
            raise RuntimeError("published authority bytes do not replay")
        collector = measurement.MeasurementCollector(
            authority=authority, row_to_document=row_to_document,
            row_token_count=token_count,
        )
        phase = "measure_cells"
        for request, program in zip(measurement.REQUESTS, bank.programs, strict=True):
            ordinal = request.ordinal
            result = backend.execute_cell(request, rows.clone(), program)
            if not isinstance(result, BackendCellResult):
                raise RuntimeError("backend returned an untyped cell result")
            result.call_ledger.validate(
                program, row_count=row_wave.row_count, batch_count=batch_count,
            )
            if result.component_tree_before_sha256 != bank.model.component_tree_sha256 or (
                result.component_tree_after_sha256 != bank.model.component_tree_sha256
            ):
                raise RuntimeError("model component tree changed during a cell")
            cell_receipt = measurement.CellMeasurementReceipt(
                authority_sha256=authority.sha256,
                request_sha256=request.sha256,
                ordinal=request.ordinal,
                cell=request.cell,
                program_realization_sha256=program.sha256,
                common_support_sha256=row_wave.common_support_sha256,
                ordered_row_identity_sha256=row_wave.ordered_row_identity_sha256,
                top1_correct_sha256=result.statistics.top1_correct_sha256,
                ce_sum_sha256=result.statistics.ce_sum_sha256,
                row_token_count_sha256=result.statistics.row_token_count_sha256,
                statistics_sha256=result.statistics.sha256,
                call_ledger_sha256=result.call_ledger.sha256,
                source_closure_sha256=source.sha256,
                model_tree_before_sha256=result.component_tree_before_sha256,
                model_tree_after_sha256=result.component_tree_after_sha256,
                outer_forward_count=result.call_ledger.outer_forward_count,
                batch_count=result.call_ledger.batch_count,
            )
            collector.add_cell(
                request=request, statistics=result.statistics, receipt=cell_receipt,
            )
        phase = "close_backend_before_publication"
        closed_component_tree_sha256 = backend.close()
        backend_closed = True
        if closed_component_tree_sha256 != bank.model.component_tree_sha256:
            raise RuntimeError("backend close did not restore the authorized component tree")
        phase = "finalize_sufficient_statistics"
        bundle = collector.finalize()
        payload_value = {
            "schema_version": SCHEMA_VERSION,
            "authority_sha256": authority.sha256,
            "ordered_document_ids_sha256": row_wave.ordered_document_ids_sha256,
            "document_row_count": bundle.payload.document_row_count,
            "document_token_count": bundle.payload.document_token_count,
            "top1_correct": bundle.payload.top1_correct,
            "ce_sum": bundle.payload.ce_sum,
            "per_document_payload_sha256": bundle.payload.sha256,
        }
        payload_bytes = _torch_bytes(payload_value)
        phase = "publish_payload"
        _publish_bytes_create_only(paths.payload, payload_bytes, lock)
        _validate_installed_payload(paths.payload, payload_value)
        payload_file_sha256 = file_sha256(paths.payload)
        receipt_value = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_discovery_measurement_payload",
            "authorized_for_final_role": False,
            "authority_path": str(paths.authority.resolve()),
            "authority_file_sha256": file_sha256(paths.authority),
            "measurement_authority_sha256": authority.sha256,
            "payload_path": str(paths.payload.resolve()),
            "payload_file_sha256": payload_file_sha256,
            "measurement_receipt": asdict(bundle.receipt),
            "measurement_receipt_sha256": bundle.receipt.sha256,
            "source_closure_sha256": source.sha256,
            "row_wave_sha256": row_wave.sha256,
            "program_bank_sha256": bank.sha256,
        }
        # Receipt is the last-written publication boundary.  Rehash every installed
        # predecessor and recheck lock ownership immediately before creating it.
        phase = "publish_receipt_last"
        lock.require_owned()
        _validate_installed_payload(paths.payload, payload_value)
        if file_sha256(paths.payload) != payload_file_sha256 or paths.authority.read_bytes() != authority_bytes:
            raise RuntimeError("authority/payload changed before receipt publication")
        receipt_bytes = _json_bytes(receipt_value)
        _publish_bytes_create_only(paths.receipt, receipt_bytes, lock)
        if paths.receipt.read_bytes() != receipt_bytes:
            raise RuntimeError("published receipt bytes do not replay")
        return json.loads(receipt_bytes)
    except Exception as error:
        if authority_written and not paths.failure.exists():
            failure = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed_no_scientific_interpretation",
                "authorized_for_final_role": False,
                "phase": phase,
                "ordinal": ordinal,
                "exception_type": type(error).__name__,
                "authority_file_sha256": (
                    file_sha256(paths.authority) if paths.authority.exists() else None
                ),
            }
            try:
                _publish_bytes_create_only(paths.failure, _json_bytes(failure), lock)
            except Exception:
                pass
        raise
    finally:
        try:
            if not backend_closed:
                backend.close()
        finally:
            if lock.inode is not None:
                lock.release()


def _load_backend(module_name: str) -> MeasurementBackend:
    module = importlib.import_module(module_name)
    creator = getattr(module, "create_backend", None)
    if not callable(creator):
        raise RuntimeError("backend module lacks create_backend()")
    return creator()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-module", required=True)
    parser.add_argument("--row-receipt", type=Path, default=DEFAULT_ROW_RECEIPT)
    parser.add_argument("--row-role", default=ROW_ROLE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    arguments = parser.parse_args(argv)
    receipt = run_transaction(
        backend=_load_backend(arguments.backend_module),
        row_receipt=arguments.row_receipt,
        row_role=arguments.row_role,
        repo=REPO,
        paths=output_paths(arguments.output_directory, arguments.namespace),
    )
    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(output_paths(
            arguments.output_directory, arguments.namespace,
        ).receipt),
        "measurement_receipt_sha256": receipt["measurement_receipt_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
