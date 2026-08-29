#!/usr/bin/env python3
"""Receipt-last stage-V0 validation for the Block-3 native-gate subset assay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/tensor_language")
sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
import collect_block3_native_gate_fit_v1 as collector
import native_gate_subset as subset


HERE = ROOT / "basis_aligned" / "polynomial_causal"
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
AMENDMENT = HERE / "BLOCK3_NATIVE_GATE_SUBSET_V1_VALIDATION_AMENDMENT.md"
V1_AMENDMENT = HERE / "BLOCK3_NATIVE_GATE_SUBSET_V1_VALIDATION_V1_AMENDMENT.md"
ROWS = BQ / ".rowcache/fineweb_n192_skip7000.pt"
ROWS_FILE_SHA256 = "d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c"
ROWS_RAW_SHA256 = "10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0"
ROW_ENTRY = "n192_skip7000"
EXPECTED_FIT_SOURCE_COMMIT = "3c1f8be82cb36f43fa6ec1af055b1b7831e205f1"
FIT_FILES = {
    collector.AUTHORITY: "cd83bbcd5dbf466a7ab57617a2b28ef2f62943c2ef011e1d71821a4547f8351a",
    collector.PAYLOAD: "8b25774257dd66f34ff6fb0c21fb0613efb12ce2773a0f3dcc8082343ceebdd9",
    collector.RECEIPT: "3facf67a90beea923a666f29a0770080d2277721e4443df444b6a416b186435e",
    HERE / "block3_native_gate_subset_v1_fit_authority.json": "d166df7a02a39296c4c95052f392ac29157947db41038788afab83222babae98",
    HERE / "block3_native_gate_subset_v1_programs.pt": "6f1ac8b2043edd1cb2a73992ee869c16c7516af27c608b56e821d57a06334d36",
    HERE / "block3_native_gate_subset_v1_fit_results.json": "22517a762c2f7c570c6fc383e3530635d7030a81a7ef1f4a6d735060ce971fbe",
    HERE / "block3_native_gate_subset_v1_fit_receipt.json": "dbf76301d4c7d5ac03942465da98c331429e1b2e84c771d767b1302b0137ab89",
}
FIT_AUTHORITY, PROGRAMS, FIT_RESULTS, FIT_RECEIPT = tuple(FIT_FILES)[3:]
V0_AUTHORITY = HERE / "block3_native_gate_subset_v1_validation_v0_authority.json"
V0_FAILURE = HERE / "block3_native_gate_subset_v1_validation_v0_failure.json"
V0_RESULTS = HERE / "block3_native_gate_subset_v1_validation_v0_results.json"
V0_RECEIPT = HERE / "block3_native_gate_subset_v1_validation_v0_receipt.json"
V0_AUTHORITY_SHA256 = "24d61b89f162d84e4aee17f9a714e99b348776ae21049b6287c6163be82c61cb"
V0_FAILURE_SHA256 = "9932117237c44c863872ffbdb400d8988503837b9f5b296447f37807d8a78bf2"
AUTHORITY = HERE / "block3_native_gate_subset_v1_validation_v1_authority.json"
RESULTS = HERE / "block3_native_gate_subset_v1_validation_v1_results.json"
RECEIPT = HERE / "block3_native_gate_subset_v1_validation_v1_receipt.json"
FAILURE = HERE / "block3_native_gate_subset_v1_validation_v1_failure.json"
LOCK = Path("/workspace/runs/.block3_native_gate_subset_v1_validation_v1.lock")
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_VALIDATION_AMENDMENT.md",
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_VALIDATION_V1_AMENDMENT.md",
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_VALIDATION_V0_FAILURE.md",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_validation_v0_authority.json",
    "basis_aligned/polynomial_causal/block3_native_gate_subset_v1_validation_v0_failure.json",
    "basis_aligned/polynomial_causal/validate_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/test_validate_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/native_gate_subset.py",
    "basis_aligned/polynomial_causal/test_native_gate_subset.py",
    "basis_aligned/polynomial_causal/collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/test_collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/fit_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/test_fit_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/test_grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)
ROLE = "n192_skip7000"
ROW_COUNT = 192
ROW_WIDTH = 513
MODEL_TOKENS = 256
POSITION_START = 64
POSITION_STOP = 256
TARGET_START = 65
TARGET_STOP = 257
BATCH_SIZE = 4
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2026082908
WIDTH = 1152
LAYER = 3
CUTS = (3, 4, 8, 17)
DEVICE = "cuda"


def file_sha256(path: Path) -> str:
    return collector.file_sha256(path)


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"validation source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"validation source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    core = {"commit": commit, "paths": hashes}
    return {**core, "sha256": logical_sha256(core)}


def verify_source_closure(expected: Mapping[str, Any]) -> None:
    if set(expected) != {"commit", "paths", "sha256"} or set(
        expected["paths"]
    ) != set(SOURCE_PATHS) or logical_sha256({
        "commit": expected["commit"], "paths": expected["paths"],
    }) != expected["sha256"]:
        raise RuntimeError("validation source closure is malformed")
    for relative, digest in expected["paths"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"validation source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected["commit"], "origin/main"],
        cwd=ROOT, check=True,
    )


def fit_file_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in FIT_FILES}


def validate_fit_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    observed = fit_file_hashes()
    expected = {str(path.relative_to(ROOT)): digest for path, digest in FIT_FILES.items()}
    if observed != expected:
        raise RuntimeError("sealed fit artifact bytes changed")
    collector_authority = json.loads(collector.AUTHORITY.read_text())
    collector_receipt = json.loads(collector.RECEIPT.read_text())
    fit_authority = json.loads(FIT_AUTHORITY.read_text())
    fit_results = json.loads(FIT_RESULTS.read_text())
    fit_receipt = json.loads(FIT_RECEIPT.read_text())
    raw_programs = torch.load(PROGRAMS, map_location="cpu", weights_only=True)
    if collector_authority.get("source_closure", {}).get("commit") != EXPECTED_FIT_SOURCE_COMMIT or (
        collector_receipt.get("authority_file_sha256") != FIT_FILES[collector.AUTHORITY]
    ) or collector_receipt.get("payload_file_sha256") != FIT_FILES[collector.PAYLOAD] or (
        fit_authority.get("source_closure", {}).get("commit") != EXPECTED_FIT_SOURCE_COMMIT
    ) or fit_receipt.get("fit_authority_file_sha256") != FIT_FILES[FIT_AUTHORITY] or (
        fit_receipt.get("programs_file_sha256") != FIT_FILES[PROGRAMS]
    ) or fit_receipt.get("results_file_sha256") != FIT_FILES[FIT_RESULTS] or (
        fit_receipt.get("collector_receipt_file_sha256") != FIT_FILES[collector.RECEIPT]
    ) or raw_programs.get("fit_authority_sha256") != fit_authority.get(
        "fit_authority_sha256"
    ) or fit_results.get("fit_authority_sha256") != fit_authority.get(
        "fit_authority_sha256"
    ) or set(raw_programs.get("programs", {})) != {
        f"{family}_k{budget}"
        for budget in (256, 512)
        for family in ("activation_selected", "random_prefilter", "label_permutation")
    }:
        raise RuntimeError("sealed fit artifact joins failed")
    return raw_programs, {"file_sha256s": observed, "fit_authority_sha256": fit_authority["fit_authority_sha256"]}


def v0_failure_lineage() -> dict[str, Any]:
    if file_sha256(V0_AUTHORITY) != V0_AUTHORITY_SHA256 or (
        file_sha256(V0_FAILURE) != V0_FAILURE_SHA256
    ) or V0_RESULTS.exists() or V0_RECEIPT.exists():
        raise RuntimeError("preserved validation V0 failure lineage changed")
    authority = json.loads(V0_AUTHORITY.read_text())
    failure = json.loads(V0_FAILURE.read_text())
    if authority.get("schema") != "block3_native_gate_subset_v1_validation_v0_authority" or (
        failure.get("schema") != "block3_native_gate_subset_v1_validation_v0_failure"
    ) or failure.get("authority_exists") is not True or failure.get("results_exists") is not False or (
        failure.get("receipt_exists") is not False
    ) or failure.get("error") != "native validation polarization did not replay MLP3":
        raise RuntimeError("preserved validation V0 failure joins changed")
    return {
        "authority_file_sha256": V0_AUTHORITY_SHA256,
        "failure_file_sha256": V0_FAILURE_SHA256,
        "authority_sha256": authority["authority_sha256"],
        "source_commit": authority["source_closure"]["commit"],
        "candidate_arms_scored": 0,
        "result_exists": False,
        "receipt_exists": False,
        "terminal_error": failure["error"],
    }


def row_binding() -> dict[str, Any]:
    if file_sha256(collector.ROW_RECEIPT) != collector.ROW_RECEIPT_SHA256 or (
        file_sha256(ROWS) != ROWS_FILE_SHA256
    ):
        raise RuntimeError("validation row bytes changed")
    receipt = json.loads(collector.ROW_RECEIPT.read_text())
    entry = receipt.get("entries", {}).get(ROW_ENTRY, {})
    records = receipt.get("document_provenance", {}).get("sets", {}).get(ROW_ENTRY)
    if entry.get("tensor_raw_sha256") != ROWS_RAW_SHA256 or not isinstance(
        records, list
    ) or len(records) != ROW_COUNT:
        raise RuntimeError("validation row receipt changed")
    document_ids = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("document_id"), str):
            raise RuntimeError("validation document identity changed")
        document_ids.append(record["document_id"])
    ordered_documents = list(dict.fromkeys(document_ids))
    document_index = {document: index for index, document in enumerate(ordered_documents)}
    row_to_document = [document_index[document] for document in document_ids]
    if len(ordered_documents) != 79 or len(row_to_document) != ROW_COUNT or max(
        row_to_document
    ) != 78:
        raise RuntimeError("canonical V0 source-document clustering changed")
    provenance = collector.validate_row_provenance()
    return {
        "receipt_sha256": collector.ROW_RECEIPT_SHA256,
        "row_file_sha256": ROWS_FILE_SHA256,
        "row_raw_sha256": ROWS_RAW_SHA256,
        "ordered_document_ids": ordered_documents,
        "ordered_document_ids_sha256": logical_sha256(ordered_documents),
        "row_to_document": row_to_document,
        "row_to_document_sha256": logical_sha256(row_to_document),
        "global_role_disjointness_sha256": provenance["disjointness_sha256"],
    }


def load_rows(binding: Mapping[str, Any]) -> torch.Tensor:
    before = file_sha256(ROWS)
    raw = torch.load(ROWS, map_location="cpu", weights_only=True)
    rows = raw["rows"] if isinstance(raw, dict) and set(raw) == {"rows"} else raw
    if before != binding["row_file_sha256"] or file_sha256(ROWS) != before or not (
        torch.is_tensor(rows)
    ) or tuple(rows.shape) != (ROW_COUNT, ROW_WIDTH) or rows.dtype != torch.long or (
        collector.tensor_sha256(rows) != ROWS_RAW_SHA256
    ):
        raise RuntimeError("validation rows failed load-time replay")
    return rows.contiguous()


def verify_inputs(
    source: Mapping[str, Any], fit: Mapping[str, Any], rows: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt, v0_lineage: Mapping[str, Any],
) -> None:
    verify_source_closure(source)
    if fit_file_hashes() != fit["file_sha256s"] or row_binding() != dict(rows) or (
        facade.validate_snapshot(verify_weights_sha256=True) != checkpoint
    ) or v0_failure_lineage() != dict(v0_lineage):
        raise RuntimeError("validation input drift")


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(
        list(model.named_parameters()) + list(model.named_buffers()), key=lambda item: item[0],
    ):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps({
            "name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        }, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def materialize_program(value: Mapping[str, torch.Tensor], device: str = DEVICE) -> subset.NativeGateSubsetProgram:
    required = {"indices", "left", "right", "decoder", "bias"}
    if set(value) != required:
        raise RuntimeError("stored validation program schema changed")
    program = subset.NativeGateSubsetProgram(**{
        key: tensor.detach().to(device=device).contiguous() for key, tensor in value.items()
    })
    if program.left.dtype != torch.float32:
        raise RuntimeError("validation program is not executable float32")
    return program


@dataclass(slots=True)
class CallLedger:
    attention: list[int]
    mlp: list[int]
    prefix_batches_by_wave: dict[str, int] = field(default_factory=dict)
    teacher_mlp3_calls_by_wave: dict[str, int] = field(default_factory=dict)
    suffix_calls_by_wave_arm: dict[str, int] = field(default_factory=dict)
    suffix_calls_by_family: dict[str, int] = field(default_factory=dict)
    native_typed_down_calls_by_wave_term: dict[str, int] = field(default_factory=dict)
    candidate_typed_decoder_calls_by_wave_term: dict[str, int] = field(default_factory=dict)
    direct_program_calls_by_wave_arm: dict[str, int] = field(default_factory=dict)
    outer_model_forward_calls: int = 0
    outer_model_returned: int = 0
    student_native_mlp3_calls: int = 0

    @classmethod
    def empty(cls) -> "CallLedger":
        return cls([0] * 18, [0] * 18)

    def receipt(self) -> dict[str, Any]:
        return {
            "attention_calls_by_site": {str(i): n for i, n in enumerate(self.attention)},
            "mlp_calls_by_site": {str(i): n for i, n in enumerate(self.mlp)},
            "prefix_batches_by_wave": dict(sorted(self.prefix_batches_by_wave.items())),
            "teacher_mlp3_calls_by_wave": dict(sorted(self.teacher_mlp3_calls_by_wave.items())),
            "suffix_calls_by_wave_arm": dict(sorted(self.suffix_calls_by_wave_arm.items())),
            "suffix_calls_by_family": dict(sorted(self.suffix_calls_by_family.items())),
            "native_typed_down_calls_by_wave_term": dict(sorted(
                self.native_typed_down_calls_by_wave_term.items()
            )),
            "candidate_typed_decoder_calls_by_wave_term": dict(sorted(
                self.candidate_typed_decoder_calls_by_wave_term.items()
            )),
            "direct_program_calls_by_wave_arm": dict(sorted(
                self.direct_program_calls_by_wave_arm.items()
            )),
            "outer_model_forward_calls": self.outer_model_forward_calls,
            "outer_model_returned": self.outer_model_returned,
            "student_native_mlp3_calls": self.student_native_mlp3_calls,
        }


def _increment(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def _suffix_family(arm: str) -> str:
    if arm == "native":
        return "native_teacher"
    if arm == "bias_only":
        return "bias_only_omission"
    if arm.startswith("omit_"):
        return "singleton_omission"
    if arm.startswith("activation_") and arm.rsplit("_", 1)[-1] in subset.TERM_NAMES:
        return "singleton_replacement"
    if arm.startswith("activation_"):
        return "candidate_allterm"
    if arm.startswith("random_"):
        return "random_control"
    if arm.startswith("permutation_"):
        return "permutation_control"
    if arm.startswith("mirror_"):
        return "mirror"
    raise RuntimeError(f"unregistered validation suffix arm: {arm}")


def _wave_arms(budget: int, *, include_omissions: bool) -> tuple[str, ...]:
    arms = ["native", "bias_only"]
    if include_omissions:
        arms.extend(f"omit_{name}" for name in subset.TERM_NAMES)
    arms.extend((
        f"activation_k{budget}",
        *(f"activation_k{budget}_{name}" for name in subset.TERM_NAMES),
        f"random_k{budget}", f"permutation_k{budget}", f"mirror_k{budget}",
    ))
    return tuple(arms)


def validate_call_ledger(calls: CallLedger, *, opened_512: bool) -> None:
    batches = math.ceil(ROW_COUNT / BATCH_SIZE)
    passes = 2 if opened_512 else 1
    suffix_per_batch = 14 + (10 if opened_512 else 0)
    expected = CallLedger.empty()
    for site in range(4):
        expected.attention[site] = batches * passes
    for site in range(4):
        expected.mlp[site] = batches * passes
    for site in range(4, 18):
        expected.attention[site] = batches * suffix_per_batch
        expected.mlp[site] = batches * suffix_per_batch
    waves = ((256, True),) + (((512, False),) if opened_512 else ())
    for budget, include_omissions in waves:
        wave = f"k{budget}"
        expected.prefix_batches_by_wave[wave] = batches
        expected.teacher_mlp3_calls_by_wave[wave] = batches
        for arm in _wave_arms(budget, include_omissions=include_omissions):
            expected.suffix_calls_by_wave_arm[f"{wave}/{arm}"] = batches
            _increment(expected.suffix_calls_by_family, _suffix_family(arm), batches)
        for name in subset.TERM_NAMES:
            expected.native_typed_down_calls_by_wave_term[f"{wave}/{name}"] = batches
            expected.candidate_typed_decoder_calls_by_wave_term[f"{wave}/{name}"] = batches
        for arm in (f"activation_k{budget}", f"random_k{budget}", f"permutation_k{budget}"):
            expected.direct_program_calls_by_wave_arm[f"{wave}/{arm}"] = batches
    if calls.receipt() != expected.receipt():
        raise RuntimeError("validation measured call census differs from authority")


@dataclass(slots=True)
class Prefix:
    post: torch.Tensor
    x0: torch.Tensor
    first_value: torch.Tensor
    z: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    native_write: torch.Tensor


@torch.no_grad()
def prefix_to_mlp3(
    model: torch.nn.Module, tokens: torch.Tensor, calls: CallLedger, *, wave: str = "test",
) -> Prefix:
    x = F.rms_norm(model.transformer.wte(tokens), (WIDTH,))
    x0, first_value = x, None
    for site in range(LAYER + 1):
        block = model.transformer.h[site]
        h = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first_value = block.attn(F.rms_norm(h, (WIDTH,)), first_value)
        calls.attention[site] += 1
        post = h + attention
        z = F.rms_norm(post, (WIDTH,))
        if site == LAYER:
            gamma = torch.rsqrt(post.square().mean(-1, keepdim=True) + torch.finfo(post.dtype).eps)
            u, v = gamma * h, gamma * attention
            if float((u + v - z).abs().max()) > 2e-6:
                raise RuntimeError("validation RMS polarization replay failed")
            native_write = block.mlp(z)
            calls.mlp[site] += 1
            _increment(calls.prefix_batches_by_wave, wave)
            _increment(calls.teacher_mlp3_calls_by_wave, wave)
            return Prefix(post, x0, first_value, z, u, v, native_write)
        x = post + block.mlp(z)
        calls.mlp[site] += 1
    raise AssertionError("validation prefix did not reach MLP3")


@torch.no_grad()
def suffix_from_write(
    model: torch.nn.Module, prefix: Prefix, write: torch.Tensor, calls: CallLedger,
    *, wave: str = "test", arm: str = "native",
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    x = prefix.post.clone() + write
    x0 = prefix.x0.clone()
    first_value = prefix.first_value.clone()
    states = {3: x}
    for site in range(4, 18):
        block = model.transformer.h[site]
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first_value = block.attn(F.rms_norm(x, (WIDTH,)), first_value)
        calls.attention[site] += 1
        x = x + attention
        x = x + block.mlp(F.rms_norm(x, (WIDTH,)))
        calls.mlp[site] += 1
        if site in CUTS:
            states[site] = x
    logits = model.lm_head(F.rms_norm(x, (WIDTH,)))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    _increment(calls.suffix_calls_by_wave_arm, f"{wave}/{arm}")
    _increment(calls.suffix_calls_by_family, _suffix_family(arm))
    return states, logits


def _zeros(documents: int) -> torch.Tensor:
    return torch.zeros(documents, dtype=torch.float64)


def empty_ledger(documents: int) -> dict[str, Any]:
    return {
        "token_count": _zeros(documents), "local_sse": _zeros(documents),
        "local_energy": _zeros(documents), "kl_sum": _zeros(documents),
        "ce_delta_sum": _zeros(documents), "centered_error": _zeros(documents),
        "centered_stake_energy": _zeros(documents), "response_dot": _zeros(documents),
        "response_native_energy": _zeros(documents), "response_arm_energy": _zeros(documents),
        "top1_agree": _zeros(documents), "target_top1": _zeros(documents),
        "cuts": {str(cut): {
            "error": _zeros(documents), "native_energy": _zeros(documents),
        } for cut in CUTS},
    }


def empty_term_ledger(documents: int) -> dict[str, dict[str, torch.Tensor]]:
    return {name: {"sse": _zeros(documents), "energy": _zeros(documents)} for name in subset.TERM_NAMES}


def _row_sums(value: torch.Tensor) -> torch.Tensor:
    return value.detach().double().flatten(1).sum(1).cpu()


def accumulate_local(
    ledger: dict[str, Any], native: torch.Tensor, arm: torch.Tensor,
    bias: torch.Tensor, document_indices: torch.Tensor,
) -> None:
    selection = (..., slice(POSITION_START, POSITION_STOP), slice(None))
    error = _row_sums((arm[selection] - native[selection]).square())
    energy = _row_sums((native[selection] - bias.reshape(1, 1, -1)).square())
    for row, document in enumerate(document_indices.tolist()):
        ledger["local_sse"][document] += error[row]
        ledger["local_energy"][document] += energy[row]


def accumulate_terms(
    ledger: dict[str, dict[str, torch.Tensor]], native: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor], document_indices: torch.Tensor,
) -> None:
    selection = (..., slice(POSITION_START, POSITION_STOP), slice(None))
    for name in subset.TERM_NAMES:
        sse = _row_sums((candidate[name][selection] - native[name][selection]).square())
        energy = _row_sums(native[name][selection].square())
        for row, document in enumerate(document_indices.tolist()):
            ledger[name]["sse"][document] += sse[row]
            ledger[name]["energy"][document] += energy[row]


def accumulate_final(
    ledger: dict[str, Any], native_states: Mapping[int, torch.Tensor],
    arm_states: Mapping[int, torch.Tensor], native_logits: torch.Tensor,
    bias_logits: torch.Tensor, arm_logits: torch.Tensor, targets: torch.Tensor,
    document_indices: torch.Tensor,
) -> None:
    state_slice = (..., slice(POSITION_START, POSITION_STOP), slice(None))
    logit_slice = (..., slice(POSITION_START, POSITION_STOP), slice(None))
    target = targets[:, TARGET_START:TARGET_STOP]
    native_scored = native_logits[logit_slice]
    bias_scored = bias_logits[logit_slice]
    arm_scored = arm_logits[logit_slice]
    native_logp = F.log_softmax(native_scored, -1)
    arm_logp = F.log_softmax(arm_scored, -1)
    kl = (native_logp.exp() * (native_logp - arm_logp)).sum(-1)
    native_ce = F.cross_entropy(
        native_scored.flatten(0, 1), target.flatten(), reduction="none",
    ).view_as(target)
    arm_ce = F.cross_entropy(
        arm_scored.flatten(0, 1), target.flatten(), reduction="none",
    ).view_as(target)
    native_centered = native_scored - native_scored.mean(-1, keepdim=True)
    bias_centered = bias_scored - bias_scored.mean(-1, keepdim=True)
    arm_centered = arm_scored - arm_scored.mean(-1, keepdim=True)
    native_response = native_centered - bias_centered
    arm_response = arm_centered - bias_centered
    centered_error = (arm_centered - native_centered).square()
    token_count = torch.full((len(target),), target.shape[1], dtype=torch.float64)
    values = {
        "token_count": token_count,
        "kl_sum": _row_sums(kl),
        "ce_delta_sum": _row_sums(arm_ce - native_ce),
        "centered_error": _row_sums(centered_error),
        "centered_stake_energy": _row_sums(native_response.square()),
        "response_dot": _row_sums(native_response * arm_response),
        "response_native_energy": _row_sums(native_response.square()),
        "response_arm_energy": _row_sums(arm_response.square()),
        "top1_agree": _row_sums((arm_scored.argmax(-1) == native_scored.argmax(-1)).double()),
        "target_top1": _row_sums((arm_scored.argmax(-1) == target).double()),
    }
    for cut in CUTS:
        values[f"cut_{cut}_error"] = _row_sums(
            (arm_states[cut][state_slice] - native_states[cut][state_slice]).square()
        )
        values[f"cut_{cut}_native"] = _row_sums(native_states[cut][state_slice].square())
    for row, document in enumerate(document_indices.tolist()):
        for name in (
            "token_count", "kl_sum", "ce_delta_sum", "centered_error",
            "centered_stake_energy", "response_dot", "response_native_energy",
            "response_arm_energy", "top1_agree", "target_top1",
        ):
            ledger[name][document] += values[name][row]
        for cut in CUTS:
            ledger["cuts"][str(cut)]["error"][document] += values[f"cut_{cut}_error"][row]
            ledger["cuts"][str(cut)]["native_energy"][document] += values[f"cut_{cut}_native"][row]


def bootstrap_weights(documents: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    samples = torch.randint(documents, (BOOTSTRAP_DRAWS, documents), generator=generator)
    counts = torch.zeros(BOOTSTRAP_DRAWS, documents, dtype=torch.float64)
    counts.scatter_add_(1, samples, torch.ones_like(samples, dtype=torch.float64))
    return torch.cat((torch.ones(1, documents, dtype=torch.float64), counts), 0)


def _series(numerator: torch.Tensor, denominator: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    den = weights @ denominator
    if not bool((den > 0).all()):
        raise RuntimeError("validation aggregate denominator is nonpositive")
    return (weights @ numerator) / den


def describe(series: torch.Tensor) -> dict[str, float]:
    if series.shape != (BOOTSTRAP_DRAWS + 1,) or not torch.isfinite(series).all():
        raise RuntimeError("validation bootstrap series is malformed")
    bootstrap = series[1:]
    return {
        "point": float(series[0]),
        "q05": float(torch.quantile(bootstrap, 0.05)),
        "q50": float(torch.quantile(bootstrap, 0.50)),
        "q95": float(torch.quantile(bootstrap, 0.95)),
    }


def summarize_arm(
    ledger: Mapping[str, Any], bias_ledger: Mapping[str, Any], weights: torch.Tensor,
) -> dict[str, Any]:
    local = _series(ledger["local_sse"], ledger["local_energy"], weights).sqrt()
    per_document = (ledger["local_sse"] / ledger["local_energy"].clamp_min(1e-300)).sqrt()
    kl_ratio = _series(ledger["kl_sum"], bias_ledger["kl_sum"], weights)
    ce = _series(ledger["ce_delta_sum"], ledger["token_count"], weights)
    centered = _series(ledger["centered_error"], ledger["centered_stake_energy"], weights).sqrt()
    dot = weights @ ledger["response_dot"]
    norm = ((weights @ ledger["response_native_energy"]) * (weights @ ledger["response_arm_energy"])).sqrt()
    cosine = dot / norm.clamp_min(1e-300)
    summary = {
        "summed_local_nrmse": describe(local),
        "per_document_local_nrmse_q90": float(torch.quantile(per_document, 0.90)),
        "kl_sum": describe(weights @ ledger["kl_sum"]),
        "kl_over_bias_only": describe(kl_ratio),
        "ce_delta_nat": describe(ce),
        "centered_logit_response_nrmse": describe(centered),
        "centered_logit_response_cosine": describe(cosine),
        "native_top1_agreement": describe(_series(
            ledger["top1_agree"], ledger["token_count"], weights,
        )),
        "target_top1_accuracy": describe(_series(
            ledger["target_top1"], ledger["token_count"], weights,
        )),
        "cuts": {},
    }
    for cut in CUTS:
        error = ledger["cuts"][str(cut)]["error"]
        summary["cuts"][str(cut)] = {
            "native_state_nrmse": describe(_series(
                error, ledger["cuts"][str(cut)]["native_energy"], weights,
            ).sqrt()),
            "over_bias_only_error": describe(_series(
                error, bias_ledger["cuts"][str(cut)]["error"], weights,
            ).sqrt()),
            "over_cut3_error": describe(_series(
                error, ledger["cuts"]["3"]["error"], weights,
            ).sqrt()) if cut != 3 else describe(torch.ones(BOOTSTRAP_DRAWS + 1)),
        }
    return summary


def serialize_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in ledger.items():
        if key == "cuts":
            output[key] = {
                cut: {name: tensor.tolist() for name, tensor in values.items()}
                for cut, values in value.items()
            }
        else:
            output[key] = value.tolist()
    return output


def _native_terms(
    block: torch.nn.Module, balanced_left: torch.Tensor, balanced_right: torch.Tensor,
    u: torch.Tensor, v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    lu, ru = F.linear(u, balanced_left), F.linear(u, balanced_right)
    lv, rv = F.linear(v, balanced_left), F.linear(v, balanced_right)
    features = {"uu": lu * ru, "uv": lu * rv, "vu": lv * ru, "vv": lv * rv}
    return {name: F.linear(value, block.mlp.Down.weight) for name, value in features.items()}


def _sum_terms(terms: Mapping[str, torch.Tensor], bias: torch.Tensor) -> torch.Tensor:
    output = sum((terms[name] for name in subset.TERM_NAMES), torch.zeros_like(terms["uu"]))
    return output + bias.reshape(1, 1, -1)


def replay_diagnostics(reference: torch.Tensor, replay: torch.Tensor) -> dict[str, float]:
    error = replay - reference
    if not bool(torch.isfinite(reference).all() and torch.isfinite(replay).all()):
        raise RuntimeError("validation algebraic replay produced a nonfinite tensor")
    tiny = torch.finfo(torch.float32).tiny
    absolute_max = float(error.abs().max())
    reference_max = float(reference.abs().max())
    absolute_rms = float(error.square().mean().sqrt())
    reference_rms = float(reference.square().mean().sqrt())
    diagnostics = {
        "absolute_max": absolute_max,
        "reference_max": reference_max,
        "relative_max": absolute_max / max(reference_max, tiny),
        "absolute_rms": absolute_rms,
        "reference_rms": reference_rms,
        "relative_rms": absolute_rms / max(reference_rms, tiny),
    }
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise RuntimeError("validation algebraic replay diagnostic is nonfinite")
    return diagnostics


def require_replay(diagnostics: Mapping[str, float], *, label: str) -> None:
    if diagnostics["relative_max"] > 2e-5 or diagnostics["relative_rms"] > 2e-5:
        raise RuntimeError(
            f"{label} relative replay failed: max={diagnostics['relative_max']:.9g}, "
            f"rms={diagnostics['relative_rms']:.9g}"
        )


def arm_writes(
    prefix: Prefix, block: torch.nn.Module, program: subset.NativeGateSubsetProgram,
    random_program: subset.NativeGateSubsetProgram,
    permutation_program: subset.NativeGateSubsetProgram,
    balanced_left: torch.Tensor, balanced_right: torch.Tensor, budget: int,
    calls: CallLedger,
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor],
    dict[str, dict[str, float]],
]:
    native_terms = _native_terms(block, balanced_left, balanced_right, prefix.u, prefix.v)
    wave = f"k{budget}"
    for name in subset.TERM_NAMES:
        _increment(calls.native_typed_down_calls_by_wave_term, f"{wave}/{name}")
    native_replay = _sum_terms(native_terms, block.mlp.Down_bias)
    native_diagnostics = replay_diagnostics(prefix.native_write, native_replay)
    require_replay(native_diagnostics, label="native validation polarization")
    candidate_terms = program.terms(prefix.u, prefix.v)
    for name in subset.TERM_NAMES:
        _increment(calls.candidate_typed_decoder_calls_by_wave_term, f"{wave}/{name}")
    activation = program.write(prefix.z)
    random_write = random_program.write(prefix.z)
    permutation = permutation_program.write(prefix.z)
    for arm in (f"activation_k{budget}", f"random_k{budget}", f"permutation_k{budget}"):
        _increment(calls.direct_program_calls_by_wave_arm, f"{wave}/{arm}")
    typed_replay = _sum_terms(candidate_terms, program.bias)
    candidate_diagnostics = replay_diagnostics(activation, typed_replay)
    require_replay(candidate_diagnostics, label="validation program direct/typed")
    bias = block.mlp.Down_bias.reshape(1, 1, -1)
    writes = {
        "native": prefix.native_write,
        "bias_only": bias.expand_as(prefix.native_write),
        f"activation_k{budget}": activation,
        f"random_k{budget}": random_write,
        f"permutation_k{budget}": permutation,
        f"mirror_k{budget}": 2 * prefix.native_write - activation,
    }
    for name in subset.TERM_NAMES:
        writes[f"omit_{name}"] = bias + sum(
            (native_terms[other] for other in subset.TERM_NAMES if other != name),
            torch.zeros_like(native_terms[name]),
        )
        writes[f"activation_k{budget}_{name}"] = bias + candidate_terms[name] + sum(
            (native_terms[other] for other in subset.TERM_NAMES if other != name),
            torch.zeros_like(native_terms[name]),
        )
    return writes, native_terms, candidate_terms, {
        "native": native_diagnostics, "candidate": candidate_diagnostics,
    }


def run_wave(
    *, model: torch.nn.Module, rows: torch.Tensor, document_indices: torch.Tensor,
    programs: Mapping[str, subset.NativeGateSubsetProgram], budget: int,
    ledgers: dict[str, dict[str, Any]], term_ledgers: dict[str, Any], calls: CallLedger,
    include_omissions: bool,
) -> dict[str, dict[str, float]]:
    block = model.transformer.h[LAYER]
    balanced_left, balanced_right, _ = collector.balance_product_gauge(
        block.mlp.Left.weight.detach(), block.mlp.Right.weight.detach(),
    )
    documents = int(document_indices.max()) + 1
    replay_max: dict[str, dict[str, float]] = {"native": {}, "candidate": {}}
    wave = f"k{budget}"
    for start in range(0, ROW_COUNT, BATCH_SIZE):
        tokens = rows[start:start + BATCH_SIZE, :MODEL_TOKENS].to(DEVICE)
        targets = rows[start:start + BATCH_SIZE].to(DEVICE)
        docs = document_indices[start:start + BATCH_SIZE]
        prefix = prefix_to_mlp3(model, tokens, calls, wave=wave)
        writes, native_terms, candidate_terms, replay = arm_writes(
            prefix, block, programs[f"activation_selected_k{budget}"],
            programs[f"random_prefilter_k{budget}"],
            programs[f"label_permutation_k{budget}"], balanced_left, balanced_right,
            budget, calls,
        )
        for family, diagnostics in replay.items():
            for name, value in diagnostics.items():
                replay_max[family][name] = max(replay_max[family].get(name, 0.0), value)
        if not include_omissions:
            writes = {
                name: value for name, value in writes.items()
                if name in {"native", "bias_only", f"activation_k{budget}",
                            f"random_k{budget}", f"permutation_k{budget}",
                            f"mirror_k{budget}"}
                or name.startswith(f"activation_k{budget}_")
            }
        native_states, native_logits = suffix_from_write(
            model, prefix, writes.pop("native"), calls, wave=wave, arm="native",
        )
        bias_states, bias_logits = suffix_from_write(
            model, prefix, writes.pop("bias_only"), calls, wave=wave, arm="bias_only",
        )
        for arm, write in writes.items():
            if arm not in ledgers:
                ledgers[arm] = empty_ledger(documents)
            accumulate_local(
                ledgers[arm], prefix.native_write, write, block.mlp.Down_bias, docs,
            )
            arm_states, arm_logits = suffix_from_write(
                model, prefix, write, calls, wave=wave, arm=arm,
            )
            accumulate_final(
                ledgers[arm], native_states, arm_states, native_logits, bias_logits,
                arm_logits, targets, docs,
            )
        if include_omissions and "bias_only" not in ledgers:
            ledgers["bias_only"] = empty_ledger(documents)
        if include_omissions:
            accumulate_local(
                ledgers["bias_only"], prefix.native_write,
                block.mlp.Down_bias.reshape(1, 1, -1).expand_as(prefix.native_write),
                block.mlp.Down_bias, docs,
            )
            accumulate_final(
                ledgers["bias_only"], native_states, bias_states, native_logits,
                bias_logits, bias_logits, targets, docs,
            )
        key = f"activation_k{budget}"
        if key not in term_ledgers:
            term_ledgers[key] = empty_term_ledger(documents)
        accumulate_terms(term_ledgers[key], native_terms, candidate_terms, docs)
        del native_logits, bias_logits, native_states, bias_states, prefix, writes
    return replay_max


def summarize_terms(ledger: Mapping[str, Any], weights: torch.Tensor) -> dict[str, Any]:
    return {
        name: describe(_series(values["sse"], values["energy"], weights).sqrt())
        for name, values in ledger.items()
    }


def eligibility(
    budget: int, summaries: Mapping[str, Any], weights: torch.Tensor,
    ledgers: Mapping[str, Any], bias_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = summaries[f"activation_k{budget}"]
    random = summaries[f"random_k{budget}"]
    permutation = summaries[f"permutation_k{budget}"]
    mirror = summaries[f"mirror_k{budget}"]
    full_stake = float((weights[0] @ bias_ledger["kl_sum"]))
    singleton: dict[str, Any] = {}
    all_material_positive = True
    for name in subset.TERM_NAMES:
        omission = ledgers[f"omit_{name}"]
        replacement = ledgers[f"activation_k{budget}_{name}"]
        omission_kl = float(weights[0] @ omission["kl_sum"])
        replacement_kl = float(weights[0] @ replacement["kl_sum"])
        material = omission_kl >= 0.05 * full_stake
        recovery = 1 - replacement_kl / omission_kl if omission_kl > 0 else None
        if material and not (recovery is not None and recovery > 0):
            all_material_positive = False
        singleton[name] = {
            "omission_kl": omission_kl, "material": material,
            "replacement_kl": replacement_kl, "recovery_point": recovery,
        }
    gates = {
        "summed_local_nrmse_le_0p20": candidate["summed_local_nrmse"]["point"] <= 0.20,
        "kl_ratio_point_le_0p20": candidate["kl_over_bias_only"]["point"] <= 0.20,
        "kl_ratio_q95_le_0p35": candidate["kl_over_bias_only"]["q95"] <= 0.35,
        "ce_q95_le_0p01": candidate["ce_delta_nat"]["q95"] <= 0.01,
        "beats_random_point_kl_ratio": candidate["kl_over_bias_only"]["point"] < random["kl_over_bias_only"]["point"],
        "beats_permutation_point_kl_ratio": candidate["kl_over_bias_only"]["point"] < permutation["kl_over_bias_only"]["point"],
        "material_singleton_recovery_positive": all_material_positive,
        "mirror_point_kl_ratio_le_0p35": mirror["kl_over_bias_only"]["point"] <= 0.35,
    }
    return {
        "budget": budget, "gates": gates, "eligible": all(gates.values()),
        "singleton_materiality_and_recovery": singleton,
    }


def _authority(
    source: Mapping[str, Any], fit: Mapping[str, Any], rows: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt, v0_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema": "block3_native_gate_subset_v1_validation_v1_authority",
        "status": (
            "frozen_after_bound_v0_native_only_failure_before_any_v1_candidate_"
            "write_suffix_metric_or_outcome_loaded"
        ),
        "authorized_for_final_role": False,
        "authorized_for_global_ledger_credit": False,
        "source_closure": dict(source), "fit_binding": dict(fit),
        "v0_failure_lineage": dict(v0_lineage),
        "row_binding": dict(rows), "checkpoint": asdict(checkpoint),
        "role": ROLE, "row_count": ROW_COUNT, "model_tokens": MODEL_TOKENS,
        "scored_positions_half_open": [POSITION_START, POSITION_STOP],
        "target_positions_half_open": [TARGET_START, TARGET_STOP],
        "batch_size": BATCH_SIZE, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_unit": "source_document",
        "budget_order": [256, 512],
        "k512_open_rule": "only_if_k256_is_not_validation_eligible",
        "allterm_denominator": "native_bias_only_z_Q",
        "intervention_positions_half_open": [0, MODEL_TOKENS],
    }
    return {**core, "authority_sha256": logical_sha256(core)}


def run() -> dict[str, Any]:
    namespace = (AUTHORITY, RESULTS, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in namespace):
        raise RuntimeError("validation V0 namespace is not pristine")
    claim = collector.acquire_claim(LOCK)
    started = time.time()
    try:
        source = source_closure()
        raw_programs, fit = validate_fit_artifacts()
        v0_lineage = v0_failure_lineage()
        rows_binding = row_binding()
        checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
        authority = _authority(source, fit, rows_binding, checkpoint, v0_lineage)
        claim.verify()
        collector.create_json(AUTHORITY, authority)
        if json.loads(AUTHORITY.read_text()) != authority:
            raise RuntimeError("validation authority did not replay")
        verify_inputs(source, fit, rows_binding, checkpoint, v0_lineage)

        rows = load_rows(rows_binding)
        documents = rows_binding["ordered_document_ids"]
        document_indices = torch.tensor(
            rows_binding["row_to_document"], dtype=torch.long,
        )
        model, loaded_checkpoint = facade.load_bilin18(
            device=DEVICE, dtype=torch.float32, verify_weights_sha256=True,
        )
        if loaded_checkpoint != checkpoint:
            raise RuntimeError("validation loaded checkpoint differs from authority")
        programs = {
            key: materialize_program(value)
            for key, value in raw_programs["programs"].items()
        }
        model_before = model_state_sha256(model)
        calls = CallLedger.empty()
        ledgers: dict[str, dict[str, Any]] = {}
        term_ledgers: dict[str, Any] = {}
        replay = {"256": run_wave(
            model=model, rows=rows, document_indices=document_indices,
            programs=programs, budget=256, ledgers=ledgers,
            term_ledgers=term_ledgers, calls=calls, include_omissions=True,
        )}
        weights = bootstrap_weights(len(documents))
        bias_ledger = ledgers["bias_only"]
        summaries = {
            arm: summarize_arm(ledger, bias_ledger, weights)
            for arm, ledger in ledgers.items() if arm != "bias_only"
        }
        summaries["bias_only"] = {
            "kl_sum": describe(weights @ bias_ledger["kl_sum"]),
            "summed_local_nrmse": describe(_series(
                bias_ledger["local_sse"], bias_ledger["local_energy"], weights,
            ).sqrt()),
        }
        decisions = {"256": eligibility(256, summaries, weights, ledgers, bias_ledger)}
        opened_512 = not decisions["256"]["eligible"]
        if opened_512:
            replay["512"] = run_wave(
                model=model, rows=rows, document_indices=document_indices,
                programs=programs, budget=512, ledgers=ledgers,
                term_ledgers=term_ledgers, calls=calls, include_omissions=False,
            )
            for arm, ledger in ledgers.items():
                if "k512" in arm:
                    summaries[arm] = summarize_arm(ledger, bias_ledger, weights)
            decisions["512"] = eligibility(512, summaries, weights, ledgers, bias_ledger)
        validate_call_ledger(calls, opened_512=opened_512)
        selected = next(
            (budget for budget in (256, 512) if str(budget) in decisions and decisions[str(budget)]["eligible"]),
            None,
        )
        if selected is not None:
            next_action = {
                "kind": "complete_validation_cube_for_eligible_candidate",
                "budget": selected,
            }
        else:
            k512 = summaries["activation_k512"]
            mirror512 = summaries["mirror_k512"]
            downstream_null_screen = (
                k512["summed_local_nrmse"]["point"] > 0.20
                and k512["kl_over_bias_only"]["q95"] <= 0.35
                and mirror512["kl_over_bias_only"]["q95"] <= 0.35
            )
            next_action = {
                "kind": (
                    "complete_validation_cube_for_k512_downstream_null_test"
                    if downstream_null_screen
                    else "stop_activation_family_and_preregister_finite_suffix_family"
                ),
                "budget": 512 if downstream_null_screen else None,
            }
        model_after = model_state_sha256(model)
        if model_after != model_before:
            raise RuntimeError("validation model content changed")
        result = {
            "schema": "block3_native_gate_subset_v1_validation_v1_results",
            "authority_sha256": authority["authority_sha256"],
            "status": "validation_v1_complete_no_final_rows_opened",
            "role": ROLE, "ordered_document_ids": documents,
            "opened_k512": opened_512, "validation_eligible_budget": selected,
            "registered_next_action": next_action,
            "decisions": decisions, "summaries": summaries,
            "term_nrmse": {
                key: summarize_terms(value, weights) for key, value in term_ledgers.items()
            },
            "algebraic_replay_diagnostics_maxima": replay,
            "per_document_ledgers": {
                arm: serialize_ledger(value) for arm, value in ledgers.items()
            },
            "call_ledger": calls.receipt(),
            "model_state_before_sha256": model_before,
            "model_state_after_sha256": model_after,
            "evaluation_rows_loaded": ROW_COUNT,
            "final_rows_loaded": 0,
            "elapsed_seconds": time.time() - started,
            "torch_version": torch.__version__, "python_version": platform.python_version(),
        }
        claim.verify()
        verify_inputs(source, fit, rows_binding, checkpoint, v0_lineage)
        if model_state_sha256(model) != model_before or json.loads(
            AUTHORITY.read_text()
        ) != authority:
            raise RuntimeError("validation terminal input replay failed before result")
        collector.create_json(RESULTS, result)
        result_hash = file_sha256(RESULTS)
        receipt = {
            "schema": "block3_native_gate_subset_v1_validation_v1_receipt",
            "status": "validation_v1_complete_receipt_last",
            "authority_sha256": authority["authority_sha256"],
            "authority_file_sha256": file_sha256(AUTHORITY),
            "results_file_sha256": result_hash,
            "source_closure_sha256": source["sha256"],
            "fit_authority_sha256": fit["fit_authority_sha256"],
            "v0_failure_lineage": dict(v0_lineage),
            "row_file_sha256": ROWS_FILE_SHA256,
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "model_state_sha256": model_before,
            "call_ledger": calls.receipt(),
            "opened_k512": opened_512,
            "validation_eligible_budget": selected,
            "registered_next_action": next_action,
            "elapsed_seconds": time.time() - started,
        }
        claim.verify()
        verify_inputs(source, fit, rows_binding, checkpoint, v0_lineage)
        if file_sha256(RESULTS) != result_hash or model_state_sha256(model) != model_before or (
            json.loads(AUTHORITY.read_text()) != authority
        ):
            raise RuntimeError("validation terminal integrity replay failed")
        collector.create_json(RECEIPT, receipt)
        return result
    except BaseException as error:
        if not FAILURE.exists() and not RECEIPT.exists():
            try:
                claim.verify()
                collector.create_json(FAILURE, {
                    "schema": "block3_native_gate_subset_v1_validation_v1_failure",
                    "error_type": type(error).__name__, "error": str(error),
                    "authority_exists": AUTHORITY.exists(),
                    "results_exists": RESULTS.exists(), "receipt_exists": RECEIPT.exists(),
                    "elapsed_seconds": time.time() - started,
                })
            except BaseException:
                pass
        raise
    finally:
        claim.release()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
