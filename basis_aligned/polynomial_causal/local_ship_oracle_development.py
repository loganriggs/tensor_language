#!/usr/bin/env python3
"""Run the complete MLP0--2 oracle on document-disjoint local census rows.

This is real within-corpus causal evidence and a full plumbing test, but it is
deliberately nonauthoritative.  It never consumes or writes the canonical FineWeb
receipt/result/state (existence and hashes are guarded), never creates a training
license, and never invokes code OOD.  Fresh-corpus conclusions remain reserved for
``frozen_ship_oracle_v2.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
CORPUS = BQ / "curated_rows.pt"
CORPUS_SHA256 = "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd"
FACTORS = HERE / "content_product_frontier_factors.pt"
FACTORS_SHA256 = "2c587ac3f92375ee54332799bd0ee1dc6637fd9e6e1c0298d3ea115f49f6c380"
GLUE = BQ / "mlp2_glue_params.pt"
GLUE_SHA256 = "76148b072c22f3c0d0ccdcaa08d8a6ade89d7231d0dd5a328597e10f6a0a3ef4"
MODEL_SNAPSHOT = Path("/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240")
MODEL_CONFIG_SHA256 = "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c"
MODEL_WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
DEV_RESULT = BQ / "ship_content_oracle_curated_dev_v2_results.json"
DEV_PREREG = BQ / "ship_content_oracle_curated_dev_v2_preregistration.json"
DEV_MANIFEST = BQ / "ship_content_oracle_curated_dev_v2_manifest.json"
DEV_STATE = Path("/workspace/runs/bilin18_curated_dev_v2_ship.pt")
DEV_ORACLE_STATE = Path("/workspace/runs/bilin18_curated_dev_v2_oracle_bases.pt")
DEV_LOCK = BQ / ".ship_content_oracle_curated_dev_v2.lock"
SHIP_SEED = 27182818
SOURCE_ROW_LEN = 513
MODEL_ROW_LEN = 257
REQUEST_ROLES = {
    (96, 80): "covariance",
    (480, 80): "ship_fit",
    (96, 1200): "basis",
    (192, 7000): "discovery",
    (192, 11000): "heldout",
}
CANONICAL_PATHS = (
    BQ / "ship_content_oracle_screen_results.json",
    BQ / "ship_content_oracle_screen_preliminary_results.json",
    BQ / ".rowcache/fineweb_oracle_v2_receipt.json",
    Path("/workspace/runs/bilin18_frozen_ship_v2.pt"),
    Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json"),
    HERE / "code_ood_oracle_results.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def path_snapshot(paths: tuple[Path, ...] = CANONICAL_PATHS) -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.exists() else None for path in paths}


def allocate_whole_document_splits(payload: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Exact deterministic 480/96/192/192 bin packing with no document leakage."""
    if not isinstance(payload, dict) or set(payload) < {"rows", "docid"}:
        raise RuntimeError("curated corpus must contain rows and docid tensors")
    rows, docids = payload["rows"], payload["docid"]
    if tuple(rows.shape) != (1000, SOURCE_ROW_LEN) or rows.dtype != torch.long:
        raise RuntimeError(f"invalid curated rows: {rows.shape} {rows.dtype}")
    if tuple(docids.shape) != (1000,) or docids.dtype != torch.long:
        raise RuntimeError(f"invalid curated docids: {docids.shape} {docids.dtype}")
    grouped: dict[int, list[int]] = {}
    for index, document in enumerate(docids.tolist()):
        grouped.setdefault(int(document), []).append(index)
    singles = sorted(document for document, indices in grouped.items() if len(indices) == 1)
    doubles = sorted(document for document, indices in grouped.items() if len(indices) == 2)
    if len(singles) != 376 or len(doubles) != 312:
        raise RuntimeError(
            f"curated document census changed: {len(singles)} singles, {len(doubles)} doubles"
        )
    documents = {
        "ship_fit": doubles[:240],
        "basis": doubles[240:288],
        "discovery": doubles[288:312] + singles[:144],
        "heldout": singles[144:336],
        "spare": singles[336:],
    }
    expected = {"ship_fit": 480, "basis": 96, "discovery": 192,
                "heldout": 192, "spare": 40}
    output = {}
    seen_documents: set[int] = set()
    for role, role_documents in documents.items():
        if seen_documents.intersection(role_documents):
            raise RuntimeError(f"document leakage into {role}")
        seen_documents.update(role_documents)
        indices = sorted(index for document in role_documents for index in grouped[document])
        if len(indices) != expected[role]:
            raise RuntimeError(f"{role} has {len(indices)} rows, expected {expected[role]}")
        tensor = rows[indices, :MODEL_ROW_LEN].contiguous()
        output[role] = {
            "rows": tensor,
            "indices": indices,
            "document_ids": role_documents,
            "tensor_raw_sha256": tensor_sha256(tensor),
        }
    causal_roles = ("ship_fit", "basis", "discovery", "heldout")
    for left_index, left in enumerate(causal_roles):
        left_rows = output[left]["rows"]
        left_full = {tensor_sha256(row) for row in left_rows}
        left_prefix = {tuple(row[:32].tolist()) for row in left_rows}
        for right in causal_roles[left_index + 1:]:
            right_rows = output[right]["rows"]
            if left_full.intersection(tensor_sha256(row) for row in right_rows):
                raise RuntimeError(f"full-row content overlap between {left} and {right}")
            if left_prefix.intersection(tuple(row[:32].tolist()) for row in right_rows):
                raise RuntimeError(f"prefix-32 content overlap between {left} and {right}")
    output["covariance"] = {
        "rows": output["ship_fit"]["rows"][:96].clone(),
        "indices": output["ship_fit"]["indices"][:96],
        "document_ids": sorted(set(docids[output["ship_fit"]["indices"][:96]].tolist())),
        "tensor_raw_sha256": tensor_sha256(output["ship_fit"]["rows"][:96]),
    }
    return output


def manifest_for_splits(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_kind": "curated_ship_oracle_development_v2",
        "status": "preregistered_not_run",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "training_license_sites": [],
        "scope_guardrail": "Exploratory plumbing and within-run intervention signs only. The frozen content factor lacks source-document provenance, so even document-disjoint curated evaluation is not internally held out for the content-basis claim; no FineWeb, fresh-corpus, code-OOD, training, or generalization claim.",
        "factor_provenance_status": "source document ids absent; overlap with curated rows unprovable",
        "seed": SHIP_SEED,
        "corpus": {"path": str(CORPUS.resolve()), "sha256": file_sha256(CORPUS)},
        "row_splits": {
            role: {
                "shape": list(row["rows"].shape),
                "dtype": str(row["rows"].dtype),
                "tensor_raw_sha256": row["tensor_raw_sha256"],
                "indices": row["indices"],
                "document_ids": row["document_ids"],
            }
            for role, row in splits.items()
        },
        "request_roles": {f"n{n}_skip{skip}": role for (n, skip), role in REQUEST_ROLES.items()},
        "source_hashes": {
            "runner": file_sha256(Path(__file__)),
            "ship_error_attrib": file_sha256(BQ / "ship_error_attrib.py"),
            "frozen_ship_oracle_v2": file_sha256(HERE / "frozen_ship_oracle_v2.py"),
            "content_factors": file_sha256(FACTORS),
            "mlp2_glue": file_sha256(GLUE),
            "model_loader": file_sha256(BQ / "bilin18_joint_removal.py"),
            "tier2_model": file_sha256(HERE.parent / "qk_mdl/tier2_model.py"),
            "model_config": file_sha256(MODEL_SNAPSHOT / "config.json"),
            "model_weights": file_sha256(MODEL_SNAPSHOT / "pytorch_model.bin"),
        },
        "canonical_paths_before": path_snapshot(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
    }


def exact_development_decisions(result: dict[str, Any], exact_null_test) -> list[int]:
    candidates = []
    for site in (0, 1, 2):
        key = str(site)
        heldout = result["paired_gains"][key]["heldout"]
        content_gain = heldout["content"]["global"]["mean"]
        null_gains = [heldout[f"null_{index:02d}"]["global"]["mean"] for index in range(20)]
        if not math.isfinite(content_gain) or not all(math.isfinite(value) for value in null_gains):
            raise RuntimeError(f"nonfinite exact-null input at site {site}")
        exact = exact_null_test(content_gain, null_gains)
        result["site_decisions"][key]["exact_twenty_null_test"] = exact
        decision = result["site_decisions"][key]
        if (decision["full_oracle_ci95_lower_gt_zero"]
                and decision["content_positive_both_splits"]
                and exact["passes_5pct"]):
            candidates.append(site)
    result["development_candidate_sites"] = candidates
    result["training_license_sites"] = []
    return candidates


def run_claimed(canonical_before: dict[str, str | None]) -> None:
    pinned = {
        CORPUS: CORPUS_SHA256,
        FACTORS: FACTORS_SHA256,
        GLUE: GLUE_SHA256,
        MODEL_SNAPSHOT / "config.json": MODEL_CONFIG_SHA256,
        MODEL_SNAPSHOT / "pytorch_model.bin": MODEL_WEIGHTS_SHA256,
    }
    for path, expected_hash in pinned.items():
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"pinned input hash mismatch for {path}: {observed_hash}")
    payload = torch.load(CORPUS, map_location="cpu", weights_only=True)
    splits = allocate_whole_document_splits(payload)
    preregistration = manifest_for_splits(splits)
    write_json_atomic(preregistration, DEV_PREREG)
    preregistration_hash = file_sha256(DEV_PREREG)
    manifest = {
        **preregistration,
        "preregistration_path": str(DEV_PREREG.resolve()),
        "preregistration_sha256": preregistration_hash,
    }
    write_json_atomic(manifest, DEV_MANIFEST)
    if manifest["canonical_paths_before"] != canonical_before:
        raise RuntimeError("canonical oracle artifacts changed during preregistration")

    sys.path.insert(0, str(BQ))
    sys.path.insert(0, str(HERE))
    import code_ood_oracle
    import frozen_ship_oracle_v2 as frozen
    import ship_error_attrib as sa
    import source_global_preflight

    source_global_preflight.require_defined_globals([
        BQ / "ship_error_attrib.py", Path(__file__), HERE / "frozen_ship_oracle_v2.py",
    ])

    def frozen_local_rows(n: int = 120, skip: int = 0) -> torch.Tensor:
        role = REQUEST_ROLES.get((n, skip))
        if role is None:
            raise RuntimeError(f"unregistered curated row request {(n, skip)}")
        return splits[role]["rows"].clone()

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sa.cl.fineweb_rows = frozen_local_rows
    original_callback = sa.run_oracle_content_screen
    start_time = time.time()

    def development_callback(twall: dict, all_attention: frozenset[int], callback_start: float) -> None:
        state = frozen.cpu_tree({
            "schema_version": 1,
            "status": "exploratory_plumbing_only_local_curated",
            "seed": SHIP_SEED,
            "rng_state_cpu_before_oracle": torch.get_rng_state(),
            "rng_state_cuda_before_oracle": torch.cuda.get_rng_state_all(),
            "ship": sa.SHIP,
            "corr": sa.CORR,
            "attention": twall,
            "all_attention": sorted(all_attention),
        })
        frozen.atomic_torch_save(state, DEV_STATE)
        manifest.update({
            "status": "ship_frozen_oracle_running",
            "ship_state_path": str(DEV_STATE.resolve()),
            "ship_state_file_sha256": file_sha256(DEV_STATE),
            "ship_state_tree_sha256": code_ood_oracle.tensor_tree_sha256(state),
            "baseline_fingerprint": {
                key: value for key, value in frozen.baseline_fingerprint(
                    sa, twall, all_attention, splits["heldout"]["rows"]
                ).items() if key != "sample_logits"
            },
        })
        write_json_atomic(manifest, DEV_MANIFEST)
        original_callback(
            twall, all_attention, callback_start,
            row_sets={role: splits[role]["rows"] for role in ("basis", "discovery", "heldout")},
            output_path=DEV_RESULT,
            authority="none",
            realization_path=DEV_ORACLE_STATE,
        )
        result = json.loads(DEV_RESULT.read_text())
        candidates = exact_development_decisions(result, code_ood_oracle.exact_null_test)
        result["config"].update({
            "status": "exploratory_plumbing_only_local_curated",
            "authority": "none",
            "authorized_for_scored_experiments": False,
            "preregistration_receipt_sha256": preregistration_hash,
            "preregistration_path": str(DEV_PREREG.resolve()),
            "ship_state_file_sha256": manifest["ship_state_file_sha256"],
            "ship_state_tree_sha256": manifest["ship_state_tree_sha256"],
            "oracle_realization_path": str(DEV_ORACLE_STATE.resolve()),
            "oracle_realization_sha256": file_sha256(DEV_ORACLE_STATE),
            "null_gate": "exact one-sided Monte Carlo; content beats all 20",
        })
        result["interpretation_guardrail"] = manifest["scope_guardrail"]
        write_json_atomic(result, DEV_RESULT)
        manifest.update({
            "status": "completed_exploratory_only",
            "development_candidate_sites": candidates,
            "training_license_sites": [],
            "result_path": str(DEV_RESULT.resolve()),
            "result_sha256": file_sha256(DEV_RESULT),
            "oracle_realization_path": str(DEV_ORACLE_STATE.resolve()),
            "oracle_realization_sha256": file_sha256(DEV_ORACLE_STATE),
            "runtime_s": round(time.time() - start_time, 1),
            "preregistration_receipt_sha256": preregistration_hash,
            "preregistration_path": str(DEV_PREREG.resolve()),
        })
        write_json_atomic(manifest, DEV_MANIFEST)

    sa.run_oracle_content_screen = development_callback
    sa.main(oracle_content_screen=True)
    print(json.dumps({
        "status": manifest["status"],
        "development_candidate_sites": manifest.get("development_candidate_sites", []),
        "training_license_sites": [],
        "result": str(DEV_RESULT),
    }, indent=2), flush=True)


def mark_failed(status: str, error: BaseException, canonical_after=None) -> None:
    if DEV_MANIFEST.exists():
        try:
            manifest = json.loads(DEV_MANIFEST.read_text())
        except Exception:
            manifest = {"schema_version": 1}
    else:
        manifest = {"schema_version": 1}
    manifest.update({
        "status": status,
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "training_license_sites": [],
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "recovery": "Preserve all artifacts; diagnose the recorded failure, then use a new versioned output namespace rather than deleting or overwriting this run.",
    })
    if canonical_after is not None:
        manifest["canonical_paths_after"] = canonical_after
    write_json_atomic(manifest, DEV_MANIFEST)
    if DEV_RESULT.exists():
        try:
            result = json.loads(DEV_RESULT.read_text())
        except Exception:
            result = {"config": {}}
        result.setdefault("config", {}).update({
            "status": status,
            "authority": "none",
            "authorized_for_scored_experiments": False,
        })
        result["training_license_sites"] = []
        result["invalidated_by_guard"] = True
        write_json_atomic(result, DEV_RESULT)


def main() -> None:
    if any(path.exists() for path in (
        DEV_RESULT, DEV_PREREG, DEV_MANIFEST, DEV_STATE, DEV_ORACLE_STATE,
    )):
        raise RuntimeError("refusing to overwrite an existing curated development artifact")
    try:
        DEV_LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"curated development launch is already claimed or crashed: {DEV_LOCK}; "
            "inspect before any manual recovery"
        ) from error
    canonical_before = path_snapshot()
    run_error = None
    try:
        run_claimed(canonical_before)
    except BaseException as error:
        run_error = error
    canonical_after = path_snapshot()
    try:
        if canonical_after != canonical_before:
            contamination = RuntimeError(
                "curated development run changed a canonical oracle artifact"
            )
            mark_failed("invalid_canonical_contamination", contamination, canonical_after)
            raise contamination from run_error
        if run_error is not None:
            mark_failed("failed_exploratory_run", run_error)
            raise run_error
    finally:
        DEV_LOCK.rmdir()


if __name__ == "__main__":
    main()
