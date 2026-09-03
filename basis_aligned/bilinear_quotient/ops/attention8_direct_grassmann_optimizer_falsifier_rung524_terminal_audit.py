#!/usr/bin/env python3
"""Independently audit rung 524's fail-closed planted-optimizer result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import torch


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
REPO = ROOT.parent.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import attention8_direct_grassmann_optimizer_falsifier_rung524_math as gm  # noqa: E402
import attention8_direct_grassmann_optimizer_falsifier_rung524_run as run  # noqa: E402


DEFAULT_RESULT = ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_results.json"
DEFAULT_ARCHIVE = ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_frames.pt"
DEFAULT_OUTPUT = ROOT / "attention8_direct_grassmann_optimizer_falsifier_rung524_terminal_audit.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def audit_terminal_result(
    result: Mapping[str, object],
    artifact: Mapping[str, object],
    *,
    artifact_file_sha256: str,
) -> dict[str, object]:
    if result.get("rung") != 524 or result.get("status") != "direct_subspace_instrument_falsified":
        raise ValueError("unexpected rung or terminal status")
    if result.get("claim_level") != "planted optimizer-instrument test; no circuit evidence":
        raise ValueError("claim level changed")
    receipt = result.get("frame_archive")
    if not isinstance(receipt, Mapping):
        raise ValueError("frame receipt is absent")
    if receipt.get("file_sha256") != artifact_file_sha256 or receipt.get("frame_count") != 15:
        raise ValueError("frame receipt differs from archive")
    if artifact.get("schema") != "attention8-direct-grassmann-rung524-frames-v1":
        raise ValueError("archive schema changed")
    frames = artifact.get("frames")
    records = artifact.get("records")
    planted = artifact.get("planted_frame")
    if not isinstance(frames, Mapping) or not isinstance(records, list) or len(records) != 15:
        raise ValueError("record census changed")
    if not isinstance(planted, torch.Tensor) or planted.shape != (gm.D, gm.RANK):
        raise ValueError("planted frame changed")
    if _json_value(records) != _json_value(result.get("records")):
        raise ValueError("archive/result records differ")

    pretest_fits = []
    accepted = 0
    failed = 0
    expected_line_search = 0
    for record in records:
        key = f"omit{record['omitted_target']}:seed{record['seed']}"
        frame = frames.get(key)
        if not isinstance(frame, torch.Tensor) or frame.shape != (gm.D, gm.RANK):
            raise ValueError(f"invalid frame {key}")
        if run._tensor_sha256(frame) != record.get("frame_sha256"):
            raise ValueError(f"frame hash differs for {key}")
        if abs(gm.projector_error(frame, planted) - float(record["projector_error"])) > 1e-12:
            raise ValueError(f"projector score differs for {key}")
        if abs(gm.minimum_principal_cosine(frame, planted) - float(record["minimum_principal_cosine"])) > 1e-12:
            raise ValueError(f"principal-cosine score differs for {key}")
        fit_record = run.FitRecord(**record)
        pretest_fits.append(fit_record.pretest())
        accepted += int(record["accepted_updates"])
        failed += int(record["failed_update"] is not None)
        expected_line_search += sum(int(value) + 1 for value in record["backtracks_per_update"])
        if record["failed_update"] is not None:
            expected_line_search += gm.MAX_BACKTRACKS + 1
    if set(frames) != {
        f"omit{record['omitted_target']}:seed{record['seed']}" for record in records
    }:
        raise ValueError("frame key census differs")

    pretest = gm.score_pretest(pretest_fits)
    if _json_value(pretest) != _json_value(result.get("pretest")):
        raise ValueError("pretest score differs from records")
    if pretest["pretest_passes"] or pretest["passing_fit_count"] != 0:
        raise ValueError("expected planted falsification boundary changed")
    if result.get("ood") is not None or result.get("ood_losses") is not None:
        raise ValueError("OOD was evaluated after pretest failure")
    seal = result.get("seal")
    if seal != {"requested_splits": ["FIT", "VALIDATION"], "ood_opened": False}:
        raise ValueError("OOD seal changed")
    decision = gm.final_decision(pretest, None)
    if decision != result.get("decision"):
        raise ValueError("decision differs from recomputed score")

    price = result.get("execution_price")
    if not isinstance(price, Mapping):
        raise ValueError("execution price is absent")
    expected_gradient = accepted + failed
    expected_counts = {
        "fit_evaluation": 30,
        "validation_evaluation": 30,
        "fit_gradient_evaluation": expected_gradient,
        "gradient_evaluations": expected_gradient,
        "line_search_evaluation": expected_line_search,
        "model_forwards": 0,
        "model_backwards": 0,
        "gpu_calls": 0,
    }
    for name, expected in expected_counts.items():
        if price.get(name) != expected:
            raise ValueError(f"execution count differs for {name}")
    return {
        "schema": "rung524-terminal-audit-v1",
        "passes": True,
        "status": result["status"],
        "frame_archive_file_sha256": artifact_file_sha256,
        "frame_count": len(frames),
        "passing_fit_count": 0,
        "accepted_updates": accepted,
        "failed_line_search_count": failed,
        "ood_opened": False,
        "exact_execution_counts": expected_counts,
        "decision": decision,
        "interpretation": (
            "direct Grassmann optimization failed the known planted problem; close the "
            "attention8 optimizer route and pivot to exact MLP0 branch decomposition"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.result.open(encoding="utf-8") as source:
        result = json.load(source)
    for relative, expected in result["dependency_sha256"].items():
        if _file_sha256(REPO / relative) != expected:
            raise ValueError(f"dependency changed: {relative}")
    if _file_sha256(OPS / "attention8_direct_grassmann_optimizer_falsifier_rung524_run.py") != result["runner_sha256"]:
        raise ValueError("runner changed")
    artifact_sha = _file_sha256(args.archive)
    artifact = torch.load(args.archive, map_location="cpu", weights_only=False)
    audit = {
        "result_file_sha256": _file_sha256(args.result),
        **audit_terminal_result(result, artifact, artifact_file_sha256=artifact_sha),
    }
    _atomic_json(args.output, audit)
    print(json.dumps({
        "output": str(args.output), "passes": True,
        "passing_fit_count": 0, "ood_opened": False,
        "next_action": audit["decision"]["next_action"],
    }, indent=2, sort_keys=True), flush=True)
    return audit


if __name__ == "__main__":
    main()
