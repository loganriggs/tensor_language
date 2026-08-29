#!/usr/bin/env python3
"""Physical fresh-row composition cross of frozen MLP0-C512 and MLP2-FULL512."""

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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for source_root in (ROOT, HERE, BQ):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
from mlp0_native_down_program import load_program
import prepare_mlp0_c512_mlp2_full512_composition_v1_rows as row_life
import run_mlp2_rank512_refit_v1 as refit

PREREG = HERE / "MLP0_C512_MLP2_FULL512_COMPOSITION_V1_PREREGISTRATION.md"
RUNNER = Path(__file__).resolve()
FREEZER = HERE / "prepare_mlp0_c512_mlp2_full512_composition_v1_rows.py"
TEST = HERE / "test_mlp0_c512_mlp2_full512_composition_v1.py"
AUDIT = HERE / "mlp0_c512_mlp2_full512_composition_v1_independent_audit.json"
SOURCE_PATHS = row_life.SOURCE_PATHS

ROWS_RECEIPT = BQ / "mlp0_c512_mlp2_full512_composition_v1_rows_receipt.json"
C512_PATH = BQ / "mlp0_native_down_hierarchy_v1_programs/C512_at_C512.bin"
C512_RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_fit_receipt.json"
FULL_BUNDLE = HERE / "mlp2_rank512_refit_v2_recovery_bundle.pt"
FULL_RECEIPT = HERE / "mlp2_rank512_refit_v2_recovery_receipt.json"
C512_SHA = "3ecf43b485d343bc5413e817dbd4236e5ce6cdaa7a3e0e653214e812b84ce470"
C512_RECEIPT_SHA = "79d0069864e9df521a99fc36531dd86c7ed31106f58f029d681fb1788a269f82"
FULL_BUNDLE_SHA = "d0ad8aedcfec5097e2791d64281f5cc4b644af450968456fc64dc7312123078e"
FULL_RECEIPT_SHA = "3578a68b4e8c20ea95f55a62cf9ff4e59e628bd69dbbad995f17a20f5265a7b2"

AUTHORITY = HERE / "mlp0_c512_mlp2_full512_composition_v1_authority.json"
LEDGER = HERE / "mlp0_c512_mlp2_full512_composition_v1_ledger.pt"
RESULT = HERE / "mlp0_c512_mlp2_full512_composition_v1_result.json"
RECEIPT = HERE / "mlp0_c512_mlp2_full512_composition_v1_receipt.json"
FAILURE = HERE / "mlp0_c512_mlp2_full512_composition_v1_failure.json"
LOCK = Path("/workspace/runs/.mlp0_c512_mlp2_full512_composition_v1.lock")

ARMS = ("NATIVE", "C512", "FULL512", "BOTH")
SCORING = slice(64, 256)
BOOTSTRAPS = 10_000
SEED = 2026082931


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def committed_sources() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    hashes = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        hashes[relative] = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != hashes[relative]:
            raise RuntimeError(f"uncommitted composition source: {relative}")
    return commit, hashes


def atomic_json(path: Path, value: Any) -> None:
    refit.atomic_json(path, value)


def atomic_torch(path: Path, value: Any) -> None:
    refit.atomic_torch(path, value)


def validate_parents() -> dict[str, str]:
    expected = {
        str(C512_PATH): C512_SHA, str(C512_RECEIPT): C512_RECEIPT_SHA,
        str(FULL_BUNDLE): FULL_BUNDLE_SHA, str(FULL_RECEIPT): FULL_RECEIPT_SHA,
    }
    for path, digest in expected.items():
        if file_sha256(Path(path)) != digest:
            raise RuntimeError(f"frozen composition parent changed: {path}")
    c_receipt = json.loads(C512_RECEIPT.read_text())
    if c_receipt["programs"]["C512_at_C512"]["sha256"] != C512_SHA:
        raise RuntimeError("C512 parent chain changed")
    f_receipt = json.loads(FULL_RECEIPT.read_text())
    if f_receipt.get("bundle_sha256") != FULL_BUNDLE_SHA or (
        f_receipt.get("status") != "result_complete_receipt_last"
    ):
        raise RuntimeError("FULL512 parent chain changed")
    return expected


def c512_write(event: facade.EarlyMLPEvent, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    mlp = event.block.mlp
    hidden = mlp.Left(event.state) * mlp.Right(event.state)
    latent = F.linear(hidden, tensors["right"].to(hidden.dtype))
    down = F.linear(latent, tensors["left"].to(hidden.dtype),
                    tensors["intercept"].to(hidden.dtype))
    return down + mlp.Down_bias.to(down.dtype)


def interaction_from_ledgers(ledgers: dict[str, torch.Tensor]) -> dict[str, Any]:
    for arm in ARMS:
        if ledgers[arm].shape != (192, 9):
            raise ValueError("composition ledger shape changed")
    def doc_dce(arm: str) -> torch.Tensor:
        x = ledgers[arm]
        return (x[:, 1] - x[:, 0]) / x[:, 8]
    c, f, b = (doc_dce(arm) for arm in ("C512", "FULL512", "BOTH"))
    interaction = b - c - f
    generator = torch.Generator().manual_seed(SEED)
    index = torch.randint(0, 192, (BOOTSTRAPS, 192), generator=generator)
    draws = interaction[index].mean(1)
    return {
        "interaction_dce": float(interaction.mean()),
        "interaction_ci95": [float(torch.quantile(draws, 0.025)),
                              float(torch.quantile(draws, 0.975))],
        "full_marginal_given_c512": float((b - c).mean()),
        "c512_marginal_given_full": float((b - f).mean()),
        "bootstrap_draws": BOOTSTRAPS,
    }


def summarize(ledger: torch.Tensor, prefix: int) -> dict[str, float]:
    return refit.summarize(ledger, prefix)


def derive_result(ledgers: dict[str, torch.Tensor], runtime: float) -> dict[str, Any]:
    summaries = {arm: {str(p): summarize(value, p) for p in (48, 96, 192)}
                 for arm, value in ledgers.items()}
    interaction = interaction_from_ledgers(ledgers)
    both, full = summaries["BOTH"]["192"], summaries["FULL512"]["192"]
    ci = interaction["interaction_ci95"]
    stability = all(abs(both[k] - summaries["BOTH"]["96"][k]) <= 0.01
                    for k in ("dce", "teacher_kl"))
    gates = {
        "both_dce_within_full_plus_0p01": both["dce"] <= full["dce"] + 0.01,
        "both_kl_within_full_plus_0p01": both["teacher_kl"] <= full["teacher_kl"] + 0.01,
        "abs_interaction_at_most_0p01": abs(interaction["interaction_dce"]) <= 0.01,
        "interaction_ci_inside_pm0p02": ci[0] >= -0.02 and ci[1] <= 0.02,
        "prefix_stability": stability,
    }
    if ci[1] < 0:
        label = "positive_synergy"
    elif ci[0] > 0.01:
        label = "incompatibility"
    else:
        label = "interaction_inconclusive"
    return {
        "schema": "mlp0_c512_mlp2_full512_composition_v1_result",
        "status": "composition_compatible" if all(gates.values()) else label,
        "claim_boundary": "in_distribution_frozen_parent_composition_no_strict_ledger_move",
        "documents": 192, "runtime_seconds": runtime,
        "summaries": summaries, "interaction": interaction,
        "composition_gates": gates,
    }


def main() -> None:
    if any(p.exists() for p in (AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("composition namespace already exists")
    claim = row_life.base.acquire_claim(LOCK)
    authority = None
    try:
        commit, sources = committed_sources()
        audit, audit_sha = row_life.validate_independent_audit(sources)
        parents = validate_parents()
        row_receipt_sha = file_sha256(ROWS_RECEIPT)
        row_receipt = json.loads(ROWS_RECEIPT.read_text())
        entry = row_receipt["entries"]["EVALUATION"]
        authority = {
            "schema": "mlp0_c512_mlp2_full512_composition_v1_authority",
            "status": "frozen_before_evaluation_open",
            "source_commit": commit, "source_hashes": sources,
            "audit_sha256": audit_sha, "audit_reviewer": audit["reviewer"],
            "parents": parents, "row_receipt_sha256": row_receipt_sha,
            "evaluation_rows_sha256": entry["file_sha256"],
            "arms": list(ARMS), "scored_slice": [64, 256],
            "outcome_access": False,
        }
        atomic_json(AUTHORITY, authority)
        started = time.time()
        rows = torch.load(entry["path"], map_location="cpu", weights_only=True)
        if tuple(rows.shape) != (192, 257) or refit.row_life.tensor_sha256(rows) != entry["tensor_sha256"]:
            raise RuntimeError("fresh evaluation rows changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        c = load_program(C512_PATH)
        c_tensors = {k: c[k].to(device) for k in ("intercept", "left", "right")}
        bundle = torch.load(FULL_BUNDLE, map_location="cpu", weights_only=True)
        full = refit.build_from_state(bundle["programs"]["FULL512"], device).eval()
        ledgers = {arm: [] for arm in ARMS}
        calls = {arm: {"outer": 0, "c512": 0, "full512": 0,
                       "native_mlp0": 0, "native_mlp2": 0} for arm in ARMS}
        with torch.inference_mode():
            for start in range(0, 192, 4):
                batch = rows[start:start + 4]
                tokens, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
                logits = {}
                for arm in ARMS:
                    def attention(event: facade.AttentionEvent):
                        return event.block.attn(event.state, event.first_value)
                    def mlp(event: facade.EarlyMLPEvent, arm=arm):
                        if event.site == 0:
                            if arm in ("C512", "BOTH"):
                                calls[arm]["c512"] += 1
                                return c512_write(event, c_tensors)
                            calls[arm]["native_mlp0"] += 1
                        if event.site == 2:
                            if arm in ("FULL512", "BOTH"):
                                calls[arm]["full512"] += 1
                                return full(event.state)
                            calls[arm]["native_mlp2"] += 1
                        return event.block.mlp(event.state)
                    calls[arm]["outer"] += 1
                    logits[arm] = facade.forward_with_dispatch(model, tokens, attention, mlp)
                native = logits["NATIVE"]
                for arm in ARMS:
                    ledgers[arm].append(refit.reduce_document(native, logits[arm], targets))
        packed = {arm: torch.cat(parts) for arm, parts in ledgers.items()}
        expected = {
            "NATIVE": {"outer": 48, "c512": 0, "full512": 0,
                       "native_mlp0": 48, "native_mlp2": 48},
            "C512": {"outer": 48, "c512": 48, "full512": 0,
                     "native_mlp0": 0, "native_mlp2": 48},
            "FULL512": {"outer": 48, "c512": 0, "full512": 48,
                        "native_mlp0": 48, "native_mlp2": 0},
            "BOTH": {"outer": 48, "c512": 48, "full512": 48,
                     "native_mlp0": 0, "native_mlp2": 0},
        }
        if calls != expected:
            raise RuntimeError(f"composition call census changed: {calls}")
        ledger = {"schema": "mlp0_c512_mlp2_full512_composition_v1_ledger",
                  "arms": packed, "calls": calls, "authority_sha256": file_sha256(AUTHORITY),
                  "checkpoint": checkpoint.__dict__}
        atomic_torch(LEDGER, ledger)
        result = derive_result(packed, time.time() - started)
        result["parents"] = {"authority": file_sha256(AUTHORITY),
                             "ledger": file_sha256(LEDGER)}
        atomic_json(RESULT, result)
        receipt = {"schema": "mlp0_c512_mlp2_full512_composition_v1_receipt",
                   "status": "result_complete_receipt_last",
                   "authority_sha256": file_sha256(AUTHORITY),
                   "ledger_sha256": file_sha256(LEDGER),
                   "result_sha256": file_sha256(RESULT),
                   "evaluation_opened": True}
        atomic_json(RECEIPT, receipt)
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        failure = {"schema": "mlp0_c512_mlp2_full512_composition_v1_failure",
                   "status": "terminal_failure_no_receipt", "error": repr(exc),
                   "authority_exists": AUTHORITY.exists(), "evaluation_may_have_opened": LEDGER.exists()}
        if not RECEIPT.exists() and not FAILURE.exists():
            atomic_json(FAILURE, failure)
        raise
    finally:
        row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
