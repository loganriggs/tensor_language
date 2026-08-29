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


def verify_sources(commit: str, expected: dict[str, str]) -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    if set(expected) != {str(path.relative_to(ROOT)) for path in SOURCE_PATHS}:
        raise RuntimeError("composition source family changed")
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected[relative] \
                or file_sha256(path) != expected[relative]:
            raise RuntimeError(f"composition source bytes changed: {relative}")


def atomic_json(path: Path, value: Any, *, pre_link_check=None) -> None:
    refit.atomic_json(path, value, pre_link_check=pre_link_check)


def atomic_torch(path: Path, value: Any, *, pre_link_check=None) -> None:
    refit.atomic_torch(path, value, pre_link_check=pre_link_check)


def stable_json(path: Path, expected: str | None = None) -> tuple[Any, str]:
    before = file_sha256(path)
    if expected is not None and before != expected:
        raise RuntimeError(f"JSON parent hash changed: {path}")
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON parent raced read: {path}")
    return json.loads(raw), before


def stable_torch(path: Path, expected: str | None = None) -> tuple[Any, str]:
    before = file_sha256(path)
    if expected is not None and before != expected:
        raise RuntimeError(f"tensor parent hash changed: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before:
        raise RuntimeError(f"tensor parent raced read: {path}")
    return value, before


def validate_parents() -> dict[str, str]:
    expected = {
        str(C512_PATH): C512_SHA, str(C512_RECEIPT): C512_RECEIPT_SHA,
        str(FULL_BUNDLE): FULL_BUNDLE_SHA, str(FULL_RECEIPT): FULL_RECEIPT_SHA,
    }
    c_receipt, _ = stable_json(C512_RECEIPT, C512_RECEIPT_SHA)
    f_receipt, _ = stable_json(FULL_RECEIPT, FULL_RECEIPT_SHA)
    stable_torch(FULL_BUNDLE, FULL_BUNDLE_SHA)
    if file_sha256(C512_PATH) != C512_SHA:
        raise RuntimeError("C512 frozen program changed")
    if c_receipt["programs"]["C512_at_C512"]["sha256"] != C512_SHA:
        raise RuntimeError("C512 parent chain changed")
    if f_receipt.get("bundle_sha256") != FULL_BUNDLE_SHA or (
        f_receipt.get("status") != "result_complete_receipt_last"
    ):
        raise RuntimeError("FULL512 parent chain changed")
    return expected


def validate_row_receipt(value: Any, sources: dict[str, str]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != (
        "mlp0_c512_mlp2_full512_composition_v1_rows"
    ) or value.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" \
            or value.get("source_hashes") != sources or value.get("outcome_access") != {
                "model_loaded": False, "training_run": False,
            } or value.get("selection") != {
                "start_document_index": 110000, "documents_per_role": 192,
                "token_length": 257, "scored_slice": [64, 256],
            } or value.get("roles") != {
                "TRAIN": {"authorized_for_training": True, "authorized_for_evaluation": False},
                "EVALUATION": {"authorized_for_training": False, "authorized_for_evaluation": True},
            } or set(value.get("entries", {})) != {"TRAIN", "EVALUATION"} \
            or not all(value.get("disjointness", {}).values()):
        raise RuntimeError("composition row receipt semantics changed")
    for role, entry in value["entries"].items():
        if entry.get("shape") != [192, 257] or entry.get("dtype") != "torch.int64" \
                or not Path(entry["path"]).is_file() \
                or file_sha256(Path(entry["path"])) != entry.get("file_sha256"):
            raise RuntimeError(f"composition {role} row entry changed")
    return value


def protected_snapshot(authority: dict[str, Any]) -> dict[str, Any]:
    commit, sources = authority["source_commit"], authority["source_hashes"]
    verify_sources(commit, sources)
    audit, audit_sha = row_life.validate_independent_audit(sources)
    rows, rows_sha = stable_json(ROWS_RECEIPT, authority["row_receipt_sha256"])
    validate_row_receipt(rows, sources)
    parents = validate_parents()
    if parents != authority["parents"] or audit_sha != authority["audit_sha256"]:
        raise RuntimeError("composition protected parent changed")
    snapshot = Path(facade.DEFAULT_SNAPSHOT)
    return {
        "source_commit": commit, "source_hashes": sources,
        "audit_sha256": audit_sha, "audit_commit": audit["audited_source_commit"],
        "row_receipt_sha256": rows_sha,
        "row_file_hashes": {role: file_sha256(Path(entry["path"]))
                            for role, entry in rows["entries"].items()},
        "parents": parents,
        "checkpoint_config_sha256": file_sha256(snapshot / "config.json"),
        "checkpoint_weights_sha256": file_sha256(snapshot / "pytorch_model.bin"),
        "checkpoint_weights_bytes": (snapshot / "pytorch_model.bin").stat().st_size,
    }


def verify_protected(expected: dict[str, Any], authority: dict[str, Any], claim) -> None:
    row_life.base.require_claim(claim, LOCK)
    if protected_snapshot(authority) != expected:
        raise RuntimeError("composition protected snapshot changed")
    row_life.base.require_claim(claim, LOCK)


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
    both = ledgers["BOTH"]
    both_dce = (both[:, 1] - both[:, 0]) / both[:, 8]
    both_kl = both[:, 2] / both[:, 8]
    return {
        "interaction_dce": float(interaction.mean()),
        "interaction_ci95": [float(torch.quantile(draws, 0.025)),
                              float(torch.quantile(draws, 0.975))],
        "full_marginal_given_c512": float((b - c).mean()),
        "c512_marginal_given_full": float((b - f).mean()),
        "both_dce_one_sided_95_ucb": float(torch.quantile(
            both_dce[index].mean(1), 0.95, interpolation="linear")),
        "both_kl_one_sided_95_ucb": float(torch.quantile(
            both_kl[index].mean(1), 0.95, interpolation="linear")),
        "bootstrap_draws": BOOTSTRAPS,
    }


def expected_call_census() -> dict[str, Any]:
    output = {}
    for arm in ARMS:
        native = {str(site): 48 for site in range(18)}
        if arm in ("C512", "BOTH"):
            native["0"] = 0
        if arm in ("FULL512", "BOTH"):
            native["2"] = 0
        output[arm] = {
            "outer_calls": 48, "outer_returns": 48,
            "attention_sites": {str(site): 48 for site in range(18)},
            "native_mlp_sites": native,
            "candidate_c512": 48 if arm in ("C512", "BOTH") else 0,
            "candidate_full512": 48 if arm in ("FULL512", "BOTH") else 0,
        }
    return output


def validate_ledger(value: Any, authority_sha: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or value.get("schema") != (
        "mlp0_c512_mlp2_full512_composition_v1_ledger"
    ) or value.get("authority_sha256") != authority_sha \
            or value.get("calls") != expected_call_census() \
            or set(value.get("arms", {})) != set(ARMS):
        raise RuntimeError("composition ledger metadata changed")
    arms = value["arms"]
    if any(not isinstance(x, torch.Tensor) or x.dtype != torch.float64
           or tuple(x.shape) != (192, 9) or not torch.isfinite(x).all()
           or (x[:, 8] != 192).any() for x in arms.values()):
        raise RuntimeError("composition sufficient statistics changed")
    return arms


def failure_terminal_guard(claim, artifact_hashes: dict[Path, str]) -> None:
    row_life.base.require_claim(claim, LOCK)
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("composition terminal raced failure")
    for path, digest in artifact_hashes.items():
        if file_sha256(path) != digest:
            raise RuntimeError("composition failure artifact changed")
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("composition terminal raced failure during artifact replay")
    row_life.base.require_claim(claim, LOCK)


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
    protected = None
    evaluation_opened = False
    try:
        commit, sources = committed_sources()
        audit, audit_sha = row_life.validate_independent_audit(sources)
        parents = validate_parents()
        row_receipt, row_receipt_sha = stable_json(ROWS_RECEIPT)
        validate_row_receipt(row_receipt, sources)
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

        def authority_guard() -> None:
            row_life.base.require_claim(claim, LOCK)
            verify_sources(commit, sources)
            row_life.validate_independent_audit(sources)
            stable_json(ROWS_RECEIPT, row_receipt_sha)
            if validate_parents() != parents or any(p.exists() for p in (
                AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE,
            )):
                raise RuntimeError("composition authority inputs or namespace changed")
            row_life.base.require_claim(claim, LOCK)
        atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(AUTHORITY)
        protected = protected_snapshot(authority)
        started = time.time()
        evaluation_opened = True
        rows, rows_sha = stable_torch(Path(entry["path"]), entry["file_sha256"])
        if rows_sha != authority["evaluation_rows_sha256"] \
                or tuple(rows.shape) != (192, 257) \
                or refit.row_life.tensor_sha256(rows) != entry["tensor_sha256"]:
            raise RuntimeError("fresh evaluation rows changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        verify_protected(protected, authority, claim)
        c = load_program(C512_PATH)
        c_tensors = {k: c[k].to(device) for k in ("intercept", "left", "right")}
        bundle, _ = stable_torch(FULL_BUNDLE, FULL_BUNDLE_SHA)
        full = refit.build_from_state(bundle["programs"]["FULL512"], device).eval()
        ledgers = {arm: [] for arm in ARMS}
        calls = {arm: {
            "outer_calls": 0, "outer_returns": 0,
            "attention_sites": {str(site): 0 for site in range(18)},
            "native_mlp_sites": {str(site): 0 for site in range(18)},
            "candidate_c512": 0, "candidate_full512": 0,
        } for arm in ARMS}
        with torch.inference_mode():
            for start in range(0, 192, 4):
                batch = rows[start:start + 4]
                tokens, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
                logits = {}
                for arm in ARMS:
                    def attention(event: facade.AttentionEvent, arm=arm):
                        calls[arm]["attention_sites"][str(event.site)] += 1
                        return event.block.attn(event.state, event.first_value)
                    def mlp(event: facade.EarlyMLPEvent, arm=arm):
                        if event.site == 0:
                            if arm in ("C512", "BOTH"):
                                calls[arm]["candidate_c512"] += 1
                                return c512_write(event, c_tensors)
                        if event.site == 2:
                            if arm in ("FULL512", "BOTH"):
                                calls[arm]["candidate_full512"] += 1
                                return full(event.state)
                        calls[arm]["native_mlp_sites"][str(event.site)] += 1
                        return event.block.mlp(event.state)
                    calls[arm]["outer_calls"] += 1
                    logits[arm] = facade.forward_with_dispatch(model, tokens, attention, mlp)
                    calls[arm]["outer_returns"] += 1
                native = logits["NATIVE"]
                for arm in ARMS:
                    ledgers[arm].append(refit.reduce_document(native, logits[arm], targets))
        packed = {arm: torch.cat(parts) for arm, parts in ledgers.items()}
        expected = expected_call_census()
        if calls != expected:
            raise RuntimeError(f"composition call census changed: {calls}")
        ledger = {"schema": "mlp0_c512_mlp2_full512_composition_v1_ledger",
                  "arms": packed, "calls": calls, "authority_sha256": authority_sha,
                  "checkpoint": checkpoint.__dict__}

        def ledger_guard() -> None:
            verify_protected(protected, authority, claim)
            if LEDGER.exists() or RESULT.exists() or RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("composition terminal raced ledger")
        atomic_torch(LEDGER, ledger, pre_link_check=ledger_guard)
        reloaded_ledger, ledger_sha = stable_torch(LEDGER)
        replay_arms = validate_ledger(reloaded_ledger, authority_sha)
        runtime = time.time() - started
        result = derive_result(replay_arms, runtime)
        result["parents"] = {"authority": authority_sha, "ledger": ledger_sha}

        def result_guard() -> None:
            verify_protected(protected, authority, claim)
            stable_torch(LEDGER, ledger_sha)
            if RESULT.exists() or RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("composition terminal raced result")
        atomic_json(RESULT, result, pre_link_check=result_guard)
        reloaded_result, result_sha = stable_json(RESULT)
        if reloaded_result != result or derive_result(replay_arms, runtime) != {
            key: value for key, value in result.items() if key != "parents"
        }:
            raise RuntimeError("composition result semantic replay changed")
        receipt = {"schema": "mlp0_c512_mlp2_full512_composition_v1_receipt",
                   "status": "result_complete_receipt_last",
                   "authority_sha256": authority_sha,
                   "ledger_sha256": ledger_sha,
                   "result_sha256": result_sha,
                   "evaluation_opened": True}

        def receipt_guard() -> None:
            verify_protected(protected, authority, claim)
            stable_json(AUTHORITY, authority_sha)
            stable_torch(LEDGER, ledger_sha)
            stable_json(RESULT, result_sha)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("composition terminal raced receipt")
            row_life.base.require_claim(claim, LOCK)
        atomic_json(RECEIPT, receipt, pre_link_check=receipt_guard)
        if stable_json(RECEIPT)[0] != receipt:
            raise RuntimeError("composition receipt replay changed")
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        failure = {"schema": "mlp0_c512_mlp2_full512_composition_v1_failure",
                   "status": "terminal_failure_no_receipt", "error": repr(exc),
                   "authority_exists": AUTHORITY.exists(),
                   "evaluation_may_have_opened": evaluation_opened,
                   "protected_snapshot": protected,
                   "artifact_hashes": {p.name: file_sha256(p) for p in (LEDGER, RESULT)
                                       if p.is_file()}}
        if not RECEIPT.exists() and not FAILURE.exists():
            frozen_artifacts = {HERE / name: digest
                                for name, digest in failure["artifact_hashes"].items()}
            def failure_guard() -> None:
                failure_terminal_guard(claim, frozen_artifacts)
            atomic_json(FAILURE, failure, pre_link_check=failure_guard)
        raise
    finally:
        row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
