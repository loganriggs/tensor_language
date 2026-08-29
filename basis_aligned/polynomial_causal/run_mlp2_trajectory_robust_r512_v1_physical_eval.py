#!/usr/bin/env python3
"""Fresh eight-arm physical evaluation of trajectory-robust MLP2 rank-512."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for source_root in (ROOT, HERE, BQ):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import prepare_mlp2_trajectory_robust_r512_v1_eval_rows as row_life
import run_mlp0_c512_mlp2_full512_composition_v1 as base
import run_mlp2_rank512_refit_v1 as refit
from mlp0_native_down_program import load_program

ROWS_RECEIPT = BQ / "mlp2_trajectory_robust_r512_v1_physical_eval_rows_receipt.json"
ROBUST_BUNDLE = HERE / "mlp2_trajectory_robust_r512_v1_fit_bundle.pt"
ROBUST_RECEIPT = HERE / "mlp2_trajectory_robust_r512_v1_fit_receipt.json"
ROBUST_BUNDLE_SHA = "79d13685a1e0f53aecc3ea1d34e0c332a55149bf6de510daab519712b6ed5856"
ROBUST_RECEIPT_SHA = "89badf901caec5fdecbe2ff2d1d49559dbf6eac19735486c11e4b7e2a4355323"

AUTHORITY = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_authority.json"
LEDGER = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_ledger.pt"
RESULT = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_result.json"
RECEIPT = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_receipt.json"
FAILURE = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_failure.json"
LOCK = Path("/workspace/runs/.mlp2_trajectory_robust_r512_v1_physical_eval.lock")

ARMS = (
    "NATIVE", "C512", "FULL512", "C512_FULL512", "CONTINUE512",
    "C512_CONTINUE512", "ROBUST512", "C512_ROBUST512",
)
PROGRAM_FOR_ARM = {
    "FULL512": "FULL512", "C512_FULL512": "FULL512",
    "CONTINUE512": "CONTINUE512", "C512_CONTINUE512": "CONTINUE512",
    "ROBUST512": "ROBUST512", "C512_ROBUST512": "ROBUST512",
}
C512_ARMS = {"C512", "C512_FULL512", "C512_CONTINUE512", "C512_ROBUST512"}
SCORING = slice(64, 256)
BOOTSTRAPS = 10_000
SEED = 2026082942
LOW_Q = 0.0027777778
HIGH_Q = 0.9972222222
STATE_SHAPES = {
    "left": (512, 1152), "right": (512, 1152),
    "down": (1152, 512), "bias": (1152,),
}
EXPECTED_PRICE = {
    "input_width": 1152, "output_width": 1152, "products": 512,
    "coefficient_count": 1770624, "stored_scalar_values": 1770624,
    "stored_bytes_float32": 7082496, "support_metadata_values": 0,
    "dense_matrix_multiplies_per_token": 3, "stored_dtype": "torch.float32",
    "execution_dtype": "state_dtype_bfloat16_in_deployment",
    "native_mlp_calls_per_forward": 0,
}
EXPECTED_CHECKPOINT = {
    "config_sha256": "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
    "weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    "weights_bytes": 2067738635, "tokenizer_vocab": 50257,
    "logit_vocab": 50304, "revision": "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240",
}


def committed_sources() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    return commit, row_life.source_hashes(commit)


def verify_sources(commit: str, expected: dict[str, str]) -> None:
    if row_life.source_hashes(commit) != expected:
        raise RuntimeError("physical-eval source closure changed")


def validate_parents() -> dict[str, str]:
    parents = base.validate_parents()
    receipt, _ = base.stable_json(ROBUST_RECEIPT, ROBUST_RECEIPT_SHA)
    bundle, _ = base.stable_torch(ROBUST_BUNDLE, ROBUST_BUNDLE_SHA)
    if receipt.get("status") != "fit_complete_receipt_last_evaluation_unopened" \
            or receipt.get("bundle_sha256") != ROBUST_BUNDLE_SHA \
            or receipt.get("evaluation_opened") is not False \
            or bundle.get("schema") != "mlp2_trajectory_robust_r512_v1_fit_bundle" \
            or bundle.get("evaluation_opened") is not False \
            or set(bundle.get("programs", {})) != {"CONTINUE512", "ROBUST512"}:
        raise RuntimeError("trajectory-robust frozen parent chain changed")
    old_bundle, _ = base.stable_torch(base.FULL_BUNDLE, base.FULL_BUNDLE_SHA)
    validate_program_integrity(old_bundle, bundle)
    parents[str(ROBUST_BUNDLE)] = ROBUST_BUNDLE_SHA
    parents[str(ROBUST_RECEIPT)] = ROBUST_RECEIPT_SHA
    return parents


def validate_program_state(name: str, state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != set(STATE_SHAPES):
        raise RuntimeError(f"{name} program state schema changed")
    for key, shape in STATE_SHAPES.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 \
                or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise RuntimeError(f"{name}.{key} serialization changed")
    model = refit.build_from_state(state, torch.device("cpu"))
    if model.price() != EXPECTED_PRICE \
            or any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError(f"{name} literal price or stored dtype changed")
    probe = torch.zeros(1, 1, 1152, dtype=torch.bfloat16)
    with torch.inference_mode():
        output = model(probe)
    if output.dtype != torch.bfloat16 or tuple(output.shape) != tuple(probe.shape) \
            or not torch.isfinite(output).all():
        raise RuntimeError(f"{name} deployment precision contract changed")
    return {"price": EXPECTED_PRICE, "serialized_state_dtype": "torch.float32",
            "probe_input_dtype": "torch.bfloat16", "probe_output_dtype": "torch.bfloat16",
            "probe_finite": True}


def validate_program_integrity(old_bundle: Any, robust_bundle: Any) -> dict[str, Any]:
    if not isinstance(old_bundle, dict) or "programs" not in old_bundle \
            or set(old_bundle["programs"]) != {"DOWN512", "FULL512", "RANDOM512"} \
            or not isinstance(robust_bundle, dict) \
            or set(robust_bundle.get("programs", {})) != {"CONTINUE512", "ROBUST512"}:
        raise RuntimeError("physical-eval program bundle schema changed")
    states = {"FULL512": old_bundle["programs"]["FULL512"],
              **robust_bundle["programs"]}
    return {name: validate_program_state(name, state) for name, state in states.items()}


def validate_row_receipt(value: Any, sources: dict[str, str]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != (
        "mlp2_trajectory_robust_r512_v1_physical_eval_rows"
    ) or value.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" \
            or value.get("source_hashes") != sources or value.get("outcome_access") != {
                "model_loaded": False, "training_run": False,
            } or value.get("selection") != {
                "start_document_index": 120000, "documents_per_role": 192,
                "token_length": 257, "scored_slice": [64, 256],
            } or value.get("roles") != {
                "EVALUATION": {"authorized_for_training": False,
                               "authorized_for_evaluation": True},
            } or set(value.get("entries", {})) != {"EVALUATION"} \
            or not all(value.get("disjointness", {}).values()):
        raise RuntimeError("physical-eval row receipt semantics changed")
    for entry in value["entries"].values():
        path = Path(entry["path"])
        if entry.get("shape") != [192, 257] or entry.get("dtype") != "torch.int64" \
                or not path.is_file() or base.file_sha256(path) != entry.get("file_sha256"):
            raise RuntimeError("physical-eval row entry changed")
    return value


def protected_snapshot(authority: dict[str, Any]) -> dict[str, Any]:
    verify_sources(authority["source_commit"], authority["source_hashes"])
    audit, audit_sha = row_life.validate_independent_audit(authority["source_hashes"])
    rows, rows_sha = base.stable_json(ROWS_RECEIPT, authority["row_receipt_sha256"])
    validate_row_receipt(rows, authority["source_hashes"])
    parents = validate_parents()
    if parents != authority["parents"] or audit_sha != authority["audit_sha256"]:
        raise RuntimeError("physical-eval protected parent changed")
    snapshot = Path(facade.DEFAULT_SNAPSHOT)
    return {
        "source_commit": authority["source_commit"],
        "source_hashes": authority["source_hashes"],
        "audit_sha256": audit_sha, "audit_commit": audit["audited_source_commit"],
        "row_receipt_sha256": rows_sha,
        "row_file_hashes": {k: base.file_sha256(Path(v["path"]))
                            for k, v in rows["entries"].items()},
        "parents": parents,
        "program_integrity": authority["program_integrity"],
        "checkpoint_config_sha256": base.file_sha256(snapshot / "config.json"),
        "checkpoint_weights_sha256": base.file_sha256(snapshot / "pytorch_model.bin"),
    }


def verify_protected(expected: dict[str, Any], authority: dict[str, Any], claim) -> None:
    row_life.base.require_claim(claim, LOCK)
    if protected_snapshot(authority) != expected:
        raise RuntimeError("physical-eval protected snapshot changed")
    row_life.base.require_claim(claim, LOCK)


def expected_call_census() -> dict[str, Any]:
    output = {}
    for arm in ARMS:
        native = {str(site): 48 for site in range(18)}
        if arm in C512_ARMS:
            native["0"] = 0
        if arm in PROGRAM_FOR_ARM:
            native["2"] = 0
        output[arm] = {
            "outer_calls": 48, "outer_returns": 48,
            "attention_sites": {str(site): 48 for site in range(18)},
            "native_mlp_sites": native,
            "candidate_c512": 48 if arm in C512_ARMS else 0,
            "candidate_mlp2": {name: 48 if PROGRAM_FOR_ARM.get(arm) == name else 0
                               for name in ("FULL512", "CONTINUE512", "ROBUST512")},
        }
    return output


def validate_ledger(value: Any, authority_sha: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or set(value) != {
        "schema", "arms", "calls", "authority_sha256", "checkpoint",
        "program_integrity",
    } or value.get("schema") != (
        "mlp2_trajectory_robust_r512_v1_physical_eval_ledger"
    ) or value.get("authority_sha256") != authority_sha \
            or value.get("calls") != expected_call_census() \
            or set(value.get("arms", {})) != set(ARMS) \
            or value.get("program_integrity") != expected_program_integrity():
        raise RuntimeError("physical-eval ledger metadata changed")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "snapshot", "revision", "weights_sha256", "weights_bytes", "config_sha256",
        "tokenizer_vocab", "logit_vocab",
    } or any(checkpoint.get(key) != expected for key, expected in EXPECTED_CHECKPOINT.items()):
        raise RuntimeError("physical-eval checkpoint identity changed")
    arms = value["arms"]
    if any(not isinstance(x, torch.Tensor) or x.dtype != torch.float64
           or tuple(x.shape) != (192, 9) or not torch.isfinite(x).all()
           or (x[:, 8] != 192).any() for x in arms.values()):
        raise RuntimeError("physical-eval sufficient statistics changed")
    return arms


def expected_program_integrity() -> dict[str, Any]:
    old_bundle, _ = base.stable_torch(base.FULL_BUNDLE, base.FULL_BUNDLE_SHA)
    robust_bundle, _ = base.stable_torch(ROBUST_BUNDLE, ROBUST_BUNDLE_SHA)
    return validate_program_integrity(old_bundle, robust_bundle)


def artifact_snapshot() -> dict[str, str | None]:
    return {path.name: base.file_sha256(path) if path.is_file() else None
            for path in (AUTHORITY, LEDGER, RESULT)}


def failure_terminal_guard(
    claim, expected_artifacts: dict[str, str | None],
    expected_authority: dict[str, Any] | None,
    expected_protected: dict[str, Any] | None,
) -> None:
    row_life.base.require_claim(claim, LOCK)
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("physical-eval terminal raced failure")
    if artifact_snapshot() != expected_artifacts:
        raise RuntimeError("physical-eval failure aggregate artifact state changed")
    if expected_authority is None:
        if expected_artifacts.get(AUTHORITY.name) is not None \
                or expected_protected is not None:
            raise RuntimeError("physical-eval absent authority failure state changed")
    else:
        authority_value, authority_sha = base.stable_json(
            AUTHORITY, expected_artifacts[AUTHORITY.name],
        )
        if authority_value != expected_authority or expected_protected is None \
                or protected_snapshot(authority_value) != expected_protected \
                or authority_sha != expected_artifacts[AUTHORITY.name]:
            raise RuntimeError("physical-eval failure authority or protected state changed")
    if artifact_snapshot() != expected_artifacts:
        raise RuntimeError("physical-eval failure artifact state changed during replay")
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("physical-eval terminal raced failure during artifact replay")
    row_life.base.require_claim(claim, LOCK)


def validate_receipt_value(value: Any, authority_sha: str, ledger_sha: str,
                           result_sha: str) -> dict[str, Any]:
    expected = {
        "schema": "mlp2_trajectory_robust_r512_v1_physical_eval_receipt",
        "status": "result_complete_receipt_last", "authority_sha256": authority_sha,
        "ledger_sha256": ledger_sha, "result_sha256": result_sha,
        "evaluation_opened": True,
    }
    if value != expected:
        raise RuntimeError("physical-eval receipt semantics changed")
    # Force canonical JSON serialization and a semantic round-trip before the
    # create-only receipt transaction begins.  Nothing fallible follows its link.
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if json.loads(encoded) != expected:
        raise RuntimeError("physical-eval canonical receipt replay changed")
    return expected


def publish_failure(
    claim, exc: BaseException, authority: dict[str, Any] | None,
    protected: dict[str, Any] | None, evaluation_opened: bool,
) -> dict[str, Any]:
    frozen_artifacts = artifact_snapshot()
    authority_published = frozen_artifacts[AUTHORITY.name] is not None
    published_authority = authority if authority_published else None
    published_protected = protected if authority_published else None
    failure = {
        "schema": "mlp2_trajectory_robust_r512_v1_physical_eval_failure",
        "status": "terminal_failure_no_receipt", "error": repr(exc),
        "authority_exists": authority_published,
        "evaluation_may_have_opened": evaluation_opened,
        "protected_snapshot": published_protected,
        "artifact_snapshot": frozen_artifacts,
    }
    if not RECEIPT.exists() and not FAILURE.exists():
        def failure_guard() -> None:
            failure_terminal_guard(
                claim, frozen_artifacts, published_authority, published_protected,
            )

        base.atomic_json(FAILURE, failure, pre_link_check=failure_guard)
    return failure


def document_metric(ledger: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "dce":
        return (ledger[:, 1] - ledger[:, 0]) / ledger[:, 8]
    if metric == "kl":
        return ledger[:, 2] / ledger[:, 8]
    raise ValueError(metric)


def factorial_document(ledgers: dict[str, torch.Tensor], program: str) -> torch.Tensor:
    p = document_metric(ledgers[program], "dce")
    c = document_metric(ledgers["C512"], "dce")
    cp = document_metric(ledgers[f"C512_{program}"], "dce")
    return cp - c - p


def interval(draws: torch.Tensor) -> list[float]:
    return [float(torch.quantile(draws, LOW_Q, interpolation="linear")),
            float(torch.quantile(draws, HIGH_Q, interpolation="linear"))]


def simultaneous_contrasts(ledgers: dict[str, torch.Tensor]) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    index = torch.randint(0, 192, (BOOTSTRAPS, 192), generator=generator)
    i_full = factorial_document(ledgers, "FULL512")
    i_robust = factorial_document(ledgers, "ROBUST512")

    def mean_draw(values: torch.Tensor) -> torch.Tensor:
        return values[index].mean(1)

    full_draw, robust_draw = mean_draw(i_full), mean_draw(i_robust)
    raw: dict[str, torch.Tensor] = {
        "fresh_full_interaction": full_draw,
        "half_interaction_reduction": 0.5 * full_draw.abs() - robust_draw.abs(),
        "robust_absolute_interaction": robust_draw.abs(),
        "combined_dce_gain_vs_full": mean_draw(
            document_metric(ledgers["C512_FULL512"], "dce")
            - document_metric(ledgers["C512_ROBUST512"], "dce")),
        "combined_kl_gain_vs_full": mean_draw(
            document_metric(ledgers["C512_FULL512"], "kl")
            - document_metric(ledgers["C512_ROBUST512"], "kl")),
        "standalone_dce_noninferiority_margin": mean_draw(
            0.005 + document_metric(ledgers["FULL512"], "dce")
            - document_metric(ledgers["ROBUST512"], "dce")),
        "standalone_kl_noninferiority_margin": mean_draw(
            0.005 + document_metric(ledgers["FULL512"], "kl")
            - document_metric(ledgers["ROBUST512"], "kl")),
        "combined_dce_gain_vs_continue": mean_draw(
            document_metric(ledgers["C512_CONTINUE512"], "dce")
            - document_metric(ledgers["C512_ROBUST512"], "dce")),
        "combined_kl_gain_vs_continue": mean_draw(
            document_metric(ledgers["C512_CONTINUE512"], "kl")
            - document_metric(ledgers["C512_ROBUST512"], "kl")),
    }
    points = {
        "fresh_full_interaction": float(i_full.mean()),
        "half_interaction_reduction": float(0.5 * abs(i_full.mean()) - abs(i_robust.mean())),
        "robust_absolute_interaction": float(abs(i_robust.mean())),
        "combined_dce_gain_vs_full": float((
            document_metric(ledgers["C512_FULL512"], "dce")
            - document_metric(ledgers["C512_ROBUST512"], "dce")).mean()),
        "combined_kl_gain_vs_full": float((
            document_metric(ledgers["C512_FULL512"], "kl")
            - document_metric(ledgers["C512_ROBUST512"], "kl")).mean()),
        "standalone_dce_noninferiority_margin": float((
            0.005 + document_metric(ledgers["FULL512"], "dce")
            - document_metric(ledgers["ROBUST512"], "dce")).mean()),
        "standalone_kl_noninferiority_margin": float((
            0.005 + document_metric(ledgers["FULL512"], "kl")
            - document_metric(ledgers["ROBUST512"], "kl")).mean()),
        "combined_dce_gain_vs_continue": float((
            document_metric(ledgers["C512_CONTINUE512"], "dce")
            - document_metric(ledgers["C512_ROBUST512"], "dce")).mean()),
        "combined_kl_gain_vs_continue": float((
            document_metric(ledgers["C512_CONTINUE512"], "kl")
            - document_metric(ledgers["C512_ROBUST512"], "kl")).mean()),
    }
    return {name: {"point": points[name], "simultaneous_ci": interval(draws)}
            for name, draws in raw.items()}


def optimization_inconclusive(bundle: dict[str, Any]) -> dict[str, Any]:
    curve = bundle["curves"]["ROBUST512"]
    by_step = {row["step"]: row["worst_normalized_mse"] for row in curve}
    last_four = [row["worst_normalized_mse"] for row in curve[-4:]]
    best = min(row["worst_normalized_mse"] for row in curve)
    improvement = (by_step[1100] - by_step[1200]) / by_step[1100]
    gates = {
        "all_1200_steps": curve[-1]["step"] == 1200,
        "best_worst_nrmse_above_0p25": math.sqrt(best) > 0.25,
        "last_four_strictly_decreasing": all(a > b for a, b in zip(last_four, last_four[1:])),
        "step1200_improves_at_least_1pct_vs_1100": improvement >= 0.01,
    }
    return {"applies": all(gates.values()), "gates": gates,
            "best_worst_nrmse": math.sqrt(best),
            "relative_improvement_1100_to_1200": improvement}


def derive_result(ledgers: dict[str, torch.Tensor], runtime: float,
                  bundle: dict[str, Any]) -> dict[str, Any]:
    summaries = {arm: {str(p): base.summarize(value, p) for p in (48, 96, 192)}
                 for arm, value in ledgers.items()}
    contrasts = simultaneous_contrasts(ledgers)
    prefix_stability = all(
        abs(summaries[arm]["192"][metric] - summaries[arm]["96"][metric]) <= 0.01
        for arm in PROGRAM_FOR_ARM for metric in ("dce", "teacher_kl")
    )
    gates = {
        "fresh_full_interaction_positive": contrasts["fresh_full_interaction"]["simultaneous_ci"][0] > 0,
        "robust_halves_absolute_interaction": contrasts["half_interaction_reduction"]["simultaneous_ci"][0] > 0,
        "robust_absolute_interaction_at_most_0p005": contrasts["robust_absolute_interaction"]["simultaneous_ci"][1] <= 0.005,
        "robust_combined_dce_improves_0p005": contrasts["combined_dce_gain_vs_full"]["simultaneous_ci"][0] >= 0.005,
        "robust_combined_kl_improves_0p005": contrasts["combined_kl_gain_vs_full"]["simultaneous_ci"][0] >= 0.005,
        "robust_standalone_dce_noninferior": contrasts["standalone_dce_noninferiority_margin"]["simultaneous_ci"][0] >= 0,
        "robust_standalone_kl_noninferior": contrasts["standalone_kl_noninferiority_margin"]["simultaneous_ci"][0] >= 0,
        "robust_combined_dce_beats_continue": contrasts["combined_dce_gain_vs_continue"]["simultaneous_ci"][0] > 0,
        "prefix_stability": prefix_stability,
    }
    optimization = optimization_inconclusive(bundle)
    if all(gates.values()):
        status = "trajectory_exposure_supported"
    elif optimization["applies"]:
        status = "optimization_inconclusive"
    else:
        status = "trajectory_exposure_rejected"
    return {
        "schema": "mlp2_trajectory_robust_r512_v1_physical_eval_result",
        "status": status,
        "claim_boundary": "fresh_in_distribution_composition_no_strict_ledger_move",
        "documents": 192, "runtime_seconds": runtime,
        "summaries": summaries, "contrasts": contrasts,
        "decision_gates": gates, "optimization_rule": optimization,
        "bootstrap": {"draws": BOOTSTRAPS, "seed": SEED,
                      "simultaneous_quantiles": [LOW_Q, HIGH_Q]},
    }


def main() -> None:
    if any(p.exists() for p in (AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("physical-eval namespace already exists")
    claim = row_life.base.acquire_claim(LOCK)
    authority = None
    protected = None
    evaluation_opened = False
    try:
        commit, sources = committed_sources()
        audit, audit_sha = row_life.validate_independent_audit(sources)
        parents = validate_parents()
        program_integrity = expected_program_integrity()
        row_receipt, row_receipt_sha = base.stable_json(ROWS_RECEIPT)
        validate_row_receipt(row_receipt, sources)
        entry = row_receipt["entries"]["EVALUATION"]
        authority = {
            "schema": "mlp2_trajectory_robust_r512_v1_physical_eval_authority",
            "status": "frozen_before_evaluation_open", "source_commit": commit,
            "source_hashes": sources, "audit_sha256": audit_sha,
            "audit_reviewer": audit["reviewer"], "parents": parents,
            "program_integrity": program_integrity,
            "row_receipt_sha256": row_receipt_sha,
            "evaluation_rows_sha256": entry["file_sha256"], "arms": list(ARMS),
            "scored_slice": [64, 256], "outcome_access": False,
        }
        # Compute the complete protected state before the authority link.  This
        # ensures every published authority has a matching failure-replay snapshot.
        protected = protected_snapshot(authority)

        def authority_guard() -> None:
            row_life.base.require_claim(claim, LOCK)
            verify_sources(commit, sources); row_life.validate_independent_audit(sources)
            base.stable_json(ROWS_RECEIPT, row_receipt_sha)
            if validate_parents() != parents \
                    or expected_program_integrity() != program_integrity \
                    or any(p.exists() for p in (
                AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE,
            )):
                raise RuntimeError("physical-eval authority inputs changed")
            row_life.base.require_claim(claim, LOCK)

        base.atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = base.file_sha256(AUTHORITY)
        started = time.time(); evaluation_opened = True
        rows, rows_sha = base.stable_torch(Path(entry["path"]), entry["file_sha256"])
        if rows_sha != authority["evaluation_rows_sha256"] \
                or tuple(rows.shape) != (192, 257) \
                or refit.row_life.tensor_sha256(rows) != entry["tensor_sha256"]:
            raise RuntimeError("physical-eval rows changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        verify_protected(protected, authority, claim)
        c = load_program(base.C512_PATH)
        c_tensors = {k: c[k].to(device) for k in ("intercept", "left", "right")}
        old_bundle, _ = base.stable_torch(base.FULL_BUNDLE, base.FULL_BUNDLE_SHA)
        robust_bundle, _ = base.stable_torch(ROBUST_BUNDLE, ROBUST_BUNDLE_SHA)
        programs = {
            "FULL512": refit.build_from_state(old_bundle["programs"]["FULL512"], device).eval(),
            "CONTINUE512": refit.build_from_state(robust_bundle["programs"]["CONTINUE512"], device).eval(),
            "ROBUST512": refit.build_from_state(robust_bundle["programs"]["ROBUST512"], device).eval(),
        }
        if validate_program_integrity(old_bundle, robust_bundle) != program_integrity:
            raise RuntimeError("physical-eval loaded program integrity changed")
        ledgers = {arm: [] for arm in ARMS}
        calls = {arm: {
            "outer_calls": 0, "outer_returns": 0,
            "attention_sites": {str(site): 0 for site in range(18)},
            "native_mlp_sites": {str(site): 0 for site in range(18)},
            "candidate_c512": 0,
            "candidate_mlp2": {name: 0 for name in programs},
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
                        if event.site == 0 and arm in C512_ARMS:
                            calls[arm]["candidate_c512"] += 1
                            return base.c512_write(event, c_tensors)
                        program = PROGRAM_FOR_ARM.get(arm)
                        if event.site == 2 and program is not None:
                            calls[arm]["candidate_mlp2"][program] += 1
                            return programs[program](event.state)
                        calls[arm]["native_mlp_sites"][str(event.site)] += 1
                        return event.block.mlp(event.state)

                    calls[arm]["outer_calls"] += 1
                    logits[arm] = facade.forward_with_dispatch(model, tokens, attention, mlp)
                    calls[arm]["outer_returns"] += 1
                native = logits["NATIVE"]
                for arm in ARMS:
                    ledgers[arm].append(refit.reduce_document(native, logits[arm], targets))
        packed = {arm: torch.cat(parts) for arm, parts in ledgers.items()}
        if calls != expected_call_census():
            raise RuntimeError("physical-eval call census changed")
        ledger = {"schema": "mlp2_trajectory_robust_r512_v1_physical_eval_ledger",
                  "arms": packed, "calls": calls, "authority_sha256": authority_sha,
                  "checkpoint": checkpoint.__dict__,
                  "program_integrity": program_integrity}

        def ledger_guard() -> None:
            verify_protected(protected, authority, claim)
            if any(p.exists() for p in (LEDGER, RESULT, RECEIPT, FAILURE)):
                raise RuntimeError("physical-eval terminal raced ledger")

        base.atomic_torch(LEDGER, ledger, pre_link_check=ledger_guard)
        reloaded, ledger_sha = base.stable_torch(LEDGER)
        replay_arms = validate_ledger(reloaded, authority_sha)
        runtime = time.time() - started
        result = derive_result(replay_arms, runtime, robust_bundle)
        result["parents"] = {"authority": authority_sha, "ledger": ledger_sha}
        result["program_integrity"] = program_integrity

        def result_guard() -> None:
            verify_protected(protected, authority, claim); base.stable_torch(LEDGER, ledger_sha)
            if any(p.exists() for p in (RESULT, RECEIPT, FAILURE)):
                raise RuntimeError("physical-eval terminal raced result")

        base.atomic_json(RESULT, result, pre_link_check=result_guard)
        reloaded_result, result_sha = base.stable_json(RESULT)
        if reloaded_result != result or derive_result(
            replay_arms, runtime, robust_bundle,
        ) != {key: value for key, value in result.items()
              if key not in ("parents", "program_integrity")} \
                or reloaded_result.get("program_integrity") != program_integrity:
            raise RuntimeError("physical-eval result replay changed")
        receipt = validate_receipt_value({
            "schema": "mlp2_trajectory_robust_r512_v1_physical_eval_receipt",
            "status": "result_complete_receipt_last", "authority_sha256": authority_sha,
            "ledger_sha256": ledger_sha, "result_sha256": result_sha,
            "evaluation_opened": True,
        }, authority_sha, ledger_sha, result_sha)
        rendered_result = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)

        def receipt_guard() -> None:
            verify_protected(protected, authority, claim)
            base.stable_json(AUTHORITY, authority_sha); base.stable_torch(LEDGER, ledger_sha)
            base.stable_json(RESULT, result_sha)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("physical-eval terminal raced receipt")
            row_life.base.require_claim(claim, LOCK)

        print(rendered_result)
        # Receipt publication is deliberately the final fallible transaction action.
        base.atomic_json(RECEIPT, receipt, pre_link_check=receipt_guard)
    except BaseException as exc:
        publish_failure(claim, exc, authority, protected, evaluation_opened)
        raise
    finally:
        row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
