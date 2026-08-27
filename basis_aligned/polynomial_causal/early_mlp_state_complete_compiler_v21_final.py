#!/usr/bin/env python3
"""One-shot final scorer and last-write outcome authority for compiler-v2.1.

The final role is loaded exactly once, after an authority-none attempt record
binds the unlock, program bundle, final-cache identity, and complete source
closure.  All arms retain row-level sufficient statistics; every nonlinear
margin is recomputed inside one shared source-document bootstrap.  Statistical
failure is an authoritative negative result.  Integrity failure writes no
outcome authority and requires a new committed namespace.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import early_mlp_state_complete_compiler_v2 as compiler  # noqa: E402
import early_mlp_state_complete_compiler_v2_site0 as old_site0  # noqa: E402
import early_mlp_state_complete_compiler_v21 as lifecycle  # noqa: E402
import prepare_state_complete_compiler_rows_v21 as authority  # noqa: E402


ATTEMPT = authority.BQ / "early_mlp_state_complete_compiler_v21_final_attempt.json"
RESULT = authority.BQ / "early_mlp_state_complete_compiler_v21_final_result.pt"
MANIFEST = authority.BQ / "early_mlp_state_complete_compiler_v21_final_manifest.json"
OUTCOME_AUTHORITY = (
    authority.BQ / "early_mlp_state_complete_compiler_v21_final_authority.json"
)
FINAL_OUTPUTS = (ATTEMPT, RESULT, MANIFEST, OUTCOME_AUTHORITY)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260827
FAMILIES = tuple(authority.selection.ALL_FAMILIES)
SCORER = "CUDA float32 per-token; float64 row/aggregate"
BASELINE_STAT_KEYS = (
    "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
    "row_frequency_ce_sum", "row_frequency_count",
)


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"v2.1 final binding target is absent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": authority.file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _final_entry() -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(authority.RECEIPT.read_text())
    entry = receipt["entries"]["compiler_final_v21"]
    records = receipt["document_provenance"]["sets"]["compiler_final_v21"]
    path = Path(entry["cache_path"])
    if not path.is_file() or authority.file_sha256(path) != entry[
        "cache_file_sha256"
    ] or authority.logical_json_sha256(records) != entry[
        "provenance_records_sha256"
    ]:
        raise RuntimeError("v2.1 final cache/provenance identity changed")
    return dict(entry), {"records": records, "path": path}


def _frequency_assignments(
    row_receipt: Mapping[str, Any], final_rows: torch.Tensor,
) -> torch.Tensor:
    fit = authority._load_role_cache_for_strata(row_receipt, "compiler_fit_v21")
    fit_targets = fit[:, 65:257].contiguous()
    fit_counts = torch.bincount(
        fit_targets.flatten(), minlength=authority.TOKEN_VOCAB
    ).long()
    final_targets = final_rows[:, 65:257].contiguous()
    assignments = torch.bucketize(
        fit_counts.index_select(0, final_targets.flatten()),
        torch.tensor(authority.TOKEN_FREQUENCY_BOUNDARIES, dtype=torch.long),
        right=True,
    ).view_as(final_targets)
    return assignments.contiguous()


def write_attempt_before_final_load(
    unlock: Mapping[str, Any], *, protected_before: Mapping[str, Any],
) -> dict[str, Any]:
    if any(path.exists() for path in FINAL_OUTPUTS):
        raise RuntimeError("v2.1 final output namespace is not empty")
    entry, final = _final_entry()
    attempt = {
        "schema_version": 1,
        "status": "attempt_frozen_before_compiler_final_v21_deserialization",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "authorized_for_final_scoring": False,
        "requested_role": "compiler_final_v21",
        "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
        "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
        "rows_receipt": _binding(authority.RECEIPT),
        "rows_manifest": _binding(authority.MANIFEST),
        "final_cache": {
            "path": str(final["path"].resolve()),
            "sha256": entry["cache_file_sha256"],
            "bytes": final["path"].stat().st_size,
            "tensor_full_raw_sha256": entry["tensor_full_raw_sha256"],
            "tensor_prefix257_raw_sha256": entry["tensor_prefix257_raw_sha256"],
            "provenance_records_sha256": entry["provenance_records_sha256"],
        },
        "source_commit": unlock["source_commit"],
        "source_hashes": dict(unlock["source_hashes"]),
        "protected_before": dict(protected_before),
        "final_role_loads_before_attempt": 0,
        "final_evaluations_before_attempt": 0,
    }
    authority.write_json_atomic(attempt, ATTEMPT)
    reloaded = json.loads(ATTEMPT.read_text())
    if reloaded != attempt:
        raise RuntimeError("v2.1 final attempt did not reload exactly")
    return attempt


def _states_for_arm(code: str) -> tuple[dict[int, str], set[int]]:
    if len(code) != 3 or code[0] not in "NQO" or code[1] not in "NQO" or (
        code[2] not in "NE"
    ):
        raise ValueError(f"invalid v2.1 lattice arm: {code}")
    states = {
        site: state for site, state in enumerate(code) if state != "N"
    }
    allowed = {site for site, state in states.items() if state in {"O", "E"}}
    return states, allowed


def _programs_device(
    programs: Mapping[str, Mapping[int, Mapping[str, Any]]], device: Any,
) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        name: {
            int(site): old_site0._device_state(state, device)
            for site, state in sites.items()
        }
        for name, sites in programs.items()
    }


@torch.no_grad()
def score_arm(
    sa: Any, hook: Any, rows: torch.Tensor, frequency: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int], *, name: str,
    programs: Mapping[str, Mapping[int, Mapping[str, Any]]], program_name: str,
    states: Mapping[int, str], allowed: set[int],
    teachers: Mapping[str, Sequence[torch.Tensor]] | None = None,
    retain_logits: bool = False,
) -> tuple[dict[str, Any], list[torch.Tensor] | None]:
    hook.programs = _programs_device(programs, sa.DEV)
    hook.configure(states, program_name=program_name)
    row_ce_sum = []
    row_ce_count = []
    row_copy_sum = []
    row_copy_count = []
    row_frequency_sum = []
    row_frequency_count = []
    row_kl: dict[str, list[torch.Tensor]] = {
        teacher: [] for teacher in (teachers or {})
    }
    logits_output: list[torch.Tensor] | None = [] if retain_logits else None
    with old_site0.runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
        for batch_index, start in enumerate(range(0, len(rows), 8)):
            stop = min(start + 8, len(rows))
            batch = rows[start:stop].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = sa.fwd_arm(
                idx, all_attention, twall, frozenset(range(18))
            ).float()[:, 64:]
            target = targets[:, 64:]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                reduction="none",
            ).view(stop - start, -1)
            row_ce_sum.append(ce.double().sum(dim=1).cpu())
            row_ce_count.append(torch.full(
                (stop - start,), ce.shape[1], dtype=torch.long,
            ))
            copy = old_site0._copy_mask(idx, targets)[:, 64:]
            row_copy_sum.append((ce.double() * copy).sum(dim=1).cpu())
            row_copy_count.append(copy.sum(dim=1).long().cpu())
            assignment = frequency[start:stop].to(sa.DEV)
            frequency_sums = []
            frequency_counts = []
            for bin_index in range(len(authority.TOKEN_FREQUENCY_BOUNDARIES) + 1):
                mask = assignment == bin_index
                frequency_sums.append((ce.double() * mask).sum(dim=1).cpu())
                frequency_counts.append(mask.sum(dim=1).long().cpu())
            row_frequency_sum.append(torch.stack(frequency_sums, dim=1))
            row_frequency_count.append(torch.stack(frequency_counts, dim=1))
            for teacher_name, teacher_batches in (teachers or {}).items():
                teacher = teacher_batches[batch_index].to(sa.DEV)
                teacher_logp = F.log_softmax(teacher, dim=-1)
                student_logp = F.log_softmax(logits, dim=-1)
                kl = (
                    teacher_logp.exp() * (teacher_logp - student_logp)
                ).sum(dim=-1)
                row_kl[teacher_name].append(kl.double().sum(dim=1).cpu())
            if logits_output is not None:
                logits_output.append(logits.cpu().contiguous())
    guard.assert_contract(require_allowed_calls=bool(allowed))
    expected_calls = {
        site: (math.ceil(len(rows) / 8) if site in allowed else 0)
        for site in (0, 1, 2)
    }
    if dict(guard.counts) != expected_calls:
        raise RuntimeError(
            f"v2.1 final {name} original-call counter changed: "
            f"{dict(guard.counts)} != {expected_calls}"
        )
    raw = {
        "name": name,
        "scorer": SCORER,
        "row_ce_sum": torch.cat(row_ce_sum).double().contiguous(),
        "row_ce_count": torch.cat(row_ce_count).long().contiguous(),
        "row_copy_ce_sum": torch.cat(row_copy_sum).double().contiguous(),
        "row_copy_count": torch.cat(row_copy_count).long().contiguous(),
        "row_frequency_ce_sum": torch.cat(row_frequency_sum).double().contiguous(),
        "row_frequency_count": torch.cat(row_frequency_count).long().contiguous(),
        "row_teacher_kl_sum": {
            teacher: torch.cat(values).double().contiguous()
            for teacher, values in row_kl.items()
        },
        "row_teacher_kl_count": {
            teacher: torch.cat(row_ce_count).long().contiguous()
            for teacher in row_kl
        },
        "original_mlp_call_counters": dict(guard.counts),
    }
    validate_arm_statistics(raw, len(rows))
    return raw, logits_output


def validate_arm_statistics(value: Mapping[str, Any], n_rows: int = 192) -> None:
    tensor_shapes = {
        "row_ce_sum": ((n_rows,), torch.float64),
        "row_ce_count": ((n_rows,), torch.long),
        "row_copy_ce_sum": ((n_rows,), torch.float64),
        "row_copy_count": ((n_rows,), torch.long),
        "row_frequency_ce_sum": ((n_rows, 9), torch.float64),
        "row_frequency_count": ((n_rows, 9), torch.long),
    }
    for key, (shape, dtype) in tensor_shapes.items():
        tensor = value.get(key)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or (
            tensor.dtype != dtype
        ) or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())):
            raise RuntimeError(f"v2.1 final arm statistic changed: {key}")
    if not torch.equal(
        value["row_frequency_count"].sum(dim=1), value["row_ce_count"]
    ) or bool((value["row_ce_count"] != 192).any()) or bool(
        (value["row_copy_count"] < 0).any()
    ):
        raise RuntimeError("v2.1 final arm supports do not partition")
    kl_sums = value.get("row_teacher_kl_sum")
    kl_counts = value.get("row_teacher_kl_count")
    if not isinstance(kl_sums, Mapping) or not isinstance(kl_counts, Mapping) or (
        set(kl_sums) != set(kl_counts)
    ):
        raise RuntimeError("v2.1 final KL schema changed")
    for teacher in kl_sums:
        if not torch.is_tensor(kl_sums[teacher]) or not torch.is_tensor(
            kl_counts[teacher]
        ) or tuple(kl_sums[teacher].shape) != (n_rows,) or tuple(
            kl_counts[teacher].shape
        ) != (n_rows,) or (
            kl_sums[teacher].dtype != torch.float64
        ) or kl_counts[teacher].dtype != torch.long or not bool(
            torch.isfinite(kl_sums[teacher]).all()
        ) or bool((kl_sums[teacher] < -1e-10).any()) or not torch.equal(
            kl_counts[teacher], value["row_ce_count"]
        ):
            raise RuntimeError("v2.1 final KL support changed")


def _signed_gauge_state(
    state: Mapping[str, Any], signs: torch.Tensor,
) -> dict[str, Any]:
    moved = dict(state)
    grammar = state["grammar"]
    if grammar == "affine":
        moved["right"] = state["right"] * signs
        moved["bias"] = state["bias"] * signs
    elif grammar == "native":
        moved["projected_decoder"] = state["projected_decoder"] * signs
        moved["beta"] = state["beta"] * signs
    elif grammar == "constant":
        moved["bias"] = state["bias"] * signs
    else:
        raise RuntimeError("v2.1 signed gauge encountered unknown grammar")
    return moved


def signed_gauge_programs(
    programs: Mapping[int, Mapping[str, Any]], bases: Mapping[int, torch.Tensor],
) -> tuple[dict[int, dict[str, Any]], dict[int, torch.Tensor]]:
    moved_programs = {}
    moved_bases = {}
    for site in (0, 1):
        signs = torch.where(
            torch.arange(authority.compiler.COEFFICIENT_DIM) % 2 == 0,
            torch.tensor(1.0), torch.tensor(-1.0),
        ).float()
        moved_programs[site] = _signed_gauge_state(programs[site], signs)
        moved_bases[site] = bases[site].cpu().float() * signs
    return moved_programs, moved_bases


def signed_gauge_physical_drift(
    programs: Mapping[int, Mapping[str, Any]], bases: Mapping[int, torch.Tensor],
    moved_programs: Mapping[int, Mapping[str, Any]],
    moved_bases: Mapping[int, torch.Tensor],
) -> float:
    generator = torch.Generator().manual_seed(20260827)
    z = torch.randn(17, compiler.D_MODEL, generator=generator)
    mo = torch.randn(17, compiler.D_MODEL, generator=generator)
    maximum = 0.0
    for site in (0, 1):
        before = old_site0.runtime.runtime_coefficients(
            z, mo, bases[site].cpu().float(), programs[site],
        ) @ bases[site].cpu().float().T
        after = old_site0.runtime.runtime_coefficients(
            z, mo, moved_bases[site].cpu().float(), moved_programs[site],
        ) @ moved_bases[site].cpu().float().T
        maximum = max(maximum, float((after - before).abs().max()))
    return maximum


def native_gauge_audit(bundle: Mapping[str, Any]) -> dict[str, Any]:
    checked = 0
    max_norm_drift = 0.0
    sign_failures = 0
    reciprocal_max_drift = 0.0
    swap_max_drift = 0.0
    price_failures = 0
    generator = torch.Generator().manual_seed(20260827)
    z = torch.randn(3, compiler.D_MODEL, generator=generator)
    for ledger in bundle["candidate_ledgers"].values():
        for candidate in ledger.values():
            state = candidate["state"]
            if state["grammar"] != "native":
                continue
            checked += 1
            left, right = state["left"].double(), state["right"].double()
            max_norm_drift = max(
                max_norm_drift,
                float((left.norm(dim=1) - 1).abs().max()),
                float((right.norm(dim=1) - 1).abs().max()),
            )
            pivots = left.abs().argmax(dim=1)
            sign_failures += int((left.gather(1, pivots[:, None]).flatten() < 0).sum())
            baseline = old_site0.runtime.runtime_projected_output(z, state)
            reciprocal = dict(state)
            reciprocal["left"] = state["left"] * 2.0
            reciprocal["right"] = state["right"] * 0.5
            reciprocal_output = old_site0.runtime.runtime_projected_output(
                z, reciprocal
            )
            swapped = dict(state)
            swapped["left"], swapped["right"] = state["right"], state["left"]
            swapped_output = old_site0.runtime.runtime_projected_output(z, swapped)
            reciprocal_max_drift = max(
                reciprocal_max_drift,
                float((reciprocal_output - baseline).abs().max()),
            )
            swap_max_drift = max(
                swap_max_drift, float((swapped_output - baseline).abs().max()),
            )
            expected_price = authority.selection.state_price(state)
            price_failures += int(
                authority.selection.state_price(reciprocal) != expected_price
                or authority.selection.state_price(swapped) != expected_price
            )
    return {
        "native_states_checked": checked,
        "max_unit_norm_drift": max_norm_drift,
        "canonical_sign_failures": sign_failures,
        "reciprocal_rescale_max_projected_output_drift": reciprocal_max_drift,
        "left_right_swap_max_projected_output_drift": swap_max_drift,
        "price_invariance_failures": price_failures,
        "passed": checked > 0 and max_norm_drift <= 2e-6
        and sign_failures == 0 and reciprocal_max_drift <= 2e-6
        and swap_max_drift <= 2e-6 and price_failures == 0,
    }


def document_bootstrap_weights(
    document_ids: Sequence[str], *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[torch.Tensor, list[str]]:
    if len(document_ids) != 192 or draws <= 0:
        raise ValueError("v2.1 final bootstrap support changed")
    documents = list(dict.fromkeys(document_ids))
    if len(documents) < 2:
        raise ValueError("v2.1 final bootstrap needs two source documents")
    index = {document: i for i, document in enumerate(documents)}
    row_document = torch.tensor([index[value] for value in document_ids], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randint(
        len(documents), (draws, len(documents)), generator=generator,
    )
    document_weights = torch.zeros(draws, len(documents), dtype=torch.float64)
    document_weights.scatter_add_(
        1, sampled, torch.ones_like(sampled, dtype=torch.float64),
    )
    return document_weights.index_select(1, row_document).contiguous(), documents


def _series(
    numerator: torch.Tensor, denominator: torch.Tensor, weights: torch.Tensor,
    *, allow_sparse_bootstrap: bool = False,
) -> torch.Tensor:
    numerator = numerator.double().flatten()
    denominator = denominator.double().flatten()
    if numerator.shape != denominator.shape or numerator.numel() != weights.shape[1]:
        raise ValueError("v2.1 final aggregate support changed")
    point_denominator = denominator.sum()
    bootstrap_denominator = weights @ denominator
    if float(point_denominator) <= 0:
        raise RuntimeError("v2.1 final aggregate denominator is empty")
    valid = bootstrap_denominator > 0
    if not allow_sparse_bootstrap and not bool(valid.all()):
        raise RuntimeError("v2.1 final aggregate bootstrap denominator is empty")
    bootstrap = torch.full(
        (weights.shape[0],), torch.nan, dtype=torch.float64,
    )
    bootstrap[valid] = (weights[valid] @ numerator) / bootstrap_denominator[valid]
    return torch.cat([
        (numerator.sum() / point_denominator).view(1),
        bootstrap,
    ]).double().contiguous()


def _summary(series: torch.Tensor) -> dict[str, Any]:
    finite = series[1:][torch.isfinite(series[1:])]
    output = {
        "point": float(series[0]),
        "bootstrap_draws": int(series.numel() - 1),
        "bootstrap_effective_draws": int(finite.numel()),
        "bootstrap_zero_support_draws": int((~torch.isfinite(series[1:])).sum()),
    }
    if finite.numel() == series.numel() - 1:
        output.update({
            "ci_status": "estimated_all_registered_draws",
            "ci95": [
                float(torch.quantile(finite, 0.025, interpolation="linear")),
                float(torch.quantile(finite, 0.975, interpolation="linear")),
            ],
            "bootstrap_mean": float(finite.mean()),
        })
    else:
        output.update({
            "ci_status": "unevaluable_zero_support_resamples",
            "ci95": None,
            "bootstrap_mean": None if not finite.numel() else float(finite.mean()),
        })
    return output


def _positive_with_full_ci(report: Mapping[str, Any]) -> bool:
    return report.get("ci_status") == "estimated_all_registered_draws" and (
        report["point"] > 0 and report["ci95"][0] > 0
    )


def _holm_positive(contrasts: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    pvalues = {
        name: (1 + int((series[1:] <= 0).sum())) / (series.numel())
        for name, series in contrasts.items()
    }
    ordered = sorted(pvalues, key=lambda name: (pvalues[name], name))
    rejected = {}
    continuing = True
    total = len(ordered)
    for index, name in enumerate(ordered):
        threshold = 0.05 / (total - index)
        continuing = continuing and pvalues[name] <= threshold
        rejected[name] = continuing
    return {
        "method": "one-sided paired bootstrap Holm step-down alpha=0.05",
        "pvalues": pvalues,
        "rejected_positive": rejected,
    }


def analyze(
    arms: Mapping[str, Mapping[str, Any]], document_ids: Sequence[str],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    weights, documents = document_bootstrap_weights(document_ids)
    ce = {
        name: _series(row["row_ce_sum"], row["row_ce_count"], weights)
        for name, row in arms.items()
    }
    copy_names = ("QQN", "NNN", "QQE", "NNE")
    copy = {
        name: _series(
            arms[name]["row_copy_ce_sum"], arms[name]["row_copy_count"], weights,
            allow_sparse_bootstrap=True,
        ) for name in copy_names
    }

    def kl(teacher: str, student: str) -> torch.Tensor:
        row = arms[student]
        return _series(
            row["row_teacher_kl_sum"][teacher],
            row["row_teacher_kl_count"][teacher], weights,
        )

    gain = {name: ce["NNN"] - value for name, value in ce.items()}
    ratios = {
        "R0": kl("OON", "QON") / kl("OON", "NON"),
        "R1": kl("QON", "QQN") / kl("QON", "QNN"),
        "Rjoint": kl("OON", "QQN") / kl("OON", "NNN"),
    }
    margins = {
        "site0_gain": gain["QNN"],
        "site1_increment": gain["QQN"] - gain["QNN"],
        "joint_beats_singletons": gain["QQN"] - torch.maximum(
            gain["QNN"], gain["NQN"]
        ),
        "oracle_half": gain["QQN"] - 0.5 * gain["OON"],
        "mlp2_background": ce["NNE"] - ce["QQE"],
        "v21_control_superiority": gain["QQN"] - torch.maximum(
            gain["MMN"], gain["SSN"]
        ),
        "v21_label_alignment": kl("OON", "SSN") - kl("OON", "QQN"),
    }
    modular = {
        "site0_N_context": ce["NON"] - ce["QON"],
        "site0_E_context": ce["NOE"] - ce["QOE"],
        "site1_N_context": ce["ONN"] - ce["OQN"],
        "site1_E_context": ce["ONE"] - ce["OQE"],
    }
    copy_worsening = {
        "QQN_vs_NNN": copy["QQN"] - copy["NNN"],
        "QQE_vs_NNE": copy["QQE"] - copy["NNE"],
    }
    frequency_reports = {}
    frequency_gate = True
    for candidate, background in (("QQN", "NNN"), ("QQE", "NNE")):
        cells = []
        for bin_index in range(9):
            candidate_count = arms[candidate]["row_frequency_count"][:, bin_index]
            background_count = arms[background]["row_frequency_count"][:, bin_index]
            if not torch.equal(candidate_count, background_count):
                raise RuntimeError("v2.1 final frequency supports differ by arm")
            if int(candidate_count.sum()) == 0:
                cells.append({"bin": bin_index, "status": "empty_unevaluable"})
                frequency_gate = False
                continue
            candidate_series = _series(
                arms[candidate]["row_frequency_ce_sum"][:, bin_index],
                candidate_count, weights, allow_sparse_bootstrap=True,
            )
            background_series = _series(
                arms[background]["row_frequency_ce_sum"][:, bin_index],
                background_count, weights, allow_sparse_bootstrap=True,
            )
            worsening = candidate_series - background_series
            summary = _summary(worsening)
            cells.append({
                "bin": bin_index, "status": "nonempty",
                "worsening": summary,
            })
            frequency_gate = frequency_gate and float(worsening[0]) <= 0.01 and (
                summary["ci_status"] == "estimated_all_registered_draws"
            )
        frequency_reports[f"{candidate}_vs_{background}"] = cells

    family_gain = {
        family: ce["NNN"] - ce[f"same::{family}"] for family in FAMILIES
    }
    a, b, c, d, e = FAMILIES
    descriptive_contrasts = {
        "state": family_gain[b] - family_gain[a],
        "objective": 0.5 * (
            (family_gain[c] - family_gain[b]) + (family_gain[e] - family_gain[d])
        ),
        "grammar": 0.5 * (
            (family_gain[d] - family_gain[b]) + (family_gain[e] - family_gain[c])
        ),
    }
    mechanism = {
        "claim_status": "descriptive_context_confounded_by_registered_sequential_parent",
        "contrasts": {
            name: _summary(series) for name, series in descriptive_contrasts.items()
        },
        "holm_descriptive_only": _holm_positive(descriptive_contrasts),
        "interaction": _summary(
            family_gain[e] - family_gain[d] - family_gain[c] + family_gain[b]
        ),
        "parent_consistent_site0": {
            family: _summary(ce["NNN"] - ce[f"site0::{family}"])
            for family in FAMILIES
        },
        "fixed_T0_site1": {
            family: _summary(ce["QNN"] - ce[f"site1::{family}"])
            for family in FAMILIES
        },
    }
    margin_reports = {name: _summary(value) for name, value in margins.items()}
    modular_reports = {name: _summary(value) for name, value in modular.items()}
    ratio_reports = {name: _summary(value) for name, value in ratios.items()}
    gates = {
        "ratio_denominators_positive": all(
            float(value[0]) > 0 for value in (
                kl("OON", "NON"), kl("QON", "QNN"), kl("OON", "NNN")
            )
        ),
        "ratios_point_le_0_50": all(
            0 <= float(value[0]) <= 0.50 for value in ratios.values()
        ),
        "ordered_site0_positive": _positive_with_full_ci(
            margin_reports["site0_gain"]
        ),
        "ordered_site1_positive": _positive_with_full_ci(
            margin_reports["site1_increment"]
        ),
        "joint_beats_singletons": _positive_with_full_ci(
            margin_reports["joint_beats_singletons"]
        ),
        "oracle_half": _positive_with_full_ci(margin_reports["oracle_half"])
        and float(gain["OON"][0]) > 0,
        "mlp2_background": _positive_with_full_ci(
            margin_reports["mlp2_background"]
        ),
        "true_beats_mean_and_shuffle": _positive_with_full_ci(
            margin_reports["v21_control_superiority"]
        ),
        "label_alignment": _positive_with_full_ci(
            margin_reports["v21_label_alignment"]
        ),
        "copy_collateral": max(
            float(series[0]) for series in copy_worsening.values()
        ) <= 0.01 and all(
            _summary(series)["ci_status"] == "estimated_all_registered_draws"
            for series in copy_worsening.values()
        ),
        "frequency_collateral": frequency_gate,
    }
    package_admitted = all(gates.values())
    modular_claim = package_admitted and all(
        _positive_with_full_ci(report) for report in modular_reports.values()
    )
    prices = bundle["prices"]
    return {
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "source-document cluster; row-weighted estimands",
            "unique_documents": len(documents),
        },
        "arm_ce": {name: _summary(value) for name, value in ce.items()},
        "ratios": ratio_reports,
        "margins": margin_reports,
        "modularity": modular_reports,
        "copy_worsening": {
            name: _summary(value) for name, value in copy_worsening.items()
        },
        "frequency_collateral": frequency_reports,
        "mechanism_bank": mechanism,
        "prices": {
            "currency": "registered standalone real-valued tensor count",
            "true": prices["true"], "shuffle": prices["shuffle"],
            "true_no_greater_than_shuffle": prices["true"]["total_reals"]
            <= prices["shuffle"]["total_reals"],
        },
        "registered_gates": gates,
        "package_admitted": package_admitted,
        "claim_scope": (
            "modular_sites" if modular_claim else
            "ordered_T0_to_T1_package" if package_admitted else "negative"
        ),
        "strong_ratio_rung_0_25": all(
            float(value[0]) <= 0.25 for value in ratios.values()
        ),
        "mlp2_diagnostic_QQE_minus_QQN": _summary(ce["QQN"] - ce["QQE"]),
    }


def _family_programs(bundle: Mapping[str, Any]) -> dict[str, dict[int, Any]]:
    ledgers = bundle["candidate_ledgers"]
    names = bundle["family_representatives"]
    output = {}
    for family in FAMILIES:
        state0 = ledgers["true_site0"][names["true_site0"][family]]["state"]
        state1 = ledgers["true_site1"][names["true_site1"][family]]["state"]
        output[f"same::{family}"] = {0: state0, 1: state1}
        output[f"site0::{family}"] = {0: state0}
        output[f"site1::{family}"] = {
            0: bundle["programs"]["true"][0], 1: state1,
        }
    return output


def _expected_arm_names() -> set[str]:
    lattice = {
        "".join(values) for values in itertools.product("NQO", "NQO", "NE")
    }
    return lattice | {"SSN", "MMN"} | {
        f"{prefix}::{family}"
        for prefix in ("same", "site0", "site1") for family in FAMILIES
    }


def validate_final_result(
    value: Any, *, attempt: Mapping[str, Any], unlock: Mapping[str, Any],
    protected_before: Mapping[str, Any], protected_after: Mapping[str, Any],
) -> None:
    """Semantically rederive every authority-facing final result field."""

    required = {
        "schema_version", "status", "authority",
        "authorized_for_scored_experiments", "attempt", "program_unlock",
        "program_bundle", "final_cache", "source_commit", "source_hashes",
        "arms", "analysis", "complexity_reports", "integrity_subgates", "integrity",
        "execution_closure", "document_provenance", "diagnostics",
    }
    if not isinstance(value, Mapping) or set(value) != required or (
        value.get("schema_version") != 1
        or value.get("status")
        != "completed_v21_final_pending_last_written_outcome_authority"
        or value.get("authority") != "none"
        or value.get("authorized_for_scored_experiments") is not False
        or value.get("attempt") != _binding(ATTEMPT)
        or value.get("program_unlock") != _binding(authority.PROGRAMS_RECEIPT)
        or value.get("program_bundle") != _binding(authority.PROGRAMS_ARTIFACT)
        or not authority._same_value(value.get("final_cache"), attempt["final_cache"])
        or value.get("source_commit") != unlock["source_commit"]
        or value.get("source_hashes") != unlock["source_hashes"]
    ):
        raise RuntimeError("v2.1 final result authority bindings changed")

    bundle = torch.load(
        authority.PROGRAMS_ARTIFACT, map_location="cpu", weights_only=True,
    )
    authority._validate_program_bundle(bundle)
    arms = value["arms"]
    expected_names = _expected_arm_names()
    raw_keys = {
        "name", "scorer", "row_ce_sum", "row_ce_count", "row_copy_ce_sum",
        "row_copy_count", "row_frequency_ce_sum", "row_frequency_count",
        "row_teacher_kl_sum", "row_teacher_kl_count",
        "original_mlp_call_counters",
    }
    if not isinstance(arms, Mapping) or set(arms) != expected_names:
        raise RuntimeError("v2.1 final result arm coverage changed")
    for name, arm in arms.items():
        if not isinstance(arm, Mapping) or set(arm) != raw_keys or (
            arm.get("name") != name or arm.get("scorer") != SCORER
        ):
            raise RuntimeError(f"v2.1 final raw arm schema changed: {name}")
        validate_arm_statistics(arm)
        expected_teachers = set() if name == "OON" else (
            {"OON"} if name == "QON" else {"OON", "QON"}
        )
        if set(arm["row_teacher_kl_sum"]) != expected_teachers:
            raise RuntimeError(f"v2.1 final teacher coverage changed: {name}")
        expected_calls = {0: 0, 1: 0, 2: 0}
        if len(name) == 3 and name[0] in "NQO":
            _, allowed = _states_for_arm(name)
            expected_calls = {
                site: (24 if site in allowed else 0) for site in (0, 1, 2)
            }
        if arm["original_mlp_call_counters"] != expected_calls:
            raise RuntimeError(f"v2.1 final original-call counter changed: {name}")

    provenance = value["document_provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "document_ids", "document_ids_sha256", "row_to_document_cluster",
    }:
        raise RuntimeError("v2.1 final provenance schema changed")
    document_ids = provenance["document_ids"]
    row_authority = json.loads(authority.RECEIPT.read_text())
    final_records = row_authority.get("document_provenance", {}).get(
        "sets", {}
    ).get("compiler_final_v21")
    final_entry = row_authority.get("entries", {}).get("compiler_final_v21", {})
    expected_document_ids = (
        [record["document_id"] for record in final_records]
        if isinstance(final_records, list) and all(
            isinstance(record, Mapping) and isinstance(record.get("document_id"), str)
            for record in final_records
        ) else []
    )
    unique = list(dict.fromkeys(document_ids)) if isinstance(document_ids, list) else []
    expected_clusters = [unique.index(item) for item in document_ids] if unique else []
    expected_document_hash = authority.logical_json_sha256(expected_document_ids)
    if len(document_ids) != 192 or document_ids != expected_document_ids or (
        provenance["document_ids_sha256"] != expected_document_hash
    ) or final_entry.get("document_ids_sha256") != expected_document_hash or (
        final_entry.get("provenance_records_sha256")
        != authority.logical_json_sha256(final_records)
    ) or provenance["row_to_document_cluster"] != expected_clusters:
        raise RuntimeError("v2.1 final document provenance changed")
    recomputed = analyze(arms, document_ids, bundle)
    if not authority._same_value(value["analysis"], recomputed):
        raise RuntimeError("v2.1 final analysis does not recompute from raw arms")

    subgates = value["integrity_subgates"]
    expected_subgates = {
        "final_role_loaded_exactly_once", "final_evaluation_callback_exactly_once",
        "baseline_replay_bit_identical", "signed_gauge_row_ce_drift_le_2e_6",
        "signed_gauge_physical_drift_le_2e_6", "native_gauge_canonical",
        "hook_restored_and_inert", "outer_model_returned",
        "component_tree_unchanged", "protected_unchanged",
        "program_unlock_revalidated",
    }
    closure = value["execution_closure"]
    diagnostics = value["diagnostics"]
    if not isinstance(subgates, Mapping) or set(subgates) != expected_subgates or any(
        not isinstance(flag, bool) for flag in subgates.values()
    ) or value["integrity"] is not all(subgates.values()) or not isinstance(
        closure, Mapping
    ) or set(closure) != {
        "outer_model_returned", "hook_restored_and_inert",
        "component_tree_before", "component_tree_after",
    } or not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "final_role_loads", "final_evaluation_callbacks",
        "baseline_replay_bit_identical", "signed_gauge_row_ce_max_abs_drift",
        "baseline_replay_statistics",
        "signed_gauge_physical_max_abs_drift", "native_gauge", "runtime_s",
        "peak_cpu_rss_bytes", "peak_gpu_allocated_bytes",
    } or dict(protected_before) != dict(protected_after):
        raise RuntimeError("v2.1 final integrity/diagnostic schema changed")
    replay = diagnostics["baseline_replay_statistics"]
    baseline_expected = {key: arms["NNN"][key] for key in BASELINE_STAT_KEYS}
    if not isinstance(replay, Mapping) or set(replay) != set(BASELINE_STAT_KEYS):
        raise RuntimeError("v2.1 final baseline replay evidence changed")
    baseline_identical = authority._same_value(replay, baseline_expected)
    native_expected = native_gauge_audit(bundle)
    if not authority._same_value(diagnostics["native_gauge"], native_expected):
        raise RuntimeError("v2.1 final native gauge evidence does not recompute")
    row_drift = diagnostics["signed_gauge_row_ce_max_abs_drift"]
    physical_drift = diagnostics["signed_gauge_physical_max_abs_drift"]
    if not all(isinstance(item, (int, float)) and math.isfinite(item) and item >= 0
               for item in (row_drift, physical_drift)):
        raise RuntimeError("v2.1 final signed gauge evidence changed")
    derived_subgates = {
        "final_role_loaded_exactly_once": diagnostics["final_role_loads"] == 1,
        "final_evaluation_callback_exactly_once": (
            diagnostics["final_evaluation_callbacks"] == 1
        ),
        "baseline_replay_bit_identical": baseline_identical,
        "signed_gauge_row_ce_drift_le_2e_6": row_drift <= 2e-6,
        "signed_gauge_physical_drift_le_2e_6": physical_drift <= 2e-6,
        "native_gauge_canonical": native_expected["passed"] is True,
        "hook_restored_and_inert": closure["hook_restored_and_inert"] is True,
        "outer_model_returned": closure["outer_model_returned"] is True,
        "component_tree_unchanged": (
            bool(closure["component_tree_before"])
            and closure["component_tree_before"] == closure["component_tree_after"]
        ),
        "protected_unchanged": dict(protected_before) == dict(protected_after),
        "program_unlock_revalidated": authority.validate_final_unlock(
            authority.PROGRAMS_RECEIPT
        ) == unlock,
    }
    if not authority._same_value(subgates, derived_subgates) or (
        diagnostics["baseline_replay_bit_identical"] is not baseline_identical
    ) or value["integrity"] is not all(derived_subgates.values()):
        raise RuntimeError("v2.1 final integrity subgates do not recompute")
    expected_complexity = required_complexity_reports(
        bundle, final_runtime_s=diagnostics["runtime_s"],
        final_peak_cpu_rss_bytes=diagnostics["peak_cpu_rss_bytes"],
        final_peak_gpu_allocated_bytes=diagnostics["peak_gpu_allocated_bytes"],
    )
    if not authority._same_value(value["complexity_reports"], expected_complexity):
        raise RuntimeError("v2.1 final complexity reports do not recompute")


def validate_manifest_candidate(
    value: Any, *, result: Mapping[str, Any], attempt: Mapping[str, Any],
    unlock: Mapping[str, Any], protected_before: Mapping[str, Any],
    protected_after: Mapping[str, Any],
) -> None:
    required = {
        "schema_version", "status", "authority",
        "authorized_for_scored_experiments", "attempt", "result",
        "program_unlock", "program_bundle", "final_cache", "source_commit",
        "source_hashes", "protected_before", "protected_after", "integrity",
        "package_admitted", "claim_scope", "final_role_loads",
        "final_evaluations",
    }
    expected = {
        "schema_version": 1,
        "status": "completed_v21_final_pending_last_written_outcome_authority",
        "authority": "none", "authorized_for_scored_experiments": False,
        "attempt": _binding(ATTEMPT), "result": _binding(RESULT),
        "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
        "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
        "final_cache": attempt["final_cache"], "source_commit": unlock["source_commit"],
        "source_hashes": unlock["source_hashes"],
        "protected_before": protected_before, "protected_after": protected_after,
        "integrity": result["integrity"],
        "package_admitted": result["analysis"]["package_admitted"],
        "claim_scope": result["analysis"]["claim_scope"],
        "final_role_loads": result["diagnostics"]["final_role_loads"],
        "final_evaluations": result["diagnostics"]["final_evaluation_callbacks"],
    }
    if not isinstance(value, Mapping) or set(value) != required or any(
        not authority._same_value(value.get(key), item) for key, item in expected.items()
    ):
        raise RuntimeError("v2.1 final manifest candidate changed")


def validate_outcome_candidate(
    value: Any, *, result: Mapping[str, Any], attempt: Mapping[str, Any],
    unlock: Mapping[str, Any],
) -> None:
    required = {
        "schema_version", "status", "authority",
        "authorized_for_scored_experiments", "attempt", "result", "manifest",
        "program_unlock", "program_bundle", "final_cache", "source_commit",
        "source_hashes", "execution_closure", "integrity", "package_admitted",
        "claim_scope", "final_role_loads", "final_evaluations", "last_write_rule",
    }
    expected_status = (
        "authoritative_positive_v21_final" if result["analysis"]["package_admitted"]
        else "authoritative_negative_v21_final"
    )
    if not isinstance(value, Mapping) or set(value) != required or (
        value.get("schema_version") != 1 or value.get("status") != expected_status
        or value.get("authority") != "compiler_v21_scientific_outcome"
        or value.get("authorized_for_scored_experiments") is not True
        or value.get("attempt") != _binding(ATTEMPT)
        or value.get("result") != _binding(RESULT)
        or value.get("manifest") != _binding(MANIFEST)
        or value.get("program_unlock") != _binding(authority.PROGRAMS_RECEIPT)
        or value.get("program_bundle") != _binding(authority.PROGRAMS_ARTIFACT)
        or not authority._same_value(value.get("final_cache"), attempt["final_cache"])
        or value.get("source_commit") != unlock["source_commit"]
        or value.get("source_hashes") != unlock["source_hashes"]
        or value.get("execution_closure") != result["execution_closure"]
        or value.get("integrity") is not True
        or value.get("package_admitted") is not result["analysis"]["package_admitted"]
        or value.get("claim_scope") != result["analysis"]["claim_scope"]
        or value.get("final_role_loads") != 1 or value.get("final_evaluations") != 1
        or not isinstance(value.get("last_write_rule"), str)
    ):
        raise RuntimeError("v2.1 final outcome candidate changed")


def _program_encoding_report(state: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value for key, value in sorted(state.items())
        if not torch.is_tensor(value)
    }
    metadata_bytes = len(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))
    tensors = {
        key: {
            "dtype": str(value.dtype), "shape": list(value.shape),
            "raw_bytes": int(value.numel() * value.element_size()),
        }
        for key, value in sorted(state.items()) if torch.is_tensor(value)
    }
    tensor_bytes = sum(item["raw_bytes"] for item in tensors.values())
    index_count = int(state["indices"].numel()) if state.get("grammar") == "native" else 0
    bits_per_native_index = math.ceil(math.log2(compiler.NATIVE_PRODUCTS))
    return {
        "grammar": state["grammar"], "metadata": metadata,
        "metadata_encoding": "canonical compact UTF-8 JSON with sorted keys",
        "metadata_bytes": metadata_bytes, "tensor_payloads": tensors,
        "tensor_raw_bytes": tensor_bytes, "native_index_count": index_count,
        "native_index_encoding": (
            f"fixed-width {bits_per_native_index}-bit index into "
            f"{compiler.NATIVE_PRODUCTS} native products"
        ),
        "native_index_encoded_bits": index_count * bits_per_native_index,
        "serialized_index_tensor_bits": index_count * 64,
    }


def _validate_selected_fit_numerics(
    manifest: Mapping[str, Any], *, site: int, bundle: Mapping[str, Any],
) -> None:
    reports = manifest.get("selected_fit_numerics")
    if not isinstance(reports, Mapping) or set(reports) != {"true", "shuffle", "mean"}:
        raise RuntimeError(f"v2.1 site{site} fit-numerics report is absent")
    selected_manifest = manifest.get("selected")
    if not isinstance(selected_manifest, Mapping) or set(selected_manifest) != {
        "true", "shuffle"
    }:
        raise RuntimeError(f"v2.1 site{site} selected manifest changed")
    replay_keys = {
        "status", "support_positions", "reference", "deployed",
        "max_abs_coefficient_drift", "rms_coefficient_drift",
    }
    condition_keys = {
        "matrix", "rows", "columns", "minimum_gram_eigenvalue",
        "maximum_gram_eigenvalue", "condition_number_by_lambda",
    }
    for arm in ("true", "shuffle", "mean"):
        report = reports[arm]
        expected_name = (
            bundle["selection_receipts"][f"{arm}_site{site}"]["selected"]
            if arm != "mean" else f"mean_site{site}"
        )
        state = bundle["programs"][arm][site]
        if not isinstance(report, Mapping) or set(report) != {
            "selected", "grammar", "ridge_condition_numbers",
            "float64_to_float32_replay", "quantization_status",
        } or report.get("selected") != expected_name or report.get(
            "grammar"
        ) != state["grammar"] or report.get("quantization_status") != (
            "none; all floating parameter tensors float32; native indices int64"
        ) or (arm != "mean" and selected_manifest.get(arm) != expected_name):
            raise RuntimeError(f"v2.1 site{site} {arm} fit-numerics binding changed")
        condition = report["ridge_condition_numbers"]
        if not isinstance(condition, Mapping) or set(condition) != condition_keys or (
            condition["matrix"] != "normalized fit Gram plus lambda I"
            or condition["rows"] != authority.FIT_CAPTURE_COUNT
            or condition["columns"] != compiler.D_MODEL
        ):
            raise RuntimeError(f"v2.1 site{site} {arm} condition schema changed")
        minimum = condition["minimum_gram_eigenvalue"]
        maximum = condition["maximum_gram_eigenvalue"]
        if not all(isinstance(item, (int, float)) and math.isfinite(item)
                   for item in (minimum, maximum)) or maximum < minimum:
            raise RuntimeError(f"v2.1 site{site} {arm} condition spectrum changed")
        by_lambda = condition["condition_number_by_lambda"]
        expected_lambdas = {str(float(value)) for value in old_site0.affine_v1.LAMBDA_GRID}
        if not isinstance(by_lambda, Mapping) or set(by_lambda) != expected_lambdas:
            raise RuntimeError(f"v2.1 site{site} {arm} ridge ladder changed")
        for key, cell in by_lambda.items():
            ridge = float(key)
            denominator = minimum + ridge
            expected_status = "singular_or_indefinite" if denominator <= 0 else "evaluated"
            expected_value = None if denominator <= 0 else (maximum + ridge) / denominator
            if not isinstance(cell, Mapping) or set(cell) != {"status", "value"} or (
                cell["status"] != expected_status
            ) or not authority._same_value(cell["value"], expected_value):
                raise RuntimeError(f"v2.1 site{site} {arm} condition does not recompute")
        replay = report["float64_to_float32_replay"]
        if not isinstance(replay, Mapping) or set(replay) != replay_keys or (
            replay["status"] != "evaluated_serialized_float32_parameters"
            or replay["support_positions"] != 64
            or replay["reference"] != "float64 accumulation"
            or replay["deployed"] != "float32 accumulation"
        ) or any(
            not isinstance(replay[key], (int, float)) or not math.isfinite(replay[key])
            or replay[key] < 0
            for key in ("max_abs_coefficient_drift", "rms_coefficient_drift")
        ):
            raise RuntimeError(f"v2.1 site{site} {arm} precision replay changed")


def required_complexity_reports(
    bundle: Mapping[str, Any], *, final_runtime_s: float,
    final_peak_cpu_rss_bytes: int, final_peak_gpu_allocated_bytes: int,
) -> dict[str, Any]:
    site0_manifest = json.loads(authority.SITE0_MANIFEST.read_text())
    site1_manifest = json.loads(authority.SITE1_MANIFEST.read_text())
    selected_encoding = {
        arm: {
            f"site{site}": _program_encoding_report(bundle["programs"][arm][site])
            for site in (0, 1)
        } for arm in ("true", "shuffle", "mean")
    }
    prices = bundle["prices"]
    computation = {}
    for arm in ("true", "shuffle", "mean"):
        multiplies = int(prices[arm]["inference_multiplies_per_token"])
        hadamards = int(prices[arm]["hadamard_products_per_token"])
        computation[arm] = {
            "inference_multiplies_per_token": multiplies,
            "hadamard_products_per_token": hadamards,
            "flop_convention": "two FLOPs per multiply-accumulate plus one per Hadamard",
            "inference_flops_per_token": 2 * multiplies + hadamards,
        }
    original_multiplies = 3 * compiler.NATIVE_PRODUCTS * compiler.D_MODEL
    original_reals = original_multiplies + compiler.D_MODEL
    artifacts = {
        name: _binding(path) for name, path in {
            "program_bundle": authority.PROGRAMS_ARTIFACT,
            "site0_ledger": authority.SITE0_LEDGER_ARTIFACT,
            "site1_ledger": authority.SITE1_LEDGER_ARTIFACT,
            "site0_manifest": authority.SITE0_MANIFEST,
            "site1_manifest": authority.SITE1_MANIFEST,
        }.items()
    }
    for site, manifest in enumerate((site0_manifest, site1_manifest)):
        _validate_selected_fit_numerics(manifest, site=site, bundle=bundle)
        for key in ("runtime_s", "peak_cpu_rss_bytes", "peak_gpu_allocated_bytes"):
            if not isinstance(manifest.get(key), (int, float)) or manifest[key] < 0:
                raise RuntimeError(f"v2.1 site{site} runtime/memory report is absent")
    return {
        "currency": {
            "selection": "registered standalone real-valued tensor count",
            "amortized_guardrail": (
                "basis may be omitted only under an independently admitted shared-basis library"
            ),
            "prices": prices,
        },
        "serialized_programs": selected_encoding,
        "artifact_bytes": artifacts,
        "computation": computation,
        "runtime_and_peak_memory": {
            "site0": {
                key: site0_manifest[key] for key in (
                    "runtime_s", "peak_cpu_rss_bytes", "peak_gpu_allocated_bytes"
                )
            },
            "site1": {
                key: site1_manifest[key] for key in (
                    "runtime_s", "peak_cpu_rss_bytes", "peak_gpu_allocated_bytes"
                )
            },
            "final": {
                "runtime_s": float(final_runtime_s),
                "peak_cpu_rss_bytes": int(final_peak_cpu_rss_bytes),
                "peak_gpu_allocated_bytes": int(final_peak_gpu_allocated_bytes),
            },
        },
        "fit_numerics": {
            "site0": site0_manifest["selected_fit_numerics"],
            "site1": site1_manifest["selected_fit_numerics"],
            "quantization_status": (
                "none; all floating parameter tensors are serialized float32; "
                "native indices are int64; no weight quantization"
            ),
        },
        "exact_search_budget": {
            "candidate_banks": 4, "cells_per_bank": 108,
            "candidate_cells_fit_and_validation_scored": 432,
            "affine_cells_per_bank": 96, "native_cells_per_bank": 12,
            "affine_cells_total": 384,
            "causal_affine_C_trajectories": 128,
            "causal_affine_epochs_per_C_cell": int(
                old_site0.fit.CAUSAL_EPOCHS
            ),
            "causal_affine_minibatch_size": int(old_site0.fit.CAUSAL_BATCH),
            "native_objective_paths": 8,
            "native_l1_ratios": list(old_site0.native_solver.L1_RATIOS),
            "native_l1_trajectories": int(
                8 * len(old_site0.native_solver.L1_RATIOS)
            ),
            "native_power_iterations_per_path": int(
                old_site0.native_solver.POWER_ITERATIONS
            ),
            "native_fista_iterations_per_l1_trajectory": int(
                old_site0.native_solver.FISTA_ITERATIONS
            ),
            "native_refit_frontier_solves": int(8 * len(compiler.NATIVE_K_GRID)),
            "mean_controls_fit_and_scored": 2,
            "full_native_adequacy_controls": 3,
            "final_scored_primary_arms": 35,
            "final_additional_replays": 2,
        },
        "full_original_comparator": {
            "per_site_reals": original_reals,
            "two_site_reals": 2 * original_reals,
            "per_site_multiplies_per_token": original_multiplies,
            "per_site_hadamard_products_per_token": compiler.NATIVE_PRODUCTS,
            "two_site_flops_per_token": 2 * (
                2 * original_multiplies + compiler.NATIVE_PRODUCTS
            ),
            "compiled_QQ_original_mlp_calls": {0: 0, 1: 0, 2: 0},
        },
    }


def _run_final(
    final_rows_full: torch.Tensor, row_receipt: Mapping[str, Any],
    unlock: Mapping[str, Any], attempt: Mapping[str, Any],
    protected_before: Mapping[str, Any], *, final_role_loads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_rows = final_rows_full[:, :257].contiguous()
    frequency = _frequency_assignments(row_receipt, final_rows_full)
    records = row_receipt["document_provenance"]["sets"]["compiler_final_v21"]
    document_ids = [record["document_id"] for record in records]
    bundle = torch.load(authority.PROGRAMS_ARTIFACT, map_location="cpu", weights_only=True)
    authority._validate_program_bundle(bundle)
    old_receipt, _ = old_site0.old_rows.validate_receipt()
    code_rows, _ = old_site0.code_oracle.load_frozen_corpus()
    old_site0.frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = old_site0.v3.validate_basis_pair()
    torch.manual_seed(old_site0.exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(old_site0.exact_runner.SHIP_SEED)
    sys.path.insert(0, str(authority.BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    callback_count = 0
    callback_state: dict[str, Any] = {}
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        nonlocal callback_count
        callback_count += 1
        if callback_count != 1:
            raise RuntimeError("v2.1 final model callback ran more than once")
        realization, _ = old_site0.frozen.restore_ship_realization(
            sa, twall, all_attention, old_receipt, code_rows
        )
        if realization != old_site0.SHIP_HASH:
            raise RuntimeError("v2.1 frozen ship realization changed")
        component_before = old_site0.exact_runner.component_tree_sha256(
            sa, twall, all_attention
        )
        bases = {
            site: bases_payload["sites"][site]["basis"].to(sa.DEV).float()
            for site in (0, 1)
        }
        hook = old_site0.runtime.StateCompleteCorrectionHook(bases, {})
        prior_hook = sa.add_oracle_correction
        sa.add_oracle_correction = hook
        arms: dict[str, Any] = {}
        try:
            lattice = [
                "".join(values) for values in itertools.product("NQO", "NQO", "NE")
            ]
            # Teacher arms first; only their batch logits are retained temporarily.
            states, allowed = _states_for_arm("OON")
            arms["OON"], oon_logits = score_arm(
                sa, hook, final_rows, frequency, twall, all_attention, name="OON",
                programs={"true": bundle["programs"]["true"]},
                program_name="true", states=states, allowed=allowed,
                retain_logits=True,
            )
            states, allowed = _states_for_arm("QON")
            arms["QON"], qon_logits = score_arm(
                sa, hook, final_rows, frequency, twall, all_attention, name="QON",
                programs={"true": bundle["programs"]["true"]},
                program_name="true", states=states, allowed=allowed,
                teachers={"OON": oon_logits}, retain_logits=True,
            )
            teachers = {"OON": oon_logits, "QON": qon_logits}
            for code in lattice:
                if code in arms:
                    continue
                states, allowed = _states_for_arm(code)
                arms[code], _ = score_arm(
                    sa, hook, final_rows, frequency, twall, all_attention,
                    name=code, programs={"true": bundle["programs"]["true"]},
                    program_name="true", states=states, allowed=allowed,
                    teachers=teachers,
                )
            for name, arm in (("SSN", "shuffle"), ("MMN", "mean")):
                arms[name], _ = score_arm(
                    sa, hook, final_rows, frequency, twall, all_attention,
                    name=name, programs={arm: bundle["programs"][arm]},
                    program_name=arm, states={0: "Q", 1: "Q"}, allowed=set(),
                    teachers=teachers,
                )
            for name, programs in _family_programs(bundle).items():
                states = {site: "Q" for site in programs}
                arms[name], _ = score_arm(
                    sa, hook, final_rows, frequency, twall, all_attention,
                    name=name, programs={name: programs}, program_name=name,
                    states=states, allowed=set(), teachers=teachers,
                )

            baseline_replay, _ = score_arm(
                sa, hook, final_rows, frequency, twall, all_attention,
                name="NNN_replay", programs={"true": bundle["programs"]["true"]},
                program_name="true", states={}, allowed=set(), teachers=teachers,
            )
            baseline_identical = authority._same_value(
                {key: arms["NNN"][key] for key in BASELINE_STAT_KEYS},
                {key: baseline_replay[key] for key in BASELINE_STAT_KEYS},
            )
            moved_programs, moved_bases = signed_gauge_programs(
                bundle["programs"]["true"], bases
            )
            gauge_physical_drift = signed_gauge_physical_drift(
                bundle["programs"]["true"], bases, moved_programs, moved_bases,
            )
            gauge_hook = old_site0.runtime.StateCompleteCorrectionHook(
                {site: basis.to(sa.DEV) for site, basis in moved_bases.items()}, {}
            )
            sa.add_oracle_correction = gauge_hook
            gauge_stats, _ = score_arm(
                sa, gauge_hook, final_rows, frequency, twall, all_attention,
                name="QQN_signed_gauge", programs={"gauge": moved_programs},
                program_name="gauge", states={0: "Q", 1: "Q"}, allowed=set(),
            )
            gauge_row_ce = gauge_stats["row_ce_sum"] / gauge_stats["row_ce_count"]
            true_row_ce = arms["QQN"]["row_ce_sum"] / arms["QQN"]["row_ce_count"]
            gauge_drift = float((gauge_row_ce - true_row_ce).abs().max())
            gauge_hook.clear()
            sa.add_oracle_correction = hook
            native_gauge = native_gauge_audit(bundle)
            analysis = analyze(arms, document_ids, bundle)
            component_after = old_site0.exact_runner.component_tree_sha256(
                sa, twall, all_attention
            )
            if component_after != component_before:
                raise RuntimeError("v2.1 component tree changed during final")
            callback_state.update({
                "component_before": component_before,
                "component_after": component_after,
                "arms": arms,
                "analysis": analysis,
                "baseline_replay_bit_identical": baseline_identical,
                "baseline_replay_statistics": {
                    key: baseline_replay[key] for key in BASELINE_STAT_KEYS
                },
                "signed_gauge_row_ce_max_abs_drift": gauge_drift,
                "signed_gauge_physical_max_abs_drift": gauge_physical_drift,
                "native_gauge": native_gauge,
            })
        finally:
            hook.clear()
            sa.add_oracle_correction = prior_hook
            old_site0.exact_runner.require_inert_correction_state(sa)
            callback_state["hook_restored"] = True

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)
    closure = lifecycle.close_execution(
        sa, outer_model_returned=True,
        component_tree_before=callback_state.get("component_before", ""),
        component_tree_after=callback_state.get("component_after", ""),
    )
    protected_after = authority.protected_snapshot()
    runtime_s = round(time.time() - started, 1)
    peak_cpu_rss_bytes = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    peak_gpu_allocated_bytes = int(
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    integrity = {
        "final_role_loaded_exactly_once": final_role_loads == 1,
        "final_evaluation_callback_exactly_once": callback_count == 1,
        "baseline_replay_bit_identical": callback_state.get(
            "baseline_replay_bit_identical"
        ) is True,
        "signed_gauge_row_ce_drift_le_2e_6": callback_state.get(
            "signed_gauge_row_ce_max_abs_drift", math.inf
        ) <= 2e-6,
        "signed_gauge_physical_drift_le_2e_6": callback_state.get(
            "signed_gauge_physical_max_abs_drift", math.inf
        ) <= 2e-6,
        "native_gauge_canonical": callback_state.get("native_gauge", {}).get(
            "passed"
        ) is True,
        "hook_restored_and_inert": callback_state.get("hook_restored") is True,
        "outer_model_returned": closure.outer_model_returned,
        "component_tree_unchanged": closure.component_tree_before
        == closure.component_tree_after,
        "protected_unchanged": dict(protected_before) == protected_after,
        "program_unlock_revalidated": authority.validate_final_unlock(
            authority.PROGRAMS_RECEIPT
        ) == unlock,
    }
    result = {
        "schema_version": 1,
        "status": "completed_v21_final_pending_last_written_outcome_authority",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "attempt": _binding(ATTEMPT),
        "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
        "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
        "final_cache": dict(attempt["final_cache"]),
        "source_commit": unlock["source_commit"],
        "source_hashes": dict(unlock["source_hashes"]),
        "arms": callback_state["arms"],
        "analysis": callback_state["analysis"],
        "complexity_reports": required_complexity_reports(
            bundle, final_runtime_s=runtime_s,
            final_peak_cpu_rss_bytes=peak_cpu_rss_bytes,
            final_peak_gpu_allocated_bytes=peak_gpu_allocated_bytes,
        ),
        "integrity_subgates": integrity,
        "integrity": all(integrity.values()),
        "execution_closure": {
            "outer_model_returned": closure.outer_model_returned,
            "hook_restored_and_inert": closure.hook_restored_and_inert,
            "component_tree_before": closure.component_tree_before,
            "component_tree_after": closure.component_tree_after,
        },
        "document_provenance": {
            "document_ids": document_ids,
            "document_ids_sha256": authority.logical_json_sha256(document_ids),
            "row_to_document_cluster": [
                list(dict.fromkeys(document_ids)).index(value)
                for value in document_ids
            ],
        },
        "diagnostics": {
            "final_role_loads": int(final_role_loads),
            "final_evaluation_callbacks": int(callback_count),
            "baseline_replay_bit_identical": callback_state[
                "baseline_replay_bit_identical"
            ],
            "baseline_replay_statistics": callback_state[
                "baseline_replay_statistics"
            ],
            "signed_gauge_row_ce_max_abs_drift": callback_state[
                "signed_gauge_row_ce_max_abs_drift"
            ],
            "signed_gauge_physical_max_abs_drift": callback_state[
                "signed_gauge_physical_max_abs_drift"
            ],
            "native_gauge": callback_state["native_gauge"],
            "runtime_s": runtime_s,
            "peak_cpu_rss_bytes": peak_cpu_rss_bytes,
            "peak_gpu_allocated_bytes": peak_gpu_allocated_bytes,
        },
    }
    return result, protected_after


def main() -> None:
    with lifecycle.exclusive_run_claim():
        if any(path.exists() for path in FINAL_OUTPUTS):
            raise RuntimeError("v2.1 final output namespace is not empty")
        unlock = authority.validate_final_unlock(authority.PROGRAMS_RECEIPT)
        protected_before = authority.protected_snapshot()
        attempt: dict[str, Any] | None = None
        protected_after: dict[str, Any] = dict(protected_before)
        final_forward_begun = False
        final_role_loads = 0
        try:
            attempt = write_attempt_before_final_load(
                unlock, protected_before=protected_before,
            )
            final_role_loads += 1
            row_receipt, final_rows = authority.load_final_for_scoring(
                authority.PROGRAMS_RECEIPT
            )
            final_forward_begun = True
            result, protected_after = _run_final(
                final_rows, row_receipt, unlock, attempt, protected_before,
                final_role_loads=final_role_loads,
            )
            authority.write_torch_atomic(result, RESULT)
            reloaded = torch.load(RESULT, map_location="cpu", weights_only=True)
            if not authority._same_value(reloaded, result):
                raise RuntimeError("v2.1 final result did not reload exactly")
            validate_final_result(
                reloaded, attempt=attempt, unlock=unlock,
                protected_before=protected_before, protected_after=protected_after,
            )
            manifest = {
                "schema_version": 1,
                "status": "completed_v21_final_pending_last_written_outcome_authority",
                "authority": "none",
                "authorized_for_scored_experiments": False,
                "attempt": _binding(ATTEMPT),
                "result": _binding(RESULT),
                "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
                "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
                "final_cache": attempt["final_cache"],
                "source_commit": unlock["source_commit"],
                "source_hashes": dict(unlock["source_hashes"]),
                "protected_before": dict(protected_before),
                "protected_after": dict(protected_after),
                "integrity": result["integrity"],
                "package_admitted": result["analysis"]["package_admitted"],
                "claim_scope": result["analysis"]["claim_scope"],
                "final_role_loads": final_role_loads,
                "final_evaluations": 1,
            }
            validate_manifest_candidate(
                manifest, result=reloaded, attempt=attempt, unlock=unlock,
                protected_before=protected_before, protected_after=protected_after,
            )
            authority.write_json_atomic(manifest, MANIFEST)
            if not result["integrity"]:
                raise RuntimeError(
                    "v2.1 final integrity failed; outcome authority is forbidden"
                )
            # Revalidate every input and cross-binding immediately before authority.
            final_reloaded = torch.load(RESULT, map_location="cpu", weights_only=True)
            validate_final_result(
                final_reloaded, attempt=attempt, unlock=unlock,
                protected_before=protected_before, protected_after=protected_after,
            )
            manifest_reloaded = json.loads(MANIFEST.read_text())
            validate_manifest_candidate(
                manifest_reloaded, result=final_reloaded, attempt=attempt,
                unlock=unlock, protected_before=protected_before,
                protected_after=protected_after,
            )
            if authority.validate_final_unlock(authority.PROGRAMS_RECEIPT) != unlock or (
                json.loads(ATTEMPT.read_text()) != attempt
            ) or manifest_reloaded != manifest or manifest_reloaded["result"] != (
                _binding(RESULT)
            ) or (
                authority.protected_snapshot() != protected_before
            ):
                raise RuntimeError("v2.1 final last-write revalidation failed")
            outcome = {
                "schema_version": 1,
                "status": (
                    "authoritative_positive_v21_final" if result["analysis"][
                        "package_admitted"
                    ] else "authoritative_negative_v21_final"
                ),
                "authority": "compiler_v21_scientific_outcome",
                "authorized_for_scored_experiments": True,
                "attempt": _binding(ATTEMPT),
                "result": _binding(RESULT),
                "manifest": _binding(MANIFEST),
                "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
                "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
                "final_cache": attempt["final_cache"],
                "source_commit": unlock["source_commit"],
                "source_hashes": dict(unlock["source_hashes"]),
                "execution_closure": result["execution_closure"],
                "integrity": True,
                "package_admitted": result["analysis"]["package_admitted"],
                "claim_scope": result["analysis"]["claim_scope"],
                "final_role_loads": final_role_loads,
                "final_evaluations": 1,
                "last_write_rule": (
                    "Written only after result, manifest, all inputs, source, "
                    "protected artifacts, hook restoration, and output hashes revalidated."
                ),
            }
            validate_outcome_candidate(
                outcome, result=final_reloaded, attempt=attempt, unlock=unlock,
            )
            authority.write_json_atomic(outcome, OUTCOME_AUTHORITY)
        except Exception as error:
            if OUTCOME_AUTHORITY.exists():
                raise
            # No output was spent if attempt publication itself never happened.
            if not ATTEMPT.is_file():
                raise
            attempt_on_disk: Any = None
            try:
                attempt_on_disk = json.loads(ATTEMPT.read_text())
            except Exception:
                pass
            prior_manifest = None
            if MANIFEST.is_file():
                prior_manifest = {
                    "sha256_before_failure_record": authority.file_sha256(MANIFEST),
                    "bytes_before_failure_record": MANIFEST.stat().st_size,
                }
            preserved_outputs = {
                "attempt": _binding(ATTEMPT),
                "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
                "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
            }
            if RESULT.is_file():
                preserved_outputs["result"] = _binding(RESULT)
            failure = {
                "schema_version": 1,
                "attempt": _binding(ATTEMPT),
                "program_unlock": _binding(authority.PROGRAMS_RECEIPT),
                "program_bundle": _binding(authority.PROGRAMS_ARTIFACT),
                "final_cache": (
                    attempt_on_disk.get("final_cache")
                    if isinstance(attempt_on_disk, Mapping) else None
                ),
                "source_commit": unlock["source_commit"],
                "source_hashes": dict(unlock["source_hashes"]),
                "preserved_outputs": preserved_outputs,
                "prior_manifest_snapshot": prior_manifest,
            }
            failure.update({
                "status": "failed_v21_final_without_outcome_authority",
                "authority": "none",
                "authorized_for_scored_experiments": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "final_forward_begun": final_forward_begun,
                "retry": "new committed protocol namespace required",
            })
            authority.write_json_atomic(failure, MANIFEST)
            raise


if __name__ == "__main__":
    main()
