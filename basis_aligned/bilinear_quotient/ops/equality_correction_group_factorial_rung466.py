#!/usr/bin/env python3
"""RUNG466 -- five-site equality correction group removal factorial."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_correction_paired_necessity_rung465 as parent


PREREG = POLY / "EQUALITY_CORRECTION_GROUP_FACTORIAL_RUNG466_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_correction_paired_necessity_rung465_results.json"
PARENT_SOURCE = ROOT / "ops/equality_correction_paired_necessity_rung465.py"
OUT = ROOT / "equality_correction_group_factorial_rung466_results.json"
BUNDLE = ROOT / "equality_correction_group_factorial_rung466_sufficient_statistics.pt"
SOURCES = parent.SOURCES
ALL_SOURCES = parent.ALL_SOURCES
CELLS = parent.CELLS
CONTEXT_CELLS = parent.CONTEXT_CELLS
DOCUMENTS = parent.DOCUMENTS
BATCH = parent.BATCH
SITES = ("m8", "m9", "m12", "a14", "m17")
TASK_SITES = ("m8", "m9", "m12")
SUPPRESSOR_SITES = ("a14", "m17")
SUBSETS = tuple(range(1 << len(SITES)))
TASK_MASK = sum(1 << SITES.index(site) for site in TASK_SITES)
SUPPRESSOR_MASK = sum(1 << SITES.index(site) for site in SUPPRESSOR_SITES)
ALL_MASK = (1 << len(SITES)) - 1
EXPECTED_FORWARDS = (DOCUMENTS // BATCH) * (
    2 + len(ALL_SOURCES) + len(SOURCES) * len(SUBSETS) + len(SOURCES)
)
HASHES = {
    PREREG: "98b08bffd7567f5ccca36e1b8a9fad48ca377d8dc43df9b3741da5ccb3132c3c",
    PARENT_RESULT: "1502b188cdefbc1be1a7b0c96523b78acfb681a06b438582c5149a024eabecca",
    PARENT_SOURCE: "367474e4aa4f3f859421d0db7504f724de3054926761acdbd06429b82932d5d4",
    parent.parent.path_parent.ROW_RECEIPT:
        "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    parent.parent.path_parent.ROWS:
        "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_sites(mask):
    return tuple(site for i, site in enumerate(SITES) if mask & (1 << i))


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 465 or result.get("pred_a_instrument") is not True \
            or result.get("pred_b_mlp17_shared_role") is not True \
            or result.get("pred_c_mlp17_context_correction") is not False \
            or result.get("pred_d_shared_multisite_program") is not True \
            or result.get("pred_e_not_raw_amplitude") is not True \
            or result.get("strong_null") is not False:
        raise RuntimeError("rung465 registered verdict identity changed")
    payload, masks, scale, metadata, _ = parent.validate_inputs()
    metadata = {
        **metadata,
        "rung465_result_sha256": sha256(PARENT_RESULT),
        "rung465_source_sha256": sha256(PARENT_SOURCE),
        "factorial_sites": list(SITES), "task_sites": list(TASK_SITES),
        "suppressor_sites": list(SUPPRESSOR_SITES),
        "subset_masks": list(SUBSETS), "document_halves": [[0, 96], [96, 192]],
    }
    return payload, masks, scale, metadata, result


def _effect(base, other, counts, start, stop):
    return parent.parent.path_parent.parent.effect_report(base, other, counts, start, stop)


def _metrics(left, right):
    left = torch.tensor(left, dtype=torch.float64)
    right = torch.tensor(right, dtype=torch.float64)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    cosine = float(torch.dot(left, right) / max(left_norm * right_norm, 1e-30))
    projection = float(torch.dot(right, left) / max(float(torch.dot(left, left)), 1e-30))
    return {
        "cosine": cosine, "left_norm": left_norm, "right_norm": right_norm,
        "larger_over_smaller": max(left_norm, right_norm)
        / max(min(left_norm, right_norm), 1e-30),
        "right_projection_on_left": projection,
    }


def _sign_pattern(vector):
    return bool(vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0)


def _all_negative(vector):
    return all(value < 0 for value in vector)


def _window(base_loss, losses, direct, counts, start, stop):
    reports = {
        source: {
            mask: _effect(base_loss, losses[SOURCES.index(source), mask], counts, start, stop)
            for mask in SUBSETS
        }
        for source in SOURCES
    }
    direct_reports = {
        source: _effect(base_loss, direct[SOURCES.index(source)], counts, start, stop)
        for source in SOURCES
    }
    values = {source: {} for source in SOURCES}
    dividends = {source: {} for source in SOURCES}
    for source in SOURCES:
        full = reports[source][0]
        for mask in SUBSETS:
            values[source][mask] = {
                cell: full[cell]["effect_nat"] - reports[source][mask][cell]["effect_nat"]
                for cell in CELLS
            }
        for mask in SUBSETS[1:]:
            dividend = {}
            submask = mask
            contained = []
            while True:
                contained.append(submask)
                if submask == 0:
                    break
                submask = (submask - 1) & mask
            for cell in CELLS:
                dividend[cell] = sum(
                    ((-1) ** ((mask.bit_count() - child.bit_count())))
                    * values[source][child][cell] for child in contained
                )
            dividends[source][mask] = dividend
    vectors = {
        source: {
            mask: [values[source][mask][cell] for cell in CONTEXT_CELLS]
            for mask in SUBSETS
        }
        for source in SOURCES
    }
    correction = {
        source: [reports[source][0][cell]["effect_nat"]
                 - direct_reports[source][cell]["effect_nat"] for cell in CONTEXT_CELLS]
        for source in SOURCES
    }
    interaction = {
        source: [vectors[source][ALL_MASK][i] - vectors[source][TASK_MASK][i]
                 - vectors[source][SUPPRESSOR_MASK][i] for i in range(len(CONTEXT_CELLS))]
        for source in SOURCES
    }
    return {
        "reports": reports, "direct_reports": direct_reports,
        "subset_values": values, "mobius_dividends": dividends,
        "subset_vectors": vectors, "total_correction_vectors": correction,
        "task_source_comparison": _metrics(vectors["N"][TASK_MASK], vectors["H"][TASK_MASK]),
        "suppressor_source_comparison": _metrics(
            vectors["N"][SUPPRESSOR_MASK], vectors["H"][SUPPRESSOR_MASK]
        ),
        "all_source_comparison": _metrics(vectors["N"][ALL_MASK], vectors["H"][ALL_MASK]),
        "task_correction_alignment": {
            source: _metrics(correction[source], vectors[source][TASK_MASK]) for source in SOURCES
        },
        "suppressor_correction_alignment": {
            source: _metrics(correction[source], vectors[source][SUPPRESSOR_MASK])
            for source in SOURCES
        },
        "all_correction_alignment": {
            source: _metrics(correction[source], vectors[source][ALL_MASK]) for source in SOURCES
        },
        "cross_group_interaction_vectors": interaction,
        "interaction_source_comparison": _metrics(interaction["N"], interaction["H"]),
    }


def analyze(base_loss, losses, direct, counts):
    pooled = _window(base_loss, losses, direct, counts, 0, DOCUMENTS)
    halves = [_window(base_loss, losses, direct, counts, start, start + 96)
              for start in (0, 96)]
    pred_b = bool(
        all(_sign_pattern(pooled["subset_vectors"][source][TASK_MASK])
            and pooled["task_correction_alignment"][source]["right_norm"] >= .04
            and pooled["task_correction_alignment"][source]["cosine"] >= .90
            for source in SOURCES)
        and all(_sign_pattern(half["subset_vectors"][source][TASK_MASK])
                and half["task_correction_alignment"][source]["cosine"] > 0
                for half in halves for source in SOURCES)
        and pooled["task_source_comparison"]["cosine"] >= .90
        and pooled["task_source_comparison"]["larger_over_smaller"] <= 2.0
        and all(half["task_source_comparison"]["cosine"] > 0 for half in halves)
    )
    pred_c = bool(
        all(_all_negative(pooled["subset_vectors"][source][SUPPRESSOR_MASK])
            and pooled["suppressor_correction_alignment"][source]["cosine"] < .70
            for source in SOURCES)
        and all(_all_negative(half["subset_vectors"][source][SUPPRESSOR_MASK])
                for half in halves for source in SOURCES)
        and pooled["suppressor_source_comparison"]["cosine"] >= .90
        and all(half["suppressor_source_comparison"]["cosine"] > 0 for half in halves)
    )
    pred_d = bool(
        all(torch.linalg.vector_norm(torch.tensor(
            pooled["cross_group_interaction_vectors"][source]
        )).item() >= .01 for source in SOURCES)
        and pooled["interaction_source_comparison"]["cosine"] >= .80
        and all(half["interaction_source_comparison"]["cosine"] > 0 for half in halves)
    )
    pred_e = bool(
        all(pooled["all_correction_alignment"][source]["cosine"] >= .85
            and .50 <= pooled["all_correction_alignment"][source][
                "right_projection_on_left"
            ] <= 1.50 for source in SOURCES)
        and pooled["all_source_comparison"]["cosine"] >= .85
        and all(half["all_correction_alignment"][source]["cosine"] > 0
                for half in halves for source in SOURCES)
    )
    full_positive = all(
        pooled["reports"][source][0]["all_positive"]["effect_nat"] > 0
        for source in SOURCES
    )
    task_norms = [pooled["task_correction_alignment"][source]["right_norm"]
                  for source in SOURCES]
    all_inert = all(
        abs(pooled["subset_values"][source][ALL_MASK][cell]) < 1e-6
        for source in SOURCES for cell in CONTEXT_CELLS
    )
    strong_science_null = bool(
        not full_positive or (all(value < .01 for value in task_norms))
        or pooled["task_source_comparison"]["cosine"] <= 0 or all_inert
    )
    return {
        "pooled": pooled, "halves": halves,
        "pred_b_task_group_context": pred_b,
        "pred_c_broad_suppressor_role": pred_c,
        "pred_d_cross_group_interaction": pred_d,
        "pred_e_five_site_extraction": pred_e,
        "strong_science_null": strong_science_null,
    }


@torch.no_grad()
def collect(model, payload, masks, scale):
    rows = payload["rows"]
    base_loss = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    losses = torch.zeros(len(SOURCES), len(SUBSETS), DOCUMENTS, len(CELLS),
                         dtype=torch.float64)
    direct = torch.zeros(len(SOURCES), DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = parent.parent.run_forward(model, tokens, arm="native")
        parent.parent.path_parent.parent._record_audit(
            audit_totals, "group:native", audit,
            analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = parent.parent.run_forward(model, tokens, arm="replay")
        parent.parent.path_parent.parent._record_audit(
            audit_totals, "group:replay", audit,
            analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        del native, replay_logits, difference
        captures = {}
        for source in ALL_SOURCES:
            logits, source_captures, audit, error = parent.parent.run_forward(
                model, tokens, arm=parent.parent.SOURCE_ARMS[source], scale=scale,
                capture_keys=parent.CANDIDATES,
            )
            parent.parent.path_parent.parent._record_audit(
                audit_totals, f"group:capture:{source}", audit,
                analytical=True, captures=len(parent.CANDIDATES), patches=0,
            )
            reconstruction = max(reconstruction, error)
            sums, observed = parent.parent.path_parent.parent._ce_sums(
                logits, batch_rows, masks, start,
            )
            if source == "0":
                base_loss[start:start + BATCH] = sums
                counts[start:start + BATCH] = observed
            elif not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("group capture support changed")
            captures[source] = source_captures
            del logits
        for si, source in enumerate(SOURCES):
            for mask in SUBSETS:
                patches = {site: captures["0"][site] for site in subset_sites(mask)}
                logits, _, audit, error = parent.parent.run_forward(
                    model, tokens, arm=parent.parent.SOURCE_ARMS[source], scale=scale,
                    patch_writes=patches,
                )
                parent.parent.path_parent.parent._record_audit(
                    audit_totals, f"group:{source}:{mask:02d}", audit,
                    analytical=True, captures=0, patches=len(patches),
                )
                reconstruction = max(reconstruction, error)
                sums, observed = parent.parent.path_parent.parent._ce_sums(
                    logits, batch_rows, masks, start,
                )
                if not torch.equal(observed, counts[start:start + BATCH]):
                    raise RuntimeError("group subset support changed")
                losses[si, mask, start:start + BATCH] = sums
                del logits
            logits, _, audit, error = parent.parent.run_forward(
                model, tokens, arm=parent.parent.SOURCE_ARMS[source], scale=scale,
                patch_writes=captures["0"],
            )
            parent.parent.path_parent.parent._record_audit(
                audit_totals, f"group:direct:{source}", audit,
                analytical=True, captures=0, patches=len(parent.CANDIDATES),
            )
            reconstruction = max(reconstruction, error)
            sums, observed = parent.parent.path_parent.parent._ce_sums(
                logits, batch_rows, masks, start,
            )
            if not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("group direct support changed")
            direct[si, start:start + BATCH] = sums
            del logits
        del captures
    return base_loss, losses, direct, counts, audit_totals, replay, reconstruction


def _parent_errors(analysis, parent_result):
    parent_pooled = parent_result["analysis"]["pooled"]
    singleton_max = 0.0
    full_max = 0.0
    direct_max = 0.0
    for source in SOURCES:
        for cell in CELLS:
            full_max = max(full_max, abs(
                analysis["pooled"]["reports"][source][0][cell]["effect_nat"]
                - parent_pooled["full_reports"][source][cell]["effect_nat"]
            ))
            direct_max = max(direct_max, abs(
                analysis["pooled"]["direct_reports"][source][cell]["effect_nat"]
                - parent_pooled["direct_reports"][source][cell]["effect_nat"]
            ))
        for site in SITES:
            mask = 1 << SITES.index(site)
            for cell in CELLS:
                singleton_max = max(singleton_max, abs(
                    analysis["pooled"]["subset_values"][source][mask][cell]
                    - parent_pooled["necessity"][source][site][cell]
                ))
    return full_max, direct_max, singleton_max


def main():
    started = time.time()
    payload, masks, scale, metadata, parent_result = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 466, "model_loaded": False,
            "new_group_factorial_outcomes_opened": False, "sealed_opened": False,
            "sites": SITES, "task_sites": TASK_SITES,
            "suppressor_sites": SUPPRESSOR_SITES, "subset_count": len(SUBSETS),
            "expected_forwards": EXPECTED_FORWARDS, "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung466 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    base_loss, losses, direct, counts, audit, replay, reconstruction = collect(
        model, payload, masks, scale,
    )
    analysis = analyze(base_loss, losses, direct, counts)
    full_error, direct_error, singleton_error = _parent_errors(analysis, parent_result)
    forwards = sum(row["forwards"] for row in audit.values())
    empty_effect = max(abs(
        analysis["pooled"]["subset_values"][source][0][cell]
    ) for source in SOURCES for cell in CELLS)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and empty_effect == 0.0 and full_error <= 1e-10 and direct_error <= 1e-10
        and singleton_error <= 1e-10
        and forwards == EXPECTED_FORWARDS
    )
    strong_null = bool(not pred_a or analysis["strong_science_null"])
    bundle = {
        "schema": "equality_correction_group_factorial_rung466_sufficient_statistics_v1",
        "base_loss_sums": base_loss, "subset_loss_sums": losses,
        "direct_only_loss_sums": direct, "counts": counts,
        "source_order": SOURCES, "site_order": SITES, "subset_masks": SUBSETS,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 466,
        "claim_level": "already_open_code_five_site_causal_group_factorial",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "frozen_sites": SITES, "frozen_task_sites": TASK_SITES,
        "frozen_suppressor_sites": SUPPRESSOR_SITES,
        "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "parent_full_effect_max_abs_error_nat": full_error,
        "parent_direct_effect_max_abs_error_nat": direct_error,
        "parent_singleton_max_abs_error_nat": singleton_error,
        "empty_subset_max_abs_effect_nat": empty_effect,
        "audit_totals": audit,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "outer_forwards": forwards, "factorial_sites": len(SITES),
            "subsets_per_source": len(SUBSETS),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_task_group_context': analysis["pred_b_task_group_context"],
        'pred_c_broad_suppressor_role': analysis["pred_c_broad_suppressor_role"],
        'pred_d_cross_group_interaction': analysis["pred_d_cross_group_interaction"],
        'pred_e_five_site_extraction': analysis["pred_e_five_site_extraction"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "task_conditioned_within_mlp8_mlp9_mlp12_split"
            if pred_a and analysis["pred_b_task_group_context"]
            else "state_level_context_modulation_quotient_not_rank_sweep"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 466,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "headline": {
            "task_source": analysis["pooled"]["task_source_comparison"],
            "task_alignment": analysis["pooled"]["task_correction_alignment"],
            "suppressor_source": analysis["pooled"]["suppressor_source_comparison"],
            "suppressor_alignment": analysis["pooled"]["suppressor_correction_alignment"],
            "interaction_source": analysis["pooled"]["interaction_source_comparison"],
            "interaction_vectors": analysis["pooled"]["cross_group_interaction_vectors"],
            "all_alignment": analysis["pooled"]["all_correction_alignment"],
        },
        "instrument_errors": {"full": full_error, "direct": direct_error,
                              "singletons": singleton_error,
                              "empty": empty_effect},
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
