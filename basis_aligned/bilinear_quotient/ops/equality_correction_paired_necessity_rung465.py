#!/usr/bin/env python3
"""RUNG465 -- paired single-write necessity map of equality correction."""

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
import equality_score_correction_interchange_rung464 as parent


PREREG = POLY / "EQUALITY_CORRECTION_PAIRED_NECESSITY_RUNG465_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_score_correction_interchange_rung464_results.json"
PARENT_SOURCE = ROOT / "ops/equality_score_correction_interchange_rung464.py"
OUT = ROOT / "equality_correction_paired_necessity_rung465_results.json"
BUNDLE = ROOT / "equality_correction_paired_necessity_rung465_sufficient_statistics.pt"
SOURCES = ("N", "H")
ALL_SOURCES = parent.SOURCES
CANDIDATES = parent.CANDIDATES
CELLS = parent.CELLS
CONTEXT_CELLS = parent.CONTEXT_CELLS
DOCUMENTS = parent.DOCUMENTS
BATCH = parent.BATCH
PRIMARY_SITE = "m17"
D_MODEL = 1152
EXPECTED_FORWARDS = (DOCUMENTS // BATCH) * (
    2 + len(ALL_SOURCES) + len(SOURCES) * len(CANDIDATES) + len(SOURCES)
)
HASHES = {
    PREREG: "8b3196660d5e7196a28c059d91989cce7279f8e23fe04f3500b9ed35b194469d",
    PARENT_RESULT: "b5b48c4ac7b7fa035f0e2dd1cba82b64059dbe732e229acd006b43c3d0003b6b",
    PARENT_SOURCE: "480dcfe3177a174a89b0c1a026eb309e620d4cd6a9f5642a051e9e2914c3ccb2",
    parent.path_parent.ROW_RECEIPT:
        "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    parent.path_parent.ROWS:
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


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 464 or not all(result.get(key) is True for key in (
        "pred_a_instrument", "pred_b_common_matched_correction",
        "pred_c_correction_interchange", "pred_d_crossed_complete_circuits",
        "pred_e_correction_not_standalone",
    )) or result.get("strong_null") is not False:
        raise RuntimeError("rung464 full-pass identity changed")
    payload, masks, scale, metadata = parent.validate_inputs()
    metadata = {
        **metadata,
        "rung464_result_sha256": sha256(PARENT_RESULT),
        "rung464_source_sha256": sha256(PARENT_SOURCE),
        "necessity_sources": list(SOURCES), "primary_site": PRIMARY_SITE,
        "candidate_order": list(CANDIDATES),
        "document_halves": [[0, 96], [96, 192]],
    }
    return payload, masks, scale, metadata, result


def _empty_response_stats():
    shape = (len(CANDIDATES), len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("native2", "hybrid2", "cross", "tokens")}


def _accumulate_response(stats, captures, masks, start):
    for ji, site in enumerate(CANDIDATES):
        native = captures["N"][site] - captures["0"][site]
        hybrid = captures["H"][site] - captures["0"][site]
        for ci, cell in enumerate(CELLS):
            selected = masks[cell][start:start + BATCH]
            if not bool(selected.any()):
                continue
            left = native[selected].float()
            right = hybrid[selected].float()
            stats["native2"][ji, ci] += left.square().sum().double().cpu()
            stats["hybrid2"][ji, ci] += right.square().sum().double().cpu()
            stats["cross"][ji, ci] += (left * right).sum().double().cpu()
            stats["tokens"][ji, ci] += int(selected.sum())


def _response_report(stats):
    output = {}
    for ji, site in enumerate(CANDIDATES):
        output[site] = {}
        for ci, cell in enumerate(CELLS):
            left2 = float(stats["native2"][ji, ci])
            right2 = float(stats["hybrid2"][ji, ci])
            cross = float(stats["cross"][ji, ci])
            tokens = max(float(stats["tokens"][ji, ci]), 1.0)
            output[site][cell] = {
                "cosine": cross / max(math.sqrt(left2 * right2), 1e-30),
                "native_raw_coordinate_rms": math.sqrt(
                    left2 / (tokens * D_MODEL)
                ),
                "hybrid_raw_coordinate_rms": math.sqrt(
                    right2 / (tokens * D_MODEL)
                ),
                "tokens": int(stats["tokens"][ji, ci]),
            }
    return output


def _effect(base, other, counts, start, stop):
    return parent.path_parent.parent.effect_report(base, other, counts, start, stop)


def _vector_metrics(left, right):
    left = torch.tensor(left, dtype=torch.float64)
    right = torch.tensor(right, dtype=torch.float64)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    return {
        "cosine": float(torch.dot(left, right) / max(left_norm * right_norm, 1e-30)),
        "left_norm": left_norm, "right_norm": right_norm,
        "larger_over_smaller": max(left_norm, right_norm)
        / max(min(left_norm, right_norm), 1e-30),
    }


def _sign_pattern(vector):
    return bool(vector[0] < 0 and vector[1] > 0 and vector[2] > 0 and vector[3] < 0)


def _rankdata(values):
    values = torch.tensor(values, dtype=torch.float64)
    return torch.argsort(torch.argsort(values)).double()


def _spearman(left, right):
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    return float(torch.dot(left_rank, right_rank) / max(
        float(torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)), 1e-30,
    ))


def _window(full, removed, direct, counts, start, stop):
    base = full[ALL_SOURCES.index("0")]
    reports = {
        source: _effect(base, full[ALL_SOURCES.index(source)], counts, start, stop)
        for source in SOURCES
    }
    direct_reports = {
        source: _effect(base, direct[SOURCES.index(source)], counts, start, stop)
        for source in SOURCES
    }
    removed_reports = {
        source: {
            site: _effect(base, removed[SOURCES.index(source), ji], counts, start, stop)
            for ji, site in enumerate(CANDIDATES)
        }
        for source in SOURCES
    }
    necessity = {
        source: {
            site: {cell: reports[source][cell]["effect_nat"]
                   - removed_reports[source][site][cell]["effect_nat"] for cell in CELLS}
            for site in CANDIDATES
        }
        for source in SOURCES
    }
    total_correction = {
        source: {cell: reports[source][cell]["effect_nat"]
                 - direct_reports[source][cell]["effect_nat"] for cell in CELLS}
        for source in SOURCES
    }
    vectors = {
        source: {site: [necessity[source][site][cell] for cell in CONTEXT_CELLS]
                 for site in CANDIDATES}
        for source in SOURCES
    }
    correction_vectors = {
        source: [total_correction[source][cell] for cell in CONTEXT_CELLS]
        for source in SOURCES
    }
    paired = {site: _vector_metrics(vectors["N"][site], vectors["H"][site])
              for site in CANDIDATES}
    correction_alignment = {
        source: {site: _vector_metrics(correction_vectors[source], vectors[source][site])
                 for site in CANDIDATES}
        for source in SOURCES
    }
    return {
        "full_reports": reports, "direct_reports": direct_reports,
        "removed_reports": removed_reports, "necessity": necessity,
        "necessity_vectors": vectors, "total_correction": total_correction,
        "total_correction_vectors": correction_vectors,
        "paired_source_metrics": paired,
        "correction_alignment": correction_alignment,
        "necessity_norms": {
            source: {site: correction_alignment[source][site]["right_norm"]
                     for site in CANDIDATES} for source in SOURCES
        },
    }


def analyze(full, removed, direct, counts, response):
    pooled = _window(full, removed, direct, counts, 0, DOCUMENTS)
    halves = [_window(full, removed, direct, counts, start, start + 96)
              for start in (0, 96)]
    primary = pooled["paired_source_metrics"][PRIMARY_SITE]
    pred_b = bool(
        primary["left_norm"] >= .02 and primary["right_norm"] >= .02
        and primary["cosine"] >= .80 and primary["larger_over_smaller"] <= 2.0
        and all(half["paired_source_metrics"][PRIMARY_SITE]["cosine"] > 0
                for half in halves)
    )
    pred_c = bool(
        all(_sign_pattern(pooled["necessity_vectors"][source][PRIMARY_SITE])
            for source in SOURCES)
        and all(_sign_pattern(half["necessity_vectors"][source][PRIMARY_SITE])
                for half in halves for source in SOURCES)
        and all(pooled["correction_alignment"][source][PRIMARY_SITE]["cosine"] >= .70
                for source in SOURCES)
        and all(half["correction_alignment"][source][PRIMARY_SITE]["cosine"] > 0
                for half in halves for source in SOURCES)
        and all(
            pooled["necessity_norms"][source][PRIMARY_SITE]
            >= 2 * abs(pooled["necessity"][source][PRIMARY_SITE]["off_target"])
            and pooled["necessity_norms"][source][PRIMARY_SITE]
            - abs(pooled["necessity"][source][PRIMARY_SITE]["off_target"]) >= .01
            for source in SOURCES
        )
    )
    qualified = [site for site in CANDIDATES
                 if pooled["paired_source_metrics"][site]["cosine"] >= .70
                 and all(pooled["necessity_norms"][source][site] >= .01
                         for source in SOURCES)]
    norm_spearman = _spearman(
        [pooled["necessity_norms"]["N"][site] for site in CANDIDATES],
        [pooled["necessity_norms"]["H"][site] for site in CANDIDATES],
    )
    half_spearman = [_spearman(
        [half["necessity_norms"]["N"][site] for site in CANDIDATES],
        [half["necessity_norms"]["H"][site] for site in CANDIDATES],
    ) for half in halves]
    pred_d = bool(
        PRIMARY_SITE in qualified and len(qualified) >= 2
        and norm_spearman >= .60 and all(value > 0 for value in half_spearman)
    )
    amplitude_spearman = {}
    primary_ranks = {}
    for source, amplitude_key in (("N", "native_raw_coordinate_rms"),
                                  ("H", "hybrid_raw_coordinate_rms")):
        norms = [pooled["necessity_norms"][source][site] for site in CANDIDATES]
        amplitudes = [response[site]["all_positive"][amplitude_key] for site in CANDIDATES]
        amplitude_spearman[source] = _spearman(norms, amplitudes)
        descending = sorted(CANDIDATES, key=lambda site: pooled["necessity_norms"][source][site],
                            reverse=True)
        primary_ranks[source] = descending.index(PRIMARY_SITE) + 1
    pred_e = bool(
        any(abs(value) < .95 for value in amplitude_spearman.values())
        and all(rank <= 5 for rank in primary_ranks.values())
    )
    matched_positive = all(
        pooled["full_reports"][source]["all_positive"]["effect_nat"] > 0
        for source in SOURCES
    )
    paired_nonzero = any(
        pooled["paired_source_metrics"][site]["cosine"] > 0
        and all(pooled["necessity_norms"][source][site] >= .005 for source in SOURCES)
        for site in CANDIDATES
    )
    strong_science_null = bool(
        not matched_positive
        or (primary["left_norm"] < .005 and primary["right_norm"] < .005)
        or primary["cosine"] <= 0 or not paired_nonzero
    )
    return {
        "pooled": pooled, "halves": halves,
        "qualified_shared_sites": qualified,
        "necessity_norm_rank_spearman": norm_spearman,
        "half_necessity_norm_rank_spearman": half_spearman,
        "necessity_vs_raw_amplitude_spearman": amplitude_spearman,
        "primary_site_necessity_norm_rank": primary_ranks,
        "raw_write_response": response,
        "pred_b_mlp17_shared_role": pred_b,
        "pred_c_mlp17_context_correction": pred_c,
        "pred_d_shared_multisite_program": pred_d,
        "pred_e_not_raw_amplitude": pred_e,
        "strong_science_null": strong_science_null,
    }


@torch.no_grad()
def collect(model, payload, masks, scale):
    rows = payload["rows"]
    full = torch.zeros(len(ALL_SOURCES), DOCUMENTS, len(CELLS), dtype=torch.float64)
    removed = torch.zeros(len(SOURCES), len(CANDIDATES), DOCUMENTS, len(CELLS),
                          dtype=torch.float64)
    direct = torch.zeros(len(SOURCES), DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    response_stats = _empty_response_stats()
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = parent.run_forward(model, tokens, arm="native")
        parent.path_parent.parent._record_audit(
            audit_totals, "necessity:native", audit,
            analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = parent.run_forward(model, tokens, arm="replay")
        parent.path_parent.parent._record_audit(
            audit_totals, "necessity:replay", audit,
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
        for ai, source in enumerate(ALL_SOURCES):
            logits, source_captures, audit, error = parent.run_forward(
                model, tokens, arm=parent.SOURCE_ARMS[source], scale=scale,
                capture_keys=CANDIDATES,
            )
            parent.path_parent.parent._record_audit(
                audit_totals, f"necessity:capture:{source}", audit,
                analytical=True, captures=len(CANDIDATES), patches=0,
            )
            reconstruction = max(reconstruction, error)
            sums, observed = parent.path_parent.parent._ce_sums(
                logits, batch_rows, masks, start,
            )
            if ai == 0:
                counts[start:start + BATCH] = observed
            elif not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("necessity support changed")
            full[ai, start:start + BATCH] = sums
            captures[source] = source_captures
            del logits
        _accumulate_response(response_stats, captures, masks, start)
        for si, source in enumerate(SOURCES):
            for ji, site in enumerate(CANDIDATES):
                logits, _, audit, error = parent.run_forward(
                    model, tokens, arm=parent.SOURCE_ARMS[source], scale=scale,
                    patch_writes={site: captures["0"][site]},
                )
                parent.path_parent.parent._record_audit(
                    audit_totals, f"necessity:{source}:{site}", audit,
                    analytical=True, captures=0, patches=1,
                )
                reconstruction = max(reconstruction, error)
                sums, observed = parent.path_parent.parent._ce_sums(
                    logits, batch_rows, masks, start,
                )
                if not torch.equal(observed, counts[start:start + BATCH]):
                    raise RuntimeError("removal support changed")
                removed[si, ji, start:start + BATCH] = sums
                del logits
            logits, _, audit, error = parent.run_forward(
                model, tokens, arm=parent.SOURCE_ARMS[source], scale=scale,
                patch_writes=captures["0"],
            )
            parent.path_parent.parent._record_audit(
                audit_totals, f"necessity:direct:{source}", audit,
                analytical=True, captures=0, patches=len(CANDIDATES),
            )
            reconstruction = max(reconstruction, error)
            sums, observed = parent.path_parent.parent._ce_sums(
                logits, batch_rows, masks, start,
            )
            if not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("direct support changed")
            direct[si, start:start + BATCH] = sums
            del logits
        del captures
    return (full, removed, direct, counts, _response_report(response_stats),
            response_stats, audit_totals, replay, reconstruction)


def _parent_effect_error(analysis, parent_result):
    expected = parent_result["analysis"]["pooled"]["reports"]
    maximum = 0.0
    for source in ALL_SOURCES:
        for cell in CELLS:
            maximum = max(maximum, abs(
                analysis["pooled"]["full_reports"].get(source, {}).get(cell, {}).get(
                    "effect_nat", 0.0
                ) - expected[source][source][cell]["effect_nat"]
            ))
    return maximum


def main():
    started = time.time()
    payload, masks, scale, metadata, parent_result = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 465, "model_loaded": False,
            "new_necessity_outcomes_opened": False, "sealed_opened": False,
            "sources": SOURCES, "candidate_order": CANDIDATES,
            "primary_site": PRIMARY_SITE, "expected_forwards": EXPECTED_FORWARDS,
            "input_metadata": metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung465 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    full, removed, direct, counts, response, response_stats, audit, replay, reconstruction = (
        collect(model, payload, masks, scale)
    )
    analysis = analyze(full, removed, direct, counts, response)
    parent_effect_error = _parent_effect_error(analysis, parent_result)
    forwards = sum(row["forwards"] for row in audit.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
        and parent_effect_error <= 1e-10 and forwards == EXPECTED_FORWARDS
    )
    strong_null = bool(not pred_a or analysis["strong_science_null"])
    bundle = {
        "schema": "equality_correction_paired_necessity_rung465_sufficient_statistics_v1",
        "full_loss_sums": full, "removed_loss_sums": removed,
        "direct_only_loss_sums": direct, "counts": counts,
        "raw_write_response_grams": response_stats,
        "source_order": SOURCES, "all_source_order": ALL_SOURCES,
        "candidate_order": CANDIDATES,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "sealed_attention0_opened": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 465,
        "claim_level": "already_open_code_paired_write_necessity_map",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "frozen_pair": parent.path_parent.PAIR_NAME,
        "frozen_primary_site": PRIMARY_SITE,
        "analysis": analysis,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "parent_effect_max_abs_error_nat": parent_effect_error,
        "audit_totals": audit,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "outer_forwards": forwards, "candidate_sites": len(CANDIDATES),
            "source_trajectories": len(SOURCES),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_mlp17_shared_role': analysis["pred_b_mlp17_shared_role"],
        'pred_c_mlp17_context_correction': analysis["pred_c_mlp17_context_correction"],
        'pred_d_shared_multisite_program': analysis["pred_d_shared_multisite_program"],
        'pred_e_not_raw_amplitude': analysis["pred_e_not_raw_amplitude"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "task_conditioned_mlp17_split_then_fixed_site_interaction_factorial"
            if pred_a and analysis["pred_b_mlp17_shared_role"]
            and analysis["pred_c_mlp17_context_correction"] and not strong_null
            else "state_level_correction_boundary_or_fixed_qualified_sites"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 465,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "headline": {
            "primary": analysis["pooled"]["paired_source_metrics"][PRIMARY_SITE],
            "primary_alignment": {
                source: analysis["pooled"]["correction_alignment"][source][PRIMARY_SITE]
                for source in SOURCES
            },
            "qualified_shared_sites": analysis["qualified_shared_sites"],
            "norm_rank_spearman": analysis["necessity_norm_rank_spearman"],
            "amplitude_spearman": analysis["necessity_vs_raw_amplitude_spearman"],
            "primary_rank": analysis["primary_site_necessity_norm_rank"],
        },
        "factor_reconstruction_relative_squared_max": reconstruction,
        "native_replay": replay, "parent_effect_max_abs_error_nat": parent_effect_error,
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
