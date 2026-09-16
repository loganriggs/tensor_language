#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_scalar_instrument pred_b_label_free_scalar_cross_validation pred_c_permutation_specificity
"""Discover a donor-free embedding-score generator for the L11H3 mediation scalar."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_removal_fresh as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
MEDIATION_DISCOVERY = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_DISCOVERY_V1_RESULT.json"
MEDIATION_FRESH = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_FRESH_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128
FORMS = ("score", "score_abs", "score_signed_quadratic", "score_distance")
NULLS, SEED = 64, 20260920
BARS = {"maximum_cross_validation_relative_l2": .25,
        "maximum_instrument_error": 1e-4,
        "minimum_cross_validation_cosine": .95,
        "minimum_cross_validation_sign": .9,
        "minimum_permutation_advantage": .2,
        "selection_simplicity_slack": .01}
PRICE = {"partial_forwards": 2, "sequences": 128, "fits": 1100,
         "behavior_logits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_scalar_instrument": None,
                       "pred_b_label_free_scalar_cross_validation": None,
                       "pred_c_permutation_specificity": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "decoder": DECODER,
             "rank1": RANK1, "mediation_discovery": MEDIATION_DISCOVERY,
             "mediation_fresh": MEDIATION_FRESH, "preregistration": PREREG}
    if binding["files"] != {name: sha(path) for name, path in paths.items()} \
            or binding["bars"] != BARS or binding["candidate_forms"] != list(FORMS) \
            or binding["nulls"] != NULLS or binding["null_seed"] != SEED \
            or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, discovery, fresh = (json.loads(path.read_text())
                                       for path in (DECODER, RANK1,
                                                    MEDIATION_DISCOVERY, MEDIATION_FRESH))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or discovery["terminal"] != "embedding_to_l11h3_mediation_null" \
            or fresh["terminal"] != "embedding_to_l11h3_small_mediation_fresh_held":
        raise ValueError("parent status changed")
    rows = authority.build_rows()
    return binding, decoder, rank, discovery, fresh, rows


def plan():
    _, decoder, rank, discovery, fresh, rows = load_bound()
    return {"schema": "subject_number_embedding_to_l11h3_scalar_discovery_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "opened_rows": len(rows), "forms": list(FORMS), "nulls": NULLS,
            "features": ["frozen_embedding_score", "absolute_score",
                         "signed_score_square", "same_norm_removal_distance"],
            "labels_or_templates_as_features": False,
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
            "rank": rank["rank"], "discovery_terminal": discovery["terminal"],
            "fresh_terminal": fresh["terminal"], "bars": BARS, "price": PRICE,
            "binding_sha256": sha(BINDING)}


def design(name, score, distance):
    one = np.ones_like(score)
    if name == "score": return np.stack([one, score], axis=1)
    if name == "score_abs": return np.stack([one, score, np.abs(score)], axis=1)
    if name == "score_signed_quadratic":
        return np.stack([one, score, np.abs(score), score * np.abs(score)], axis=1)
    if name == "score_distance":
        return np.stack([one, score, np.abs(score), distance, score * distance], axis=1)
    raise ValueError(name)


def fit_predict(matrix, target, train, test, counts):
    counts["fits"] += 1
    beta = np.linalg.lstsq(matrix[train], target[train], rcond=None)[0]
    return matrix[test] @ beta


def metric(predicted, actual):
    pn, an = np.linalg.norm(predicted), np.linalg.norm(actual)
    return {"cosine": float(predicted @ actual / max(pn * an, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(predicted - actual) / max(an, 1e-30)),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "predicted_to_actual_norm_ratio": float(pn / max(an, 1e-30))}


def cross_validate(matrix, target, rows, counts):
    all_indices = np.arange(len(rows)); pair_prediction = np.zeros_like(target)
    for pair in range(16):
        test = np.asarray([i for i, row in enumerate(rows) if row["pair_index"] == pair])
        train = np.setdiff1d(all_indices, test)
        pair_prediction[test] = fit_predict(matrix, target, train, test, counts)
    template_prediction = np.zeros_like(target)
    for template in authority.TEMPLATES:
        test = np.asarray([i for i, row in enumerate(rows)
                           if row["template_id"] == template[0]])
        train = np.setdiff1d(all_indices, test)
        template_prediction[test] = fit_predict(matrix, target, train, test, counts)
    return {"leave_pair_out": metric(pair_prediction, target),
            "leave_template_out": metric(template_prediction, target),
            "pair_predictions": pair_prediction,
            "template_predictions": template_prediction}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(600)
    binding, decoder, rank, discovery, fresh, rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    positions = torch.tensor([row["subject_position"] for row in rows],
                             dtype=torch.long, device=device)
    batch = torch.arange(len(rows), device=device)
    with torch.no_grad():
        base_input = F.rms_norm(model.transformer.wte(tokens),
                                (model.config.n_embd,)).float()
    subject = base_input[batch, positions].double()
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"],
                                dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    unit_decoder = decoder_axis / decoder_axis.norm()
    projection = subject @ unit_decoder
    target_projection = threshold / float(decoder_axis.norm())
    orthogonal = subject - projection[:, None] * unit_decoder
    radius_squared = subject.square().sum(1)
    scale = torch.sqrt((radius_squared - target_projection ** 2)
                       / orthogonal.square().sum(1))
    removed_subject = target_projection * unit_decoder + scale[:, None] * orthogonal
    removed_input = base_input.clone(); removed_input[batch, positions] = removed_subject.float()
    score = (subject @ decoder_axis - threshold).cpu().numpy()
    distance = (removed_subject - subject).norm(dim=1).cpu().numpy()
    writer_axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    writer_axis /= writer_axis.norm()
    head_slice = slice(HEAD * HEAD_WIDTH, (HEAD + 1) * HEAD_WIDTH)
    counts = {"partial_forwards": 0, "sequences": 0, "fits": 0,
              "behavior_logits": 0, "backwards": 0, "parameter_updates": 0}

    def capture(initial):
        counts["partial_forwards"] += 1; counts["sequences"] += len(rows)
        found = {}
        with torch.no_grad():
            x = initial; x0 = initial; first_value = None
            for layer, block in enumerate(model.transformer.h):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                handle = None
                if layer == LAYER:
                    def hook(_module, arguments):
                        found["head"] = arguments[0][batch, positions, head_slice].detach().clone()
                    handle = block.attn.c_proj.register_forward_pre_hook(hook)
                try:
                    attention, first_value = block.attn(F.rms_norm(x, (x.shape[-1],)),
                                                         first_value)
                finally:
                    if handle is not None: handle.remove()
                if layer == LAYER: break
                x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
        return found["head"]

    base_head = capture(base_input); removed_head = capture(removed_input)
    projection_weight = model.transformer.h[LAYER].attn.c_proj.weight[:, head_slice]
    full_delta = F.linear(base_head - removed_head, projection_weight)
    alpha = (full_delta @ writer_axis).double().cpu().numpy()
    mean_absolute_alpha = float(np.mean(np.abs(alpha)))
    expected_mean = float(discovery["mean_rescue_write_norms"]["axis"])
    instrument_error = abs(mean_absolute_alpha - expected_mean)
    reports = {}; full_coefficients = {}
    for name in FORMS:
        matrix = design(name, score, distance)
        cv = cross_validate(matrix, alpha, rows, counts)
        counts["fits"] += 1
        beta = np.linalg.lstsq(matrix, alpha, rcond=None)[0]
        full_coefficients[name] = beta.tolist()
        reports[name] = {"leave_pair_out": cv["leave_pair_out"],
                         "leave_template_out": cv["leave_template_out"],
                         "worst_relative_l2": max(cv["leave_pair_out"]["relative_l2_error"],
                                                  cv["leave_template_out"]["relative_l2_error"]),
                         "full_fit": metric(matrix @ beta, alpha)}
    best = min(report["worst_relative_l2"] for report in reports.values())
    selected = next(name for name in FORMS
                    if reports[name]["worst_relative_l2"] <= best + BARS["selection_simplicity_slack"])
    selected_matrix = design(selected, score, distance)
    rng = np.random.default_rng(SEED); permutation_errors = []
    blocks = np.asarray([[i for i, row in enumerate(rows) if row["pair_index"] == pair]
                         for pair in range(16)])
    all_indices = np.arange(len(rows))
    for _ in range(NULLS):
        order = rng.permutation(16)
        permuted = alpha.copy()
        for destination, source in enumerate(order):
            permuted[blocks[destination]] = alpha[blocks[source]]
        prediction = np.zeros_like(alpha)
        for pair in range(16):
            test = blocks[pair]; train = np.setdiff1d(all_indices, test)
            prediction[test] = fit_predict(selected_matrix, permuted, train, test, counts)
        permutation_errors.append(metric(prediction, permuted)["relative_l2_error"])
    permutation_median = float(np.median(permutation_errors))
    selected_report = reports[selected]
    permutation_advantage = permutation_median - selected_report["worst_relative_l2"]
    finite = bool(np.isfinite(np.asarray([*alpha, *score, *distance,
                                          *permutation_errors])).all())
    instrument = bool(finite and instrument_error <= BARS["maximum_instrument_error"]
                      and counts == PRICE
                      and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    cv_pass = lambda report: (report["relative_l2_error"]
                              <= BARS["maximum_cross_validation_relative_l2"]
                              and report["cosine"] >= BARS["minimum_cross_validation_cosine"]
                              and report["sign_agreement"] >= BARS["minimum_cross_validation_sign"])
    pred_b = bool(instrument and cv_pass(selected_report["leave_pair_out"])
                  and cv_pass(selected_report["leave_template_out"]))
    pred_c = bool(pred_b and permutation_advantage >= BARS["minimum_permutation_advantage"])
    predictions = dict(zip(PREDICTION_REGISTRY, (instrument, pred_b, pred_c)))
    terminal = ("embedding_to_l11h3_scalar_candidate" if pred_c else
                "invalid" if not instrument else "embedding_to_l11h3_scalar_null")
    result = {"schema": "subject_number_embedding_to_l11h3_scalar_discovery_v1_result",
              "terminal": terminal, "predictions": predictions,
              "instrument": {"finite": finite, "mean_absolute_alpha": mean_absolute_alpha,
                             "expected_mean_absolute_alpha": expected_mean,
                             "mean_absolute_alpha_error": instrument_error,
                             "counts": counts},
              "candidates": reports, "selected_form": selected,
              "selected_feature_order": {"score": ["1", "s"],
                  "score_abs": ["1", "s", "abs_s"],
                  "score_signed_quadratic": ["1", "s", "abs_s", "s_abs_s"],
                  "score_distance": ["1", "s", "abs_s", "distance", "s_distance"]}[selected],
              "selected_coefficients": full_coefficients[selected],
              "permutation": {"count": NULLS, "seed": SEED,
                              "median_pair_cv_relative_l2": permutation_median,
                              "selected_worst_relative_l2": selected_report["worst_relative_l2"],
                              "advantage": permutation_advantage,
                              "errors": permutation_errors},
              "records": [{"row_id": row["row_id"], "pair_index": row["pair_index"],
                           "template": row["template_id"], "score": float(score[i]),
                           "distance": float(distance[i]), "alpha": float(alpha[i])}
                          for i, row in enumerate(rows)],
              "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows),
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "wall_seconds": time.perf_counter() - started,
              "scope": ("Opened-panel label-free polynomial discovery for the native L11H3 "
                        "rank-one response to embedding-number coordinate removal; no behavior "
                        "logits or fresh OOD claim.")}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("terminal", "predictions", "instrument", "candidates",
                       "selected_form", "selected_feature_order", "selected_coefficients",
                       "permutation")}, indent=2, sort_keys=True))
    assert instrument


if __name__ == "__main__":
    main()
