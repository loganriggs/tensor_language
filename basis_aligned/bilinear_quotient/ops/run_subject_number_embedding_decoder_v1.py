#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_weight_and_authority_instrument pred_b_leave_family_out_number_decode pred_c_equal_norm_null_and_freeze
"""Discover and freeze a token-embedding grammatical-number decoder."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import build_subject_number_two_site_composition_v2_rows as two_site
import circuit_fast_screen_candidate_subject_number_response_weighted_fresh as fresh
import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as cardinality
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
FAMILIES = {"cardinality": tuple(cardinality.NOUN_PAIRS),
            "fresh_response": tuple(fresh.NOUN_PAIRS),
            "two_clause": tuple(two_site.NOUN_PAIRS)}
NULLS = 64
SEED = 20260916
PRICE = {"physical_model_forwards": 0, "embedding_rows": 96,
         "leave_family_out_decoders": 3, "frozen_decoders": 1,
         "equal_norm_random_axes": 192, "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {"pred_a_weight_and_authority_instrument": None,
                       "pred_b_leave_family_out_number_decode": None,
                       "pred_c_equal_norm_null_and_freeze": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "cardinality_authority": Path(cardinality.__file__),
             "fresh_authority": Path(fresh.__file__), "two_clause_builder": Path(two_site.__file__)}
    family_manifest = [{"family": key, "pairs": [list(pair) for pair in value]}
                       for key, value in FAMILIES.items()]
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["families_sha256"] != canonical(family_manifest) \
            or binding["nulls"] != NULLS or binding["seed"] != SEED \
            or binding["price"] != PRICE:
        raise ValueError("binding changed")
    return binding


def all_pairs():
    records = []
    for family, pairs in FAMILIES.items():
        for singular, plural in pairs:
            records.append({"family": family, "singular": singular, "plural": plural})
    return records


def plan():
    binding = load_bound(); records = all_pairs()
    return {"schema": "subject_number_embedding_decoder_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "families": {key: len(value) for key, value in FAMILIES.items()},
            "pairs": len(records), "forms": 2 * len(records), "nulls": NULLS,
            "method": "normalized checkpoint token embeddings; mean plural-minus-singular axis; mean midpoint threshold",
            "price": PRICE, "authority_sha256": canonical(records),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def fit_decoder(singular, plural):
    axis = (plural - singular).mean(0)
    axis = axis / np.linalg.norm(axis)
    threshold = float(np.mean(((singular + plural) * .5) @ axis))
    return axis, threshold


def score(axis, threshold, singular, plural):
    singular_score = singular @ axis - threshold
    plural_score = plural @ axis - threshold
    endpoint_correct = np.concatenate([singular_score < 0, plural_score > 0])
    pair_delta = plural_score - singular_score
    return {"endpoints": int(endpoint_correct.size),
            "endpoint_accuracy": float(np.mean(endpoint_correct)),
            "pair_order_accuracy": float(np.mean(pair_delta > 0)),
            "minimum_signed_endpoint_margin": float(min((-singular_score).min(), plural_score.min())),
            "mean_signed_endpoint_margin": float(np.mean(np.concatenate([-singular_score, plural_score]))),
            "minimum_pair_delta": float(pair_delta.min()),
            "mean_pair_delta": float(pair_delta.mean())}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    records = all_pairs(); encoding = cardinality.old_task14.ENCODING
    token_rows = []; seen = set()
    for record in records:
        ids = []
        for key in ("singular", "plural"):
            encoded = encoding.encode(" " + record[key])
            if len(encoded) != 1 or encoded[0] in seen:
                raise ValueError("forms must be distinct one-token rows")
            ids.append(encoded[0]); seen.add(encoded[0])
        token_rows.append(ids)
    ids = torch.tensor(token_rows, dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        embeddings = model.transformer.wte(ids)
        embeddings = F.rms_norm(embeddings, (embeddings.shape[-1],))
    values = embeddings.float().cpu().numpy()
    singular, plural = values[:, 0], values[:, 1]
    families = np.asarray([record["family"] for record in records])
    rng = np.random.default_rng(SEED)
    folds = {}; held_predictions = []; held_labels = []; null_accuracies = []
    for held_out in FAMILIES:
        train = families != held_out; test = ~train
        axis, threshold = fit_decoder(singular[train], plural[train])
        report = score(axis, threshold, singular[test], plural[test])
        fold_nulls = []
        for _ in range(NULLS):
            random_axis = rng.standard_normal(axis.shape)
            random_axis /= np.linalg.norm(random_axis)
            random_threshold = float(np.mean(((singular[train] + plural[train]) * .5) @ random_axis))
            fold_nulls.append(score(random_axis, random_threshold,
                                    singular[test], plural[test])["endpoint_accuracy"])
        report["random_axis_median_endpoint_accuracy"] = float(np.median(fold_nulls))
        report["random_axis_advantage"] = report["endpoint_accuracy"] - report["random_axis_median_endpoint_accuracy"]
        report["axis_float32_sha256"] = hashlib.sha256(axis.astype(np.float32).tobytes()).hexdigest()
        report["threshold"] = threshold
        folds[held_out] = report; null_accuracies.extend(fold_nulls)
        held_predictions.extend(np.concatenate([singular[test] @ axis - threshold,
                                                plural[test] @ axis - threshold]))
        held_labels.extend([-1] * int(test.sum()) + [1] * int(test.sum()))
    held_predictions = np.asarray(held_predictions); held_labels = np.asarray(held_labels)
    pooled_accuracy = float(np.mean(np.sign(held_predictions) == held_labels))
    pooled_null_median = float(np.median(null_accuracies))
    frozen_axis, frozen_threshold = fit_decoder(singular, plural)
    frozen_report = score(frozen_axis, frozen_threshold, singular, plural)
    instrument = (len(records) == 48 and len(seen) == PRICE["embedding_rows"]
                  and len(folds) == PRICE["leave_family_out_decoders"]
                  and len(null_accuracies) == PRICE["equal_norm_random_axes"]
                  and np.isfinite(frozen_axis).all() and abs(np.linalg.norm(frozen_axis) - 1) <= 1e-6)
    transfer = all(value["endpoint_accuracy"] >= .90 and value["pair_order_accuracy"] >= .90
                   for value in folds.values())
    null_pass = pooled_accuracy - pooled_null_median >= .15
    predictions = {"pred_a_weight_and_authority_instrument": bool(instrument),
                   "pred_b_leave_family_out_number_decode": bool(instrument and transfer),
                   "pred_c_equal_norm_null_and_freeze": bool(instrument and transfer and null_pass)}
    terminal = "embedding_number_decoder_frozen" if all(predictions.values()) else \
        "embedding_number_decoder_null" if instrument else "invalid"
    result = {"schema": "subject_number_embedding_decoder_frozen_v1_artifact",
              "terminal": terminal, "predictions": predictions,
              "leave_family_out": folds, "pooled_endpoint_accuracy": pooled_accuracy,
              "pooled_random_axis_median_endpoint_accuracy": pooled_null_median,
              "pooled_random_axis_advantage": pooled_accuracy - pooled_null_median,
              "frozen_decoder": {"axis": frozen_axis.astype(np.float32).tolist(),
                                 "threshold": frozen_threshold,
                                 "training_metrics": frozen_report,
                                 "axis_float32_sha256": hashlib.sha256(
                                     frozen_axis.astype(np.float32).tobytes()).hexdigest(),
                                 "input": "rms-normalized checkpoint token-embedding row",
                                 "positive_class": "plural", "negative_class": "singular"},
              "instrument": {"pairs": len(records), "embedding_rows": len(seen),
                             "leave_family_out_decoders": len(folds),
                             "equal_norm_random_axes": len(null_accuracies),
                             "model_forwards": 0},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "prompt_activations": False, "downstream_states": False,
                                 "new_text": False, "opened_number_labels": True},
              "price": PRICE, "authority_sha256": planned["authority_sha256"],
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Checkpoint-weight-only linear grammatical-number decoder on normalized subject-token embeddings, selected across three opened lexical families and frozen before any new authority."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions",
                                                    "leave_family_out", "pooled_endpoint_accuracy",
                                                    "pooled_random_axis_median_endpoint_accuracy",
                                                    "pooled_random_axis_advantage", "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
