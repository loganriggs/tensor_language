#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_frozen_authority_and_instrument pred_b_fresh_regular_and_irregular_decode pred_c_equal_norm_random_null
"""Prospective lexical OOD test of the frozen native embedding-number decoder."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_decoder_fresh as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_embedding_decoder_v1 as discovery


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FRESH_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FRESH_V1_BINDING.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_embedding_decoder_fresh_v1_result.json"
NULLS = 64
SEED = 20260916
PRICE = {"physical_model_forwards": 0, "fresh_embedding_rows": 32,
         "training_embedding_rows": 96, "equal_norm_random_axes": NULLS,
         "fits": 0, "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {"pred_a_frozen_authority_and_instrument": None,
                       "pred_b_fresh_regular_and_irregular_decode": None,
                       "pred_c_equal_norm_random_null": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"decoder": DECODER, "authority": Path(authority.__file__),
             "preregistration": PREREG}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["nulls"] != NULLS or binding["seed"] != SEED \
            or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder = json.loads(DECODER.read_text())
    if decoder["terminal"] != "embedding_number_decoder_frozen":
        raise ValueError("decoder status changed")
    return binding, decoder


def plan():
    binding, decoder = load_bound(); rows = authority.build_rows()
    return {"schema": "subject_number_embedding_decoder_fresh_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "forms": 2 * len(rows),
            "strata": {name: sum(row["stratum"] == name for row in rows)
                       for name in ("irregular", "regular")},
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
            "nulls": NULLS, "price": PRICE,
            "authority_sha256": authority.canonical(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    signed = scores * labels
    return {"count": int(len(scores)), "accuracy": float(np.mean(signed > 0)),
            "minimum_signed_margin": float(signed.min()),
            "mean_signed_margin": float(signed.mean()),
            "rms_margin": float(np.sqrt(np.mean(scores ** 2)))}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, decoder = load_bound(); rows = authority.build_rows()
    torch, F, facade = discovery.tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    device = next(model.parameters()).device
    fresh_ids = torch.tensor([[row["singular_token_id"], row["plural_token_id"]]
                              for row in rows], dtype=torch.long, device=device)
    training_records = discovery.all_pairs()
    encoding = discovery.cardinality.old_task14.ENCODING
    training_ids = torch.tensor([[encoding.encode(" " + row["singular"])[0],
                                  encoding.encode(" " + row["plural"])[0]]
                                 for row in training_records], dtype=torch.long, device=device)
    with torch.no_grad():
        fresh = F.rms_norm(model.transformer.wte(fresh_ids),
                           (model.config.n_embd,)).float().cpu().numpy()
        training = F.rms_norm(model.transformer.wte(training_ids),
                              (model.config.n_embd,)).float().cpu().numpy()
    axis = np.asarray(decoder["frozen_decoder"]["axis"], dtype=np.float32)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    scores = fresh @ axis - threshold
    labels = np.tile(np.asarray([-1., 1.]), (len(rows), 1))
    overall = metrics(scores.ravel(), labels.ravel())
    strata = {}
    for name in ("irregular", "regular"):
        index = np.asarray([row["stratum"] == name for row in rows])
        strata[name] = metrics(scores[index].ravel(), labels[index].ravel())
    pair_delta = scores[:, 1] - scores[:, 0]
    pair_order_accuracy = float(np.mean(pair_delta > 0))
    rng = np.random.default_rng(SEED); null_accuracies = []
    midpoint = training.mean(axis=1)
    for _ in range(NULLS):
        random_axis = rng.standard_normal(axis.shape)
        random_axis /= np.linalg.norm(random_axis)
        random_threshold = float(np.mean(midpoint @ random_axis))
        random_scores = fresh @ random_axis - random_threshold
        null_accuracies.append(metrics(random_scores.ravel(), labels.ravel())["accuracy"])
    null_median = float(np.median(null_accuracies)); null_advantage = overall["accuracy"] - null_median
    finite = np.isfinite(np.asarray([*scores.ravel(), *null_accuracies])).all()
    instrument = (len(rows) == 16 and fresh.size // fresh.shape[-1] == PRICE["fresh_embedding_rows"]
                  and training.size // training.shape[-1] == PRICE["training_embedding_rows"]
                  and len(null_accuracies) == NULLS and finite)
    transfer = (overall["accuracy"] >= .90 and pair_order_accuracy >= .90
                and all(value["accuracy"] >= .80 and value["mean_signed_margin"] > 0
                        for value in strata.values()))
    null_pass = null_advantage >= .20
    predictions = {"pred_a_frozen_authority_and_instrument": bool(instrument),
                   "pred_b_fresh_regular_and_irregular_decode": bool(instrument and transfer),
                   "pred_c_equal_norm_random_null": bool(instrument and transfer and null_pass)}
    terminal = "embedding_number_decoder_fresh_held" if all(predictions.values()) else \
        "embedding_number_decoder_fresh_null" if instrument else "invalid"
    result = {"schema": "subject_number_embedding_decoder_fresh_v1_result",
              "terminal": terminal, "predictions": predictions,
              "overall": overall, "pair_order_accuracy": pair_order_accuracy,
              "minimum_pair_delta": float(pair_delta.min()), "strata": strata,
              "random_axis_median_accuracy": null_median,
              "random_axis_accuracy_advantage": null_advantage,
              "random_axis_accuracies": null_accuracies,
              "records": [{"row_id": row["row_id"], "singular": row["singular"],
                           "plural": row["plural"], "stratum": row["stratum"],
                           "singular_score": float(scores[i, 0]),
                           "plural_score": float(scores[i, 1]),
                           "pair_delta": float(pair_delta[i])}
                          for i, row in enumerate(rows)],
              "instrument": {"fresh_embedding_rows": int(fresh.size // fresh.shape[-1]),
                             "training_embedding_rows": int(training.size // training.shape[-1]),
                             "equal_norm_random_axes": len(null_accuracies), "model_forwards": 0},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "prompt_activations": False, "downstream_states": False,
                                 "fresh_embedding_scores": True},
              "price": PRICE, "authority_sha256": planned["authority_sha256"],
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Prospective disjoint-vocabulary regular/irregular evaluation of the frozen checkpoint-embedding grammatical-number decoder; no prompt behavior or downstream causal claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "overall",
                                                    "pair_order_accuracy", "minimum_pair_delta",
                                                    "strata", "random_axis_median_accuracy",
                                                    "random_axis_accuracy_advantage", "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
