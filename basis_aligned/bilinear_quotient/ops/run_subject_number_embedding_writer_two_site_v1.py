#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_decoder_writer_instrument pred_b_single_site_writes_live_and_selective pred_c_two_site_additive_composition
"""Join the frozen embedding-number decoder to the frozen rank-one writer."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ROWS = POLY / "SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_ROWS.json"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
DECODER_FRESH = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_embedding_decoder_fresh_v1_result.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_RESULT.json"
LAYER = 11
ARMS = ("base", "zero", "site1", "site2", "both")
BARS = {
    "maximum_anticausal_absolute_effect": 1e-5,
    "maximum_composition_relative_l2": .25,
    "maximum_composition_norm_ratio": 1.2,
    "maximum_oracle_write_error": 1e-7,
    "maximum_unrelated_control_fraction": .75,
    "maximum_zero_replay_logit_error": 1e-5,
    "minimum_composition_cosine": .95,
    "minimum_composition_norm_ratio": .8,
    "minimum_composition_sign": .9,
    "minimum_native_accuracy": .75,
    "minimum_single_site_effect_rms": .01,
    "minimum_single_site_positive_fraction": .75,
}
PRICE = {"forwards": 5, "sequences": 80, "embedding_lookups": 32,
         "fits": 0, "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_exact_decoder_writer_instrument": None,
    "pred_b_single_site_writes_live_and_selective": None,
    "pred_c_two_site_additive_composition": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"rows": ROWS, "decoder": DECODER, "decoder_fresh_result": DECODER_FRESH,
             "rank1": RANK1, "preregistration": PREREG}
    if binding["files"] != {key: sha(path) for key, path in paths.items()}:
        raise ValueError("bound file changed")
    if binding["bars"] != BARS or binding["price"] != PRICE \
            or binding["arms"] != list(ARMS):
        raise ValueError("bound design changed")
    rows, decoder, fresh, rank = (json.loads(path.read_text())
                                  for path in (ROWS, DECODER, DECODER_FRESH, RANK1))
    if rows["row_manifest_sha256"] != binding["row_manifest_sha256"] \
            or rows["sequence_length"] != binding["sequence_length"]:
        raise ValueError("row authority changed")
    positions = sorted({tuple(site["position"] for site in row["sites"])
                        for row in rows["rows"]})
    if positions != [tuple(binding["site_positions_by_row"])]:
        raise ValueError("site positions changed")
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or fresh["terminal"] != "embedding_number_decoder_fresh_held" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or rank["rank"] != 1 or rank["rank_sweep"]:
        raise ValueError("parent status changed")
    coefficients = {key: rank["coefficients"][key]
                    for key in binding["coefficient_keys"]}
    if coefficients != binding["coefficients"]:
        raise ValueError("writer coefficients changed")
    return binding, rows, decoder, fresh, rank


def plan():
    binding, rows, decoder, _, rank = load_bound()
    return {
        "schema": "subject_number_embedding_writer_two_site_v1_plan",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "rows": len(rows["rows"]), "sites": rows["sites"],
        "sequence_length": rows["sequence_length"],
        "site_positions_by_row": binding["site_positions_by_row"],
        "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
        "rank": rank["rank"], "coefficients": binding["coefficients"],
        "bars": BARS, "price": PRICE, "arms": list(ARMS),
        "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"]),
    }


def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    pn, an = np.linalg.norm(predicted), np.linalg.norm(actual)
    return {
        "count": int(actual.size),
        "cosine": float(predicted.ravel() @ actual.ravel() / max(pn * an, 1e-30)),
        "relative_l2_error": float(np.linalg.norm(predicted - actual) / max(an, 1e-30)),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "predicted_to_actual_norm_ratio": float(pn / max(an, 1e-30)),
    }


def composition_passes(report):
    return bool(report["cosine"] >= BARS["minimum_composition_cosine"]
                and report["relative_l2_error"] <= BARS["maximum_composition_relative_l2"]
                and report["sign_agreement"] >= BARS["minimum_composition_sign"]
                and BARS["minimum_composition_norm_ratio"]
                <= report["predicted_to_actual_norm_ratio"]
                <= BARS["maximum_composition_norm_ratio"])


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    binding, frozen, decoder, fresh, rank = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter()
    device = next(model.parameters()).device
    rows = frozen["rows"]
    tokens = torch.tensor([row["token_ids"] for row in rows],
                          dtype=torch.long, device=device)
    positions = torch.tensor([[site["position"] for site in row["sites"]]
                              for row in rows], dtype=torch.long, device=device)
    batch = torch.arange(len(rows), device=device)

    # Candidate selection uses only checkpoint token embeddings and the frozen decoder.
    subject_ids = tokens[batch[:, None], positions]
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"],
                                dtype=torch.float32, device=device)
    decoder_threshold = float(decoder["frozen_decoder"]["threshold"])
    with torch.no_grad():
        subject_embeddings = F.rms_norm(model.transformer.wte(subject_ids),
                                        (model.config.n_embd,))
        decoder_scores = subject_embeddings @ decoder_axis - decoder_threshold
    candidate_plural = decoder_scores > 0
    candidate_directions = [["plural_to_singular" if bool(candidate_plural[i, j])
                             else "singular_to_plural" for j in range(2)]
                            for i in range(len(rows))]
    oracle_directions = [[site["direction"] for site in row["sites"]] for row in rows]
    direction_agreement = np.asarray(candidate_directions) == np.asarray(oracle_directions)
    signed_decoder_margin = np.asarray([
        [float(decoder_scores[i, j]) if row["sites"][j]["native_number"] == "plural"
         else -float(decoder_scores[i, j]) for j in range(2)]
        for i, row in enumerate(rows)
    ])

    axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    candidate_coefficients = torch.where(
        candidate_plural,
        torch.tensor(binding["coefficients"]["plural_to_singular.cardinality_4"], device=device),
        torch.tensor(binding["coefficients"]["singular_to_plural.cardinality_4"], device=device),
    )
    candidate_writes = candidate_coefficients.unsqueeze(-1) * axis
    oracle_coefficients = torch.tensor([
        [binding["coefficients"][f"{site['direction']}.cardinality_4"]
         for site in row["sites"]] for row in rows
    ], dtype=torch.float32, device=device)
    oracle_writes = oracle_coefficients.unsqueeze(-1) * axis
    oracle_write_error = float((candidate_writes - oracle_writes).abs().max())
    finite = bool(torch.isfinite(decoder_scores).all()
                  and torch.isfinite(candidate_writes).all())

    counts = {"forwards": 0, "sequences": 0, "embedding_lookups": int(subject_ids.numel()),
              "fits": 0, "backwards": 0, "updates": 0}

    def run(site1=False, site2=False, zero=False):
        counts["forwards"] += 1; counts["sequences"] += len(rows)

        def attention_dispatch(event):
            write, first_value = event.block.attn(event.state, event.first_value)
            if event.site == LAYER and (site1 or site2 or zero):
                write = write.clone()
                if zero:
                    write[batch, positions[:, 0]] += torch.zeros_like(candidate_writes[:, 0]).to(write.dtype)
                    write[batch, positions[:, 1]] += torch.zeros_like(candidate_writes[:, 1]).to(write.dtype)
                else:
                    if site1:
                        write[batch, positions[:, 0]] += candidate_writes[:, 0].to(write.dtype)
                    if site2:
                        write[batch, positions[:, 1]] += candidate_writes[:, 1].to(write.dtype)
            return write, first_value

        with torch.no_grad():
            return facade.forward_with_dispatch(
                model, tokens, attention_dispatch,
                lambda event: event.block.mlp(event.state),
                require_production=False).float().cpu().numpy()

    logits = {"base": run(), "zero": run(zero=True), "site1": run(site1=True),
              "site2": run(site2=True), "both": run(site1=True, site2=True)}
    zero_error = float(np.max(np.abs(logits["zero"] - logits["base"])))
    number_margins = {arm: np.zeros((len(rows), 2)) for arm in ARMS}
    control_margins = {arm: np.zeros((len(rows), 2)) for arm in ARMS}
    native_correct = np.zeros((len(rows), 2), dtype=bool)
    for i, row in enumerate(rows):
        can_id, will_id = (row["control_token_ids"][key] for key in ("can", "will"))
        for j, site in enumerate(row["sites"]):
            position = site["position"]
            native_correct[i, j] = (logits["base"][i, position, site["native_answer_id"]]
                                    > logits["base"][i, position, site["opposite_answer_id"]])
            for arm in ARMS:
                values = logits[arm][i, position]
                number_margins[arm][i, j] = (values[site["opposite_answer_id"]]
                                              - values[site["native_answer_id"]])
                control_margins[arm][i, j] = values[can_id] - values[will_id]
    number_effects = {arm: number_margins[arm] - number_margins["base"]
                      for arm in ("site1", "site2", "both")}
    control_effects = {arm: control_margins[arm] - control_margins["base"]
                       for arm in ("site1", "site2", "both")}

    capability = {}
    for j in range(2):
        for number in ("singular", "plural"):
            for template in frozen["templates"]:
                ids = [i for i, row in enumerate(rows)
                       if row["sites"][j]["native_number"] == number
                       and row["template_id"] == template]
                capability[f"site{j + 1}|{number}|{template}"] = {
                    "count": len(ids), "accuracy": float(np.mean(native_correct[ids, j]))}

    single_reports, single_pass = {}, True
    for j, arm in enumerate(("site1", "site2")):
        for direction in ("singular_to_plural", "plural_to_singular"):
            ids = [i for i, row in enumerate(rows)
                   if oracle_directions[i][j] == direction]
            target = number_effects[arm][ids, j]
            control = control_effects[arm][ids, j]
            rms = float(np.sqrt(np.mean(target ** 2)))
            report = {"count": len(ids), "effect_rms": rms,
                      "positive_fraction": float(np.mean(target > 0)),
                      "unrelated_control_fraction":
                          float(np.sqrt(np.mean(control ** 2)) / max(rms, 1e-30))}
            report["passes"] = bool(
                rms >= BARS["minimum_single_site_effect_rms"]
                and report["positive_fraction"] >= BARS["minimum_single_site_positive_fraction"]
                and report["unrelated_control_fraction"] <= BARS["maximum_unrelated_control_fraction"])
            single_reports[f"site{j + 1}|{direction}"] = report
            single_pass &= report["passes"]

    anticausal = float(np.max(np.abs(number_effects["site2"][:, 0])))
    predicted = number_effects["site1"] + number_effects["site2"]
    actual = number_effects["both"]
    composition = {"overall": metrics(predicted, actual),
                   "site1": metrics(predicted[:, 0], actual[:, 0]),
                   "site2": metrics(predicted[:, 1], actual[:, 1])}
    for report in composition.values():
        report["passes"] = composition_passes(report)
    interaction = actual - predicted
    interaction_fraction = float(np.sqrt(np.mean(interaction ** 2))
                                 / max(np.sqrt(np.mean(actual ** 2)), 1e-30))

    checkpoint_match = (checkpoint.weights_sha256
                        == decoder["checkpoint_weights_sha256"]
                        == fresh["checkpoint_weights_sha256"])
    instrument = bool(
        checkpoint_match and finite and direction_agreement.all()
        and float(signed_decoder_margin.min()) > 0
        and oracle_write_error <= BARS["maximum_oracle_write_error"]
        and zero_error <= BARS["maximum_zero_replay_logit_error"]
        and all(cell["accuracy"] >= BARS["minimum_native_accuracy"]
                for cell in capability.values())
        and counts == PRICE)
    pred_b = bool(instrument and single_pass
                  and anticausal <= BARS["maximum_anticausal_absolute_effect"])
    pred_c = bool(pred_b and all(report["passes"] for report in composition.values()))
    predictions = dict(zip(PREDICTION_REGISTRY, (instrument, pred_b, pred_c)))
    terminal = ("embedding_decoded_writer_two_site_composition" if pred_c else
                "invalid" if not instrument else "embedding_decoded_writer_two_site_null")
    result = {
        "schema": "subject_number_embedding_writer_two_site_v1_result",
        "terminal": terminal, "predictions": predictions,
        "instrument": {
            "checkpoint_match": checkpoint_match,
            "candidate_oracle_direction_agreement": float(direction_agreement.mean()),
            "candidate_oracle_write_max_absolute_error": oracle_write_error,
            "decoder_accuracy": float(np.mean(signed_decoder_margin > 0)),
            "decoder_minimum_signed_margin": float(signed_decoder_margin.min()),
            "decoder_mean_signed_margin": float(signed_decoder_margin.mean()),
            "zero_replay_max_logit_error": zero_error,
            "native_capability": capability, "finite": finite, "counts": counts,
        },
        "single_site": single_reports,
        "anticausal_site2_to_site1_max_absolute_effect": anticausal,
        "composition": composition,
        "interaction_over_joint_rms": interaction_fraction,
        "records": [{
            "row_id": row["row_id"], "template": row["template_id"],
            "number_pair": row["number_pair"],
            "decoder_scores": [float(value) for value in decoder_scores[i]],
            "candidate_directions": candidate_directions[i],
            "site1_effect": number_effects["site1"][i].tolist(),
            "site2_effect": number_effects["site2"][i].tolist(),
            "joint_effect": actual[i].tolist(), "interaction": interaction[i].tolist(),
        } for i, row in enumerate(rows)],
        "bars": BARS, "price": PRICE,
        "row_manifest_sha256": frozen["row_manifest_sha256"],
        "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": time.perf_counter() - started,
        "scope": ("Prospective composition of the frozen token-embedding number decoder, "
                  "fixed direction/cardinality coefficient lookup, and fixed L11H3 rank-one "
                  "write on untouched two-clause prompts; no direction labels enter the candidate."),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("terminal", "predictions", "instrument", "single_site",
                       "anticausal_site2_to_site1_max_absolute_effect", "composition",
                       "interaction_over_joint_rms")}, indent=2, sort_keys=True))
    assert instrument


if __name__ == "__main__":
    main()
