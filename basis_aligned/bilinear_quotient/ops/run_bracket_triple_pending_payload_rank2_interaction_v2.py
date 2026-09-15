#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 5forwards1440seq; bracket type-payload rank2 times live score;1fit0backwards0updates.
"""Fit delimiter payload prototypes without outcomes and test a fourth construction."""
from __future__ import annotations
from collections import defaultdict
import hashlib, json, math, signal, sys, time
from pathlib import Path

RUNNER = Path(__file__).resolve()
V1_RUNNER = RUNNER.with_name("run_bracket_triple_pending_payload_rank2_interaction_v1.py")
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
TRAIN_ROWS = POLY / "BRACKET_NESTED_PENDING_OOD_V1_ROWS.json"
EVAL_ROWS = POLY / "BRACKET_TRIPLE_PENDING_OOD_V1_ROWS.json"
BUILDER = POLY / "build_bracket_triple_pending_ood_v1_rows.py"
PRIOR = POLY / "BRACKET_NESTED_PENDING_SCORE_PAYLOAD_INTERACTION_V2_RESULT.json"
PREREG = POLY / "BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_PREREGISTRATION.md"
BINDING = POLY / "BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_BINDING.json"
OUT = POLY / "BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_RESULT.json"
TYPES = ("parenthesis", "square", "quote")
PRICE = {"forwards": 5, "sequences": 1440, "fits": 1, "backwards": 0, "updates": 0}
BARS = {
    "replay_max": 1e-5, "capability_accuracy_min": .75, "exact_positive_min": .90,
    "source_cosine_min": .90, "source_relative_l2_max": .50,
    "effect_cosine_min": .95, "effect_relative_l2_max": .30,
    "effect_sign_min": .90, "effect_norm_ratio_min": .70, "effect_norm_ratio_max": 1.30,
    "pair_cosine_min": .85, "pair_relative_l2_max": .40, "pair_sign_min": .90,
    "score_only_improvement_min": .20, "control_to_target_rms_max": .50,
}
PREDICTION_REGISTRY = {
    "pred_a_exact_instrument_and_capability": None,
    "pred_b_exact_joint_parent_live": None,
    "pred_c_rank2_payload_source_transfers": None,
    "pred_d_rank2_interaction_effect_transfers": None,
    "pred_e_rank2_beats_score_only": None,
    "pred_f_control_selectivity": None,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def vector_metrics(actual, predicted):
    actual = [float(x) for x in actual]
    predicted = [float(x) for x in predicted]
    dot = sum(a * b for a, b in zip(actual, predicted))
    an = math.sqrt(sum(a * a for a in actual))
    pn = math.sqrt(sum(p * p for p in predicted))
    return {
        "count": len(actual), "cosine": dot / max(an * pn, 1e-30),
        "relative_l2_error": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted))) / max(an, 1e-30),
        "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(actual),
        "predicted_to_actual_norm_ratio": pn / max(an, 1e-30),
    }


def main():
    sys.path.insert(0, str(OPS))
    from circuit_exactness_preflight import (
        managed_execution_mode, validate_literal_prediction_registry, validate_result_contract,
    )
    binding = json.loads(BINDING.read_text())
    paths = {"train_rows": TRAIN_ROWS, "eval_rows": EVAL_ROWS, "builder": BUILDER,
             "prior_result": PRIOR, "preregistration": PREREG, "v1_runner": V1_RUNNER}
    assert all(digest(paths[name]) == expected for name, expected in binding["files"].items())
    train = json.loads(TRAIN_ROWS.read_text())
    evaluate = json.loads(EVAL_ROWS.read_text())
    assert canonical(train["rows"]) == train["row_manifest_sha256"]
    assert canonical(evaluate["rows"]) == evaluate["row_manifest_sha256"]
    assert binding["bars"] == BARS and binding["price"] == PRICE
    validate_literal_prediction_registry(RUNNER.read_text(), PREDICTION_REGISTRY)
    mode = managed_execution_mode(__import__("os").environ)
    if mode == "preflight":
        print(json.dumps({"dryrun": True, "model_loaded": False, "gpu_accessed": False,
                          "train_rows": train["row_count"], "eval_rows": evaluate["row_count"],
                          "bars": BARS, "price": PRICE,
                          "predicates": list(PREDICTION_REGISTRY)}, sort_keys=True))
        return
    assert not OUT.exists()
    signal.alarm(600)
    import torch
    import run_bracket_l13h8_source_region_payload_factorial as exact
    from circuit_fast_screen_managed_runner import atomic_create_json
    torch.set_num_threads(2)
    tm, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    tagged_rows = [("train", row) for row in train["rows"]] + [("eval", row) for row in evaluate["rows"]]
    endpoints = [(split, row, side) for split, row in tagged_rows for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for _split, row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device="cuda")
    finals, sources = [], []
    for index, (_split, row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, device="cuda")
        finals.append(len(ids) - 1)
        sources.append(row[f"{side}_open_position"])
    finals_t = torch.tensor(finals, device="cuda")
    sources_t = torch.tensor(sources, device="cuda")
    ar = torch.arange(len(endpoints), device="cuda")
    counts = [0, 0]

    def count(_module, args, _output):
        counts[0] += 1
        counts[1] += len(args[0])

    handle = model.transformer.h[0].attn.register_forward_hook(count)
    tic = time.perf_counter()
    try:
        with torch.inference_mode():
            native = exact.native_logits(model, tokens, tm, F).cpu()
            replay_gpu, factors = exact.factor_forward(model, tokens, finals_t, {}, tm, F, facade)
            replay = replay_gpu.cpu()
            del replay_gpu
            torch.cuda.empty_cache()
            p = factors["p"][ar, sources_t]
            u = factors["u"][ar, sources_t]
            donor_index = ar ^ 1
            pd = p[donor_index]
            ud = u[donor_index]
            train_payloads = defaultdict(list)
            for index, (split, row, side) in enumerate(endpoints):
                if split == "train" and row["program_role"] == "target":
                    train_payloads[row[f"{side}_type"]].append(u[index])
            prototypes = torch.stack([torch.stack(train_payloads[name]).double().mean(0) for name in TYPES])
            centered = prototypes - prototypes.mean(0, keepdim=True)
            _singular_u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[:2]
            coefficients = centered @ basis.T
            reconstructed = coefficients @ basis
            program_payload = []
            for index, (split, row, side) in enumerate(endpoints):
                if split != "eval":
                    program_payload.append(torch.zeros_like(u[index], dtype=torch.double))
                    continue
                if row["program_role"] == "target":
                    other = "donor" if side == "base" else "base"
                    recipient_type, donor_type = row[f"{side}_type"], row[f"{other}_type"]
                else:
                    recipient_type = donor_type = row["inner_type"]
                ri, di = TYPES.index(recipient_type), TYPES.index(donor_type)
                program_payload.append((coefficients[di] - coefficients[ri]) @ basis)
            program_payload = torch.stack(program_payload).to(dtype=u.dtype)
            self_term = p[:, None] * u
            exact_term = pd[:, None] * ud
            compressed_term = pd[:, None] * (u + program_payload)
            score_only_term = pd[:, None] * u
            replacements = {"exact": exact_term, "compressed": compressed_term, "score_only": score_only_term}
            arms = {}
            for name, value in replacements.items():
                logits, _unused_factors = exact.factor_forward(
                    model, tokens, finals_t, {}, tm, F, facade,
                    replacement_terms=value, source_positions=sources_t,
                )
                arms[name] = logits.cpu()
                del logits, _unused_factors
                torch.cuda.empty_cache()
    finally:
        handle.remove()
    replay_error = max(float((native[i, finals[i]] - replay[i, finals[i]]).abs().max()) for i in range(len(endpoints)))
    eval_indices = [i for i, item in enumerate(endpoints) if item[0] == "eval"]
    capability_cells = defaultdict(list)
    records = []
    actual_du, predicted_du = [], []
    for index in eval_indices:
        _split, row, side = endpoints[index]
        answer = int(row[f"{side}_answer_id"])
        capability_key = (row["program_role"], answer, side) if row["program_role"] == "control" else (
            "target", answer, int(row[f"{'donor' if side == 'base' else 'base'}_answer_id"]),
        )
        capability_cells[capability_key].append(float(exact.closer_margin(native[index, finals[index]], answer)))
        record = {"row_id": row["row_id"], "side": side, "program_role": row["program_role"]}
        if row["program_role"] == "target":
            other = "donor" if side == "base" else "base"
            direction = "base_to_donor" if side == "base" else "donor_to_base"
            record["ordered_pair"] = f"{answer}->{int(row[f'{other}_answer_id'])}"
            for name in replacements:
                record[name + "_effect"] = float(exact.endpoint_change(
                    replay[index, finals[index]], arms[name][index, finals[index]], row, direction,
                ))
            actual_du.extend((ud[index] - u[index]).double().cpu().tolist())
            predicted_du.extend(program_payload[index].double().cpu().tolist())
        else:
            before = float(exact.closer_margin(replay[index, finals[index]], answer))
            record["control_changes"] = {name: float(
                exact.closer_margin(arms[name][index, finals[index]], answer) - before
            ) for name in replacements}
        records.append(record)
    capability = {"|".join(map(str, key)): {
        "n": len(values), "accuracy": sum(value > 0 for value in values) / len(values),
        "mean_closer_margin": sum(values) / len(values),
    } for key, values in sorted(capability_cells.items(), key=lambda item: str(item[0]))}
    targets = [record for record in records if record["program_role"] == "target"]
    controls = [record for record in records if record["program_role"] == "control"]
    exact_effect = [record["exact_effect"] for record in targets]
    compressed_effect = [record["compressed_effect"] for record in targets]
    score_effect = [record["score_only_effect"] for record in targets]
    effect_metrics = vector_metrics(exact_effect, compressed_effect)
    score_metrics = vector_metrics(exact_effect, score_effect)
    source_metrics = vector_metrics(actual_du, predicted_du)
    by_pair = {}
    for pair in sorted({record["ordered_pair"] for record in targets}):
        items = [record for record in targets if record["ordered_pair"] == pair]
        by_pair[pair] = {
            "n": len(items),
            "exact_positive_fraction": sum(record["exact_effect"] > 0 for record in items) / len(items),
            "compressed_vs_exact": vector_metrics(
                [record["exact_effect"] for record in items],
                [record["compressed_effect"] for record in items],
            ),
        }
    target_rms = math.sqrt(sum(value * value for value in exact_effect) / len(exact_effect))
    control_rms = math.sqrt(sum(
        record["control_changes"]["compressed"] ** 2 for record in controls
    ) / len(controls))
    rank = int((singular_values > singular_values[0] * 1e-10).sum().cpu())
    capable = all(value["accuracy"] >= BARS["capability_accuracy_min"]
                  and value["mean_closer_margin"] > 0 for value in capability.values())
    instrument = counts == [PRICE["forwards"], PRICE["sequences"]] and replay_error <= BARS["replay_max"] \
        and rank <= 2 and len(targets) == len(controls) == 72 and capable
    live = all(value["exact_positive_fraction"] >= BARS["exact_positive_min"] for value in by_pair.values())
    source_ok = source_metrics["cosine"] >= BARS["source_cosine_min"] \
        and source_metrics["relative_l2_error"] <= BARS["source_relative_l2_max"]
    effect_ok = effect_metrics["cosine"] >= BARS["effect_cosine_min"] \
        and effect_metrics["relative_l2_error"] <= BARS["effect_relative_l2_max"] \
        and effect_metrics["sign_agreement"] >= BARS["effect_sign_min"] \
        and BARS["effect_norm_ratio_min"] <= effect_metrics["predicted_to_actual_norm_ratio"] <= BARS["effect_norm_ratio_max"] \
        and all(value["compressed_vs_exact"]["cosine"] >= BARS["pair_cosine_min"]
                and value["compressed_vs_exact"]["relative_l2_error"] <= BARS["pair_relative_l2_max"]
                and value["compressed_vs_exact"]["sign_agreement"] >= BARS["pair_sign_min"]
                for value in by_pair.values())
    improvement = score_metrics["relative_l2_error"] - effect_metrics["relative_l2_error"]
    selective = control_rms / max(target_rms, 1e-30) <= BARS["control_to_target_rms_max"]
    predictions = {
        "pred_a_exact_instrument_and_capability": bool(instrument),
        "pred_b_exact_joint_parent_live": bool(instrument and live),
        "pred_c_rank2_payload_source_transfers": bool(instrument and source_ok),
        "pred_d_rank2_interaction_effect_transfers": bool(instrument and effect_ok),
        "pred_e_rank2_beats_score_only": bool(instrument and improvement >= BARS["score_only_improvement_min"]),
        "pred_f_control_selectivity": bool(instrument and selective),
    }
    if not (counts == [PRICE["forwards"], PRICE["sequences"]] and replay_error <= BARS["replay_max"] and rank <= 2):
        terminal = "invalid"
    elif not capable:
        terminal = "fourth_construction_capability_null"
    elif all(predictions.values()):
        terminal = "payload_rank2_interaction_transfer"
    else:
        terminal = "payload_rank2_interaction_null"
    result = {
        "schema": "bracket_triple_pending_payload_rank2_interaction_v2_result",
        "terminal": terminal, "predictions": predictions,
        "instrument": {"native_factor_replay_max_logit_error": replay_error,
                       "centered_prototype_rank": rank,
                       "singular_values": singular_values.cpu().tolist()},
        "capability_cells": capability,
        "source_payload_difference": source_metrics,
        "compressed_vs_exact_effect": effect_metrics,
        "score_only_vs_exact_effect": score_metrics,
        "relative_l2_improvement_over_score_only": improvement,
        "by_ordered_pair": by_pair,
        "target_exact_effect_rms": target_rms, "compressed_control_rms": control_rms,
        "control_to_target_rms": control_rms / max(target_rms, 1e-30),
        "program": {"types": list(TYPES), "rank": 2,
                    "basis": basis.cpu().tolist(), "type_coefficients": coefficients.cpu().tolist(),
                    "fit_inputs": "third-construction target opener payloads only; no logits/outcomes"},
        "price": {**PRICE, "observed_forwards": counts[0], "observed_sequences": counts[1]},
        "claim_boundary": "Outcome-blind rank-two delimiter payload differences multiplied by live donor score, tested prospectively on a fresh triple-pending construction; native recipient payload and suffix remain external.",
        "train_rows_sha256": digest(TRAIN_ROWS), "eval_rows_sha256": digest(EVAL_ROWS),
        "binding_sha256": digest(BINDING), "runner_sha256": digest(RUNNER),
        "checkpoint_sha256": checkpoint.weights_sha256, "wall_seconds": time.perf_counter() - tic,
        "records": records,
    }
    validate_result_contract(result, PREDICTION_REGISTRY)
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "terminal", "predictions", "instrument", "source_payload_difference",
        "compressed_vs_exact_effect", "score_only_vs_exact_effect",
        "relative_l2_improvement_over_score_only", "control_to_target_rms", "price",
    )}, indent=2))
    assert terminal != "invalid"


if __name__ == "__main__":
    main()
