#!/usr/bin/env python3
"""Multi-seed cross-family DAS at the verified pending-opener site."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402

ROWS = ROOT / "pending_opener_multifamily_rows_rung537.json"
CONTROLS = ROOT / "pending_opener_controls_rung537.json"
TARGET_CEILINGS = ROOT / "pending_opener_common_site_rung538_results.json"
CONTROL_CEILINGS = ROOT / "pending_opener_control_ceilings_rung539_results.json"
PREREG = POLY / "PENDING_OPENER_CROSS_FAMILY_DAS_RUNG540_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_cross_family_das_rung540_results.json"
BUNDLE = ROOT / "pending_opener_cross_family_das_rung540_bundle.pt"
HASHES = {
    ROWS: "c62cdf3929231e06de6883d74f3ab2c86bd524e02474bb2259267d6976e9e7d9",
    CONTROLS: "f2693b9b78a9266619afc45ceb6f70e4f2339aa1980263ca22d3ea4453145494",
    TARGET_CEILINGS: "f011399614953c958faf2a12ef15e938dcc2f5e3f52ea868763de2a82443a205",
    CONTROL_CEILINGS: "d0cf53b6e26df46b113a9a8bf18bc9b86222536b3d2621ff90d690240a8e3a0c",
    PREREG: "c1541e8a9dba7e73daef20c16223b37a0ee845c18551d428bee63a74430a9b2e",
}
TARGET_FAMILIES = ("opener_type_substitution", "closed_then_reopened_type")
CONTROL_FAMILIES = ("pending_state_preserved_surface_edit", "nonopener_punctuation_substitution")
TRAIN_SOURCES = ("direct", "structural", "joint")
SOURCE_FAMILIES = {"direct": TARGET_FAMILIES[:1], "structural": TARGET_FAMILIES[1:], "joint": TARGET_FAMILIES}
SPLITS, RANKS, SEEDS = ("FIT", "SELECT"), (1, 2, 4, 8, 16), (0, 1, 2)
RANDOM_SEEDS = (100, 101, 102, 103, 104)
STEPS, LR, BATCH = 240, 5e-3, 16
D, PATCH_LAYER = 1152, 8
BOOTSTRAPS, BOOTSTRAP_SEED = 2000, 540
CONTROL_DENOM_FLOOR = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    targets, controls = json.loads(TARGET_CEILINGS.read_text()), json.loads(CONTROL_CEILINGS.read_text())
    target_gate_keys = tuple("pred_" + suffix for suffix in (
        "a_exact_instrument", "b_common_live_site", "c_frozen_causal_order_selection"))
    if targets.get("selected_site") != "resid8" or not all(targets.get(key) is True for key in target_gate_keys):
        raise RuntimeError("R538 did not authorize resid8")
    control_gate_keys = tuple("pred_" + suffix for suffix in (
        "a_exact_instrument", "b_surface_invariance_causally_testable",
        "c_nonopener_control_causally_testable"))
    if not all(controls.get(key) is True for key in control_gate_keys):
        raise RuntimeError("R539 controls are not causally informative")
    main, control_doc = json.loads(ROWS.read_text())["rows"], json.loads(CONTROLS.read_text())["rows"]
    target_rows = [row for row in main if row["split"] in SPLITS and row["family_id"] in TARGET_FAMILIES]
    control_rows = ([row for row in main if row["split"] in SPLITS and row["family_id"] == CONTROL_FAMILIES[0]]
                    + [row for row in control_doc if row["split"] in SPLITS and row["family_id"] == CONTROL_FAMILIES[1]])
    if len(target_rows) != 128 or len(control_rows) != 128:
        raise RuntimeError("R540 row count changed")
    return target_rows, control_rows, targets, controls


@torch.no_grad()
def prefix_state(model, tokens):
    x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1 = x, None
    for layer in range(PATCH_LAYER):
        x, v1 = model.transformer.h[layer](x, v1, x0)
    return x, x0, v1


def tail_logits(model, x, x0, v1, *, grad):
    with (torch.enable_grad() if grad else torch.no_grad()):
        for layer in range(PATCH_LAYER, 18):
            x, v1 = model.transformer.h[layer](x, v1, x0)
        return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30)).float()


def ceiling_maps(target_rows, control_rows, targets, controls):
    target_map, control_map = {}, {}
    for split in SPLITS:
        for family in TARGET_FAMILIES:
            ordered = [row for row in target_rows if row["split"] == split and row["family_id"] == family]
            for direction in ("base_to_donor", "donor_to_base"):
                values = targets["raw_donorward_movements"]["resid8"][split][family][direction]
                if len(ordered) != len(values): raise RuntimeError("target ceiling alignment")
                target_map.update({(row["row_id"], direction): float(value) for row, value in zip(ordered, values)})
        for family in CONTROL_FAMILIES:
            ordered = [row for row in control_rows if row["split"] == split and row["family_id"] == family]
            for direction in ("base_to_donor", "donor_to_base"):
                values = controls["raw_sufficient_statistics"][split][family][direction]["endpoint_change"]
                if len(ordered) != len(values): raise RuntimeError("control ceiling alignment")
                control_map.update({(row["row_id"], direction): float(value) for row, value in zip(ordered, values)})
    if min(target_map.values()) <= 0: raise RuntimeError("nonpositive target denominator")
    return target_map, control_map


def build_sequences(rows):
    length = max(max(len(row["base_ids"]), len(row["donor_ids"])) for row in rows)
    sequences, finals, lookup = [], [], {}
    for row in rows:
        for side in ("base", "donor"):
            lookup[(row["row_id"], side)] = len(sequences)
            ids = row[f"{side}_ids"]
            sequences.append(ids + [50256] * (length - len(ids))); finals.append(len(ids) - 1)
    return torch.tensor(sequences), torch.tensor(finals), lookup


@torch.no_grad()
def collect_states(model, rows):
    tokens, finals, lookup = build_sequences(rows)
    xs, x0s, v1s, native = [], [], [], []
    for start in range(0, len(tokens), BATCH):
        batch, fin = tokens[start:start+BATCH].cuda(), finals[start:start+BATCH].cuda()
        x, x0, v1 = prefix_state(model, batch)
        logits = tail_logits(model, x, x0, v1, grad=False)
        arange = torch.arange(len(batch), device="cuda")
        xs.append(x.detach()); x0s.append(x0.detach()); v1s.append(v1.detach())
        native.append(logits[arange, fin][:, [1, 8]].detach())
    return {"x": torch.cat(xs), "x0": torch.cat(x0s), "v1": torch.cat(v1s),
            "native": torch.cat(native), "finals": finals.cuda(), "lookup": lookup}


def build_samples(rows, states, ceiling, *, controls):
    samples = []
    for row in rows:
        for direction, target_side, source_side in (
            ("base_to_donor", "base", "donor"), ("donor_to_base", "donor", "base")):
            sample = {"row_id": row["row_id"], "family": row["family_id"], "split": row["split"],
                      "direction": direction, "target_index": states["lookup"][(row["row_id"], target_side)],
                      "source_index": states["lookup"][(row["row_id"], source_side)],
                      "full_effect": ceiling[(row["row_id"], direction)]}
            if controls:
                sample.update(source_answer_col=1, target_answer_col=0)
            else:
                source_id = row["donor_answer_id"] if direction == "base_to_donor" else row["base_answer_id"]
                target_id = row["base_answer_id"] if direction == "base_to_donor" else row["donor_answer_id"]
                sample.update(source_answer_col=0 if source_id == 1 else 1,
                              target_answer_col=0 if target_id == 1 else 1)
            samples.append(sample)
    return samples


def patch_batch(model, states, samples, Q, dose, *, grad):
    ti = torch.tensor([s["target_index"] for s in samples], device="cuda")
    si = torch.tensor([s["source_index"] for s in samples], device="cuda")
    finals = states["finals"][ti]; arange = torch.arange(len(samples), device="cuda")
    x = states["x"][ti].clone(); current = x[arange, finals]
    delta = states["x"][si, states["finals"][si]] - current
    x[arange, finals] = current + dose * (delta @ Q) @ Q.T
    logits = tail_logits(model, x, states["x0"][ti], states["v1"][ti], grad=grad)
    return logits[arange, finals][:, [1, 8]]


def train_projector(model, states, samples, rank, seed):
    generator = torch.Generator().manual_seed(10_000 * rank + seed)
    P = (torch.randn(D, rank, generator=generator) / math.sqrt(D)).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([P], lr=LR)
    for _ in range(STEPS):
        batch = [samples[i] for i in torch.randint(len(samples), (BATCH,), generator=generator).tolist()]
        Q = torch.linalg.qr(P, mode="reduced")[0]
        logits = patch_batch(model, states, batch, Q, 1.0, grad=True)
        ti = torch.tensor([s["target_index"] for s in batch], device="cuda")
        sc = torch.tensor([s["source_answer_col"] for s in batch], device="cuda")
        tc = torch.tensor([s["target_answer_col"] for s in batch], device="cuda")
        arange = torch.arange(len(batch), device="cuda")
        movement = ((logits[arange, sc] - logits[arange, tc])
                    - (states["native"][ti][arange, sc] - states["native"][ti][arange, tc]))
        recovery = movement / torch.tensor([s["full_effect"] for s in batch], device="cuda")
        loss = (recovery - 1).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return torch.linalg.qr(P.detach(), mode="reduced")[0]


def evaluate_values(model, states, samples, Q, dose, *, controls):
    values = []
    with torch.no_grad():
        for start in range(0, len(samples), BATCH):
            batch = samples[start:start+BATCH]
            logits = patch_batch(model, states, batch, Q, dose, grad=False)
            ti = torch.tensor([s["target_index"] for s in batch], device="cuda")
            native = states["native"][ti]; arange = torch.arange(len(batch), device="cuda")
            if controls:
                effect = (logits[:, 1] - logits[:, 0]) - (native[:, 1] - native[:, 0])
                values.extend(effect.cpu().tolist())
            else:
                sc = torch.tensor([s["source_answer_col"] for s in batch], device="cuda")
                tc = torch.tensor([s["target_answer_col"] for s in batch], device="cuda")
                movement = ((logits[arange, sc] - logits[arange, tc])
                            - (native[arange, sc] - native[arange, tc]))
                full = torch.tensor([s["full_effect"] for s in batch], device="cuda")
                values.extend((movement / full).cpu().tolist())
    return values


def bootstrap_lower(values, seed):
    array = np.asarray(values); generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def score_projector(model, states, target_samples, control_samples, Q, seed):
    target_report, control_report, response, passed = {}, {}, [], True
    for family in TARGET_FAMILIES:
        target_report[family] = {}
        for direction in ("base_to_donor", "donor_to_base"):
            subset = [s for s in target_samples if s["split"] == "SELECT" and s["family"] == family and s["direction"] == direction]
            values = evaluate_values(model, states, subset, Q, 1.0, controls=False)
            report = {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                      "bootstrap95_lower_mean": bootstrap_lower(values, seed),
                      "positive_fraction": float(np.mean(np.asarray(values) > 0)), "values": values}
            seed += 1
            report["passed"] = bool(report["median"] >= .5 and report["bootstrap95_lower_mean"] > 0
                                    and report["positive_fraction"] >= .75)
            passed &= report["passed"]; target_report[family][direction] = report; response.extend(values)
    for family in CONTROL_FAMILIES:
        control_report[family] = {}
        for direction in ("base_to_donor", "donor_to_base"):
            subset = [s for s in control_samples if s["split"] == "SELECT" and s["family"] == family and s["direction"] == direction]
            values = evaluate_values(model, states, subset, Q, 1.0, controls=True)
            full = [s["full_effect"] for s in subset]
            mean_abs = float(np.mean(np.abs(values))); ratio = mean_abs / float(np.mean(np.abs(full)))
            report = {"n": len(values), "mean_absolute": mean_abs, "fraction_of_full": ratio, "values": values,
                      "passed": bool(mean_abs <= .10 and ratio <= .25)}
            passed &= report["passed"]; control_report[family][direction] = report
            response.extend([v / math.copysign(max(abs(d), CONTROL_DENOM_FLOOR), d or 1.0)
                             for v, d in zip(values, full)])
    return {"targets": target_report, "controls": control_report, "passed": bool(passed)}, response


def cosine_rms(a, b):
    x, y = np.asarray(a), np.asarray(b)
    return (float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)),
            float(np.sqrt(np.mean((x-y)**2))))


def main():
    started = time.time()
    target_rows, control_rows, target_ceiling, control_ceiling = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "target_pairs": len(target_rows),
                          "control_pairs": len(control_rows), "ranks": RANKS, "seeds": SEEDS,
                          "training_sources": TRAIN_SOURCES, "fits": 45,
                          "gradient_suffix_evaluations": 45*STEPS,
                          "final_or_ood_opened": False}, indent=2)); return
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    target_map, control_map = ceiling_maps(target_rows, control_rows, target_ceiling, control_ceiling)
    states = collect_states(model, target_rows + control_rows)
    target_samples = build_samples(target_rows, states, target_map, controls=False)
    control_samples = build_samples(control_rows, states, control_map, controls=True)
    fits, q_bundle, response_bundle, score_seed = {}, {}, {}, BOOTSTRAP_SEED
    for rank in RANKS:
        fits[str(rank)] = {}
        for source in TRAIN_SOURCES:
            fits[str(rank)][source] = {}
            training = [s for s in target_samples if s["split"] == "FIT" and s["family"] in SOURCE_FAMILIES[source]]
            for seed in SEEDS:
                Q = train_projector(model, states, training, rank, seed)
                report, response = score_projector(model, states, target_samples, control_samples, Q, score_seed)
                score_seed += 20; fits[str(rank)][source][str(seed)] = report
                q_bundle[(rank, source, seed)] = Q.cpu(); response_bundle[(rank, source, seed)] = response
    random_controls, random_pass = {}, {}
    select_targets = [s for s in target_samples if s["split"] == "SELECT"]
    for rank in RANKS:
        means = []
        for seed in RANDOM_SEEDS:
            generator = torch.Generator().manual_seed(seed + 10_000*rank)
            Q = torch.linalg.qr(torch.randn(D, rank, generator=generator).cuda(), mode="reduced")[0]
            means.append(float(np.mean(evaluate_values(model, states, select_targets, Q, 1.0, controls=False))))
        random_controls[str(rank)] = means; random_pass[str(rank)] = float(np.mean(means)) < .10
    eligible = []
    for rank in RANKS:
        stable = {source: sum(fits[str(rank)][source][str(seed)]["passed"] for seed in SEEDS) >= 2
                  for source in TRAIN_SOURCES}
        fits[str(rank)]["source_seed_stability"] = stable
        fits[str(rank)]["rank_eligible"] = bool(all(stable.values()) and random_pass[str(rank)])
        if fits[str(rank)]["rank_eligible"]: eligible.append(rank)
    selected_rank = min(eligible) if eligible else None
    selected, equivalence, doses = {}, {}, {}
    if selected_rank:
        for source in TRAIN_SOURCES:
            selected[source] = min(seed for seed in SEEDS if fits[str(selected_rank)][source][str(seed)]["passed"])
        for i, a in enumerate(TRAIN_SOURCES):
            for b in TRAIN_SOURCES[i+1:]:
                ka, kb = (selected_rank, a, selected[a]), (selected_rank, b, selected[b])
                cosine, rms = cosine_rms(response_bundle[ka], response_bundle[kb])
                overlap = float(torch.linalg.norm(q_bundle[ka].T @ q_bundle[kb]) / math.sqrt(selected_rank))
                equivalence[f"{a}~{b}"] = {"response_cosine": cosine, "response_rms_difference": rms,
                    "subspace_overlap_rms": overlap, "operationally_equivalent": bool(cosine >= .90 and rms <= .15)}
        for source in TRAIN_SOURCES:
            Q = q_bundle[(selected_rank, source, selected[source])].cuda(); doses[source] = {}
            for dose in (0., .5, 1., 1.5):
                values = evaluate_values(model, states, select_targets, Q, dose, controls=False)
                doses[source][str(dose)] = {"mean": float(np.mean(values)), "median": float(np.median(values))}
    equivalent = bool(selected_rank and all(item["operationally_equivalent"] for item in equivalence.values()))
    result = {"rung": 540, "stage": "fit_select_cross_family_das",
        "pred_a_exact_instrument": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "pred_b_two_way_cross_family_transfer": selected_rank is not None,
        "pred_c_operational_equivalence_across_training_sources": equivalent,
        "pred_d_controls_selective_and_random_below_bar": selected_rank is not None,
        "strong_null": selected_rank is None, "selected_rank": selected_rank,
        "selected_seed_by_training_source": selected, "fits": fits,
        "random_controls": random_controls, "operational_equivalence": equivalence,
        "dose_response": doses, "model_weights_updated": False,
        "gradient_suffix_evaluations": 45*STEPS, "model_backwards": 45*STEPS,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "elapsed_seconds": time.time()-started,
        "next_step": ("freeze_one_shot_final_and_ood_test" if selected_rank and equivalent
                      else "interpret_shared_private_or_optimization_failure_without_opening_final_ood")}
    torch.save({"projectors": q_bundle, "selected": selected, "selected_rank": selected_rank}, BUNDLE)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    summary = {key: value for key, value in result.items() if key.startswith("pred_")}
    summary.update({key: result[key] for key in ("strong_null", "selected_rank",
        "selected_seed_by_training_source", "gradient_suffix_evaluations", "model_backwards",
        "forbidden_splits_opened", "next_step")})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
