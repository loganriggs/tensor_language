#!/usr/bin/env python3
"""R556: jointly train pending-opener transfer and answer-preserving invariance.

Pred A: the exact fresh R545/R546/R548 authority, checkpoint, split opening,
state collection, and frozen 15-fit/3,600-backward budget are live.

Pred B: at least one rank in {1,2,4,8,16} has >=2/3 seeds that pass every
SELECT target and control cell after training on FIT only.

Pred C: five dimension-matched random projectors per rank average <.10 target
recovery.  Target bars are median recovery >=.50, bootstrap lower mean >0,
and >=.75 donorward rows.  Control bars are <=.10 absolute closer-margin
change, <=.25 of complete-head margin change, and <=.25 of complete-head
full-vocabulary RMS in every family/direction cell.

Null: no rank is stable; do not extend the rank sweep.  Literal experiment
price is 15 projectors of 128*k values, 3,600 gradient-bearing layer-13--17
suffix evaluations, <=700 no-gradient suffix evaluations, and frozen model
weights.  This is a selective causal-interchange test, not reconstruction.
"""

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

ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT = ROOT / "pending_opener_three_value_fresh_rows_rung545_receipt.json"
CEILING = ROOT / "pending_opener_three_value_confirmation_rung546_results.json"
AUDIT = ROOT / "pending_opener_three_value_confirmation_rung548_audit.json"
PREREG = POLY / "PENDING_OPENER_TARGET_INVARIANCE_DAS_RUNG556_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_target_invariance_das_rung556_results.json"
BUNDLE = ROOT / "pending_opener_target_invariance_das_rung556_projectors.pt"
HASHES = {
    ROWS: "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9",
    RECEIPT: "a6b3e7468f510277b247cb78148b619625ecdde07f9ba264e5358f7bb5138609",
    CEILING: "209b9bfcc20bff13bb37d822137003d6878506e66b0d9321ba0a0f7e9d8f2c5c",
    AUDIT: "25acb35355f457163c1ed1183aeb55aea0c08a224992d688250ba5e272564875",
    PREREG: "706a4fda4788dace89de6a4ae5f41cf3bd7e56ff194d8956d8e977b7ced1dc44",
}
TARGETS = ("direct_three_value_type_substitution", "completed_then_reopened_three_value_order")
CONTROLS = (
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
)
SPLITS = ("FIT", "SELECT")
RANKS = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)
RANDOM_SEEDS = (100, 101, 102, 103, 104)
D, HEAD_D, PATCH_LAYER = 1152, 128, 13
STEPS, LR, BATCH = 240, 5e-3, 16
STATE_BATCH = 16
BOOTSTRAPS, BOOTSTRAP_SEED = 2000, 556
CONTROL_RMS_FLOOR = .01
EXPECTED_PAIRS = 540
EXPECTED_SEQUENCES = 1080
EXPECTED_NATIVE_FORWARDS = math.ceil(EXPECTED_SEQUENCES / STATE_BATCH)
EXPECTED_GRAD_SUFFIX_EVALS = len(RANKS) * len(SEEDS) * STEPS
EXPECTED_SCORE_SUFFIX_EVALS = 675


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_authority() -> tuple[list[dict], dict]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    receipt = json.loads(RECEIPT.read_text())
    ceiling = json.loads(CEILING.read_text())
    audit = json.loads(AUDIT.read_text())
    assert receipt["unique_token_sequence_count"] == 1800
    assert ceiling["all_gates_pass"] is True and ceiling["model_forwards"] == 204
    assert ceiling["evaluated_splits"] == list(SPLITS) and ceiling["forbidden_splits_opened"] == []
    assert audit["all_gates_held"] is True and audit["independent_summary_recomputation_exact"] is True
    rows = [row for row in json.loads(ROWS.read_text())["rows"]
            if row["split"] in SPLITS and row["family_id"] in TARGETS + CONTROLS]
    assert len(rows) == EXPECTED_PAIRS
    assert len({tuple(row[side]) for row in rows for side in ("base_ids", "donor_ids")}) == EXPECTED_SEQUENCES
    return rows, ceiling


@torch.no_grad()
def prefix_state(model: torch.nn.Module, tokens: torch.Tensor):
    x = F.rms_norm(model.transformer.wte(tokens), (D,))
    x0, v1 = x, None
    for layer in range(PATCH_LAYER):
        x, v1 = model.transformer.h[layer](x, v1, x0)
    return x, x0, v1


def tail_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    x0: torch.Tensor,
    v1: torch.Tensor,
    finals: torch.Tensor,
    replacement: torch.Tensor | None,
    *,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    captured: list[torch.Tensor] = []

    def hook(_module, args):
        value = args[0]
        arange = torch.arange(value.shape[0], device=value.device)
        if capture:
            captured.append(value[arange, finals, 8 * HEAD_D:9 * HEAD_D].detach().clone())
        if replacement is None:
            return None
        changed = value.clone()
        changed[arange, finals, 8 * HEAD_D:9 * HEAD_D] = replacement.to(value.dtype)
        return (changed,) + tuple(args[1:])

    handle = model.transformer.h[PATCH_LAYER].attn.c_proj.register_forward_pre_hook(hook)
    try:
        for layer in range(PATCH_LAYER, 18):
            x, v1 = model.transformer.h[layer](x, v1, x0)
        logits = (30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)).float()
    finally:
        handle.remove()
    if capture and len(captured) != 1:
        raise RuntimeError("L13H8 capture failed")
    return logits, captured[0] if capture else None


def build_sequences(rows: list[dict]):
    length = max(len(row[side]) for row in rows for side in ("base_ids", "donor_ids"))
    sequences, finals, lookup = [], [], {}
    for row in rows:
        for side in ("base", "donor"):
            ids = row[f"{side}_ids"]
            key = (row["row_id"], side)
            assert key not in lookup
            lookup[key] = len(sequences)
            sequences.append(ids + [50256] * (length - len(ids)))
            finals.append(len(ids) - 1)
    assert len(sequences) == EXPECTED_SEQUENCES
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(finals, dtype=torch.long), lookup


@torch.no_grad()
def collect_states(model: torch.nn.Module, rows: list[dict]) -> tuple[dict, int]:
    tokens, finals_cpu, lookup = build_sequences(rows)
    xs, x0s, v1s, heads, native = [], [], [], [], []
    calls = 0
    for start in range(0, len(tokens), STATE_BATCH):
        batch = tokens[start:start + STATE_BATCH].cuda()
        finals = finals_cpu[start:start + STATE_BATCH].cuda()
        x, x0, v1 = prefix_state(model, batch)
        logits, head = tail_logits(model, x, x0, v1, finals, None, capture=True)
        arange = torch.arange(len(batch), device="cuda")
        xs.append(x.detach())
        x0s.append(x0.detach())
        v1s.append(v1.detach())
        heads.append(head)
        native.append(logits[arange, finals].detach())
        calls += 1
    return {
        "x": torch.cat(xs), "x0": torch.cat(x0s), "v1": torch.cat(v1s),
        "head": torch.cat(heads), "native": torch.cat(native),
        "finals": finals_cpu.cuda(), "lookup": lookup,
    }, calls


def ceiling_maps(rows: list[dict], ceiling: dict) -> tuple[dict, dict]:
    target, control = {}, {}
    raw = ceiling["raw_site_effects"]
    for split in SPLITS:
        for family in TARGETS + CONTROLS:
            expected_rows = [row for row in rows if row["split"] == split and row["family_id"] == family]
            expected_ids = {row["row_id"] for row in expected_rows}
            for direction in ("base_to_donor", "donor_to_base"):
                values = raw[split][family][direction]
                assert {item["row_id"] for item in values} == expected_ids
                for item in values:
                    key = (item["row_id"], direction)
                    if family in TARGETS:
                        assert item["endpoint_change"] > 0
                        target[key] = float(item["endpoint_change"])
                    else:
                        control[key] = {
                            "endpoint_change": float(item["endpoint_change"]),
                            "full_logit_rms": float(item["full_logit_rms"]),
                        }
    return target, control


def build_samples(rows: list[dict], states: dict, target_map: dict, control_map: dict):
    targets, controls = [], []
    for row in rows:
        for direction, target_side, source_side in (
            ("base_to_donor", "base", "donor"),
            ("donor_to_base", "donor", "base"),
        ):
            sample = {
                "row_id": row["row_id"], "group_id": row["group_id"], "split": row["split"],
                "family": row["family_id"], "direction": direction,
                "target_index": states["lookup"][(row["row_id"], target_side)],
                "source_index": states["lookup"][(row["row_id"], source_side)],
            }
            if row["family_id"] in TARGETS:
                sample.update({
                    "source_answer": row[f"{source_side}_answer_id"],
                    "target_answer": row[f"{target_side}_answer_id"],
                    "full_effect": target_map[(row["row_id"], direction)],
                })
                targets.append(sample)
            else:
                assert row["base_answer_id"] == row["donor_answer_id"]
                sample.update({
                    "answer": row["base_answer_id"],
                    "full_endpoint_effect": control_map[(row["row_id"], direction)]["endpoint_change"],
                    "full_logit_rms": control_map[(row["row_id"], direction)]["full_logit_rms"],
                })
                controls.append(sample)
    return targets, controls


def patch_batch(model: torch.nn.Module, states: dict, samples: list[dict], Q: torch.Tensor) -> torch.Tensor:
    target_indices = torch.tensor([sample["target_index"] for sample in samples], device="cuda")
    source_indices = torch.tensor([sample["source_index"] for sample in samples], device="cuda")
    finals = states["finals"][target_indices]
    target_head = states["head"][target_indices]
    difference = states["head"][source_indices] - target_head
    replacement = target_head + (difference @ Q) @ Q.T
    logits, _ = tail_logits(
        model, states["x"][target_indices], states["x0"][target_indices], states["v1"][target_indices],
        finals, replacement,
    )
    return logits[torch.arange(len(samples), device="cuda"), finals]


def target_recovery(logits: torch.Tensor, states: dict, samples: list[dict]) -> torch.Tensor:
    ti = torch.tensor([sample["target_index"] for sample in samples], device="cuda")
    source = torch.tensor([sample["source_answer"] for sample in samples], device="cuda")
    target = torch.tensor([sample["target_answer"] for sample in samples], device="cuda")
    arange = torch.arange(len(samples), device="cuda")
    native = states["native"][ti]
    movement = ((logits[arange, source] - logits[arange, target])
                - (native[arange, source] - native[arange, target]))
    full = torch.tensor([sample["full_effect"] for sample in samples], device="cuda")
    return movement / full


def closer_margin(logits: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    closers = torch.tensor([1, 8, 60], device=logits.device)
    picked = logits[:, closers]
    answer_columns = (closers[None, :] == answers[:, None]).nonzero(as_tuple=False)[:, 1]
    arange = torch.arange(len(logits), device=logits.device)
    answer_values = picked[arange, answer_columns]
    return answer_values - (picked.sum(1) - answer_values) / 2


def train_projector(model: torch.nn.Module, states: dict, targets: list[dict], controls: list[dict], rank: int, seed: int):
    generator = torch.Generator().manual_seed(10_000 * rank + seed)
    P = (torch.randn(HEAD_D, rank, generator=generator) / math.sqrt(HEAD_D)).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([P], lr=LR)
    for _ in range(STEPS):
        target_batch = [targets[index] for index in torch.randint(len(targets), (BATCH // 2,), generator=generator).tolist()]
        control_batch = [controls[index] for index in torch.randint(len(controls), (BATCH // 2,), generator=generator).tolist()]
        samples = target_batch + control_batch
        Q = torch.linalg.qr(P, mode="reduced")[0]
        logits = patch_batch(model, states, samples, Q)
        recovery = target_recovery(logits[:BATCH // 2], states, target_batch)
        ci = torch.tensor([sample["target_index"] for sample in control_batch], device="cuda")
        native_control = states["native"][ci]
        denominator = torch.tensor([
            max(sample["full_logit_rms"], CONTROL_RMS_FLOOR) for sample in control_batch
        ], device="cuda")
        control_loss = (logits[BATCH // 2:] - native_control).square().mean(-1) / denominator.square()
        loss = (recovery - 1).square().mean() + control_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.linalg.qr(P.detach(), mode="reduced")[0]


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


@torch.no_grad()
def evaluate_targets(model, states, samples, Q):
    values = []
    for start in range(0, len(samples), BATCH):
        batch = samples[start:start + BATCH]
        values.extend(target_recovery(patch_batch(model, states, batch, Q), states, batch).cpu().tolist())
    return values, math.ceil(len(samples) / BATCH)


@torch.no_grad()
def evaluate_controls(model, states, samples, Q):
    endpoint, rms = [], []
    for start in range(0, len(samples), BATCH):
        batch = samples[start:start + BATCH]
        logits = patch_batch(model, states, batch, Q)
        ti = torch.tensor([sample["target_index"] for sample in batch], device="cuda")
        native = states["native"][ti]
        answers = torch.tensor([sample["answer"] for sample in batch], device="cuda")
        endpoint.extend((closer_margin(logits, answers) - closer_margin(native, answers)).cpu().tolist())
        rms.extend((logits - native).square().mean(-1).sqrt().cpu().tolist())
    return endpoint, rms, math.ceil(len(samples) / BATCH)


def score_projector(model, states, targets, controls, Q, seed):
    report = {"targets": {}, "controls": {}}
    passed, calls = True, 0
    for family in TARGETS:
        report["targets"][family] = {}
        for direction in ("base_to_donor", "donor_to_base"):
            cell = [sample for sample in targets if sample["split"] == "SELECT"
                    and sample["family"] == family and sample["direction"] == direction]
            values, used = evaluate_targets(model, states, cell, Q)
            calls += used
            summary = {
                "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                "bootstrap95_lower_mean": bootstrap_lower(values, seed),
                "positive_fraction": float(np.mean(np.asarray(values) > 0)), "values": values,
            }
            seed += 1
            summary["passed"] = bool(summary["median"] >= .5 and summary["bootstrap95_lower_mean"] > 0
                                     and summary["positive_fraction"] >= .75)
            passed &= summary["passed"]
            report["targets"][family][direction] = summary
    for family in CONTROLS:
        report["controls"][family] = {}
        for direction in ("base_to_donor", "donor_to_base"):
            cell = [sample for sample in controls if sample["split"] == "SELECT"
                    and sample["family"] == family and sample["direction"] == direction]
            endpoint, rms, used = evaluate_controls(model, states, cell, Q)
            calls += used
            full_endpoint = [sample["full_endpoint_effect"] for sample in cell]
            full_rms = [sample["full_logit_rms"] for sample in cell]
            mean_abs = float(np.mean(np.abs(endpoint)))
            endpoint_ratio = mean_abs / float(np.mean(np.abs(full_endpoint)))
            rms_ratio = float(np.mean(rms)) / float(np.mean(full_rms))
            summary = {
                "n": len(endpoint), "mean_absolute_closer_margin_change": mean_abs,
                "fraction_of_complete_head_margin_change": endpoint_ratio,
                "mean_full_vocabulary_logit_rms": float(np.mean(rms)),
                "fraction_of_complete_head_full_vocabulary_rms": rms_ratio,
                "endpoint_values": endpoint, "full_vocabulary_logit_rms_values": rms,
            }
            summary["passed"] = bool(mean_abs <= .10 and endpoint_ratio <= .25 and rms_ratio <= .25)
            passed &= summary["passed"]
            report["controls"][family][direction] = summary
    report["passed"] = bool(passed)
    return report, calls


def main() -> None:
    started = time.time()
    rows, ceiling = load_authority()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dryrun_passed", "pairs": len(rows), "unique_sequences": EXPECTED_SEQUENCES,
            "ranks": list(RANKS), "seeds": list(SEEDS), "fits": len(RANKS) * len(SEEDS),
            "gradient_suffix_evaluations": EXPECTED_GRAD_SUFFIX_EVALS,
            "maximum_scoring_suffix_evaluations": 700, "final_or_ood_opened": False,
        }, indent=2))
        return

    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    states, native_calls = collect_states(model, rows)
    target_map, control_map = ceiling_maps(rows, ceiling)
    target_samples, control_samples = build_samples(rows, states, target_map, control_map)
    fit_targets = [sample for sample in target_samples if sample["split"] == "FIT"]
    fit_controls = [sample for sample in control_samples if sample["split"] == "FIT"]
    reports, projectors, score_calls, score_seed = {}, {}, 0, BOOTSTRAP_SEED
    for rank in RANKS:
        reports[str(rank)] = {}
        for seed in SEEDS:
            Q = train_projector(model, states, fit_targets, fit_controls, rank, seed)
            report, calls = score_projector(model, states, target_samples, control_samples, Q, score_seed)
            score_seed += 20
            score_calls += calls
            reports[str(rank)][str(seed)] = report
            projectors[(rank, seed)] = Q.cpu()

    random_controls, random_calls = {}, 0
    select_targets = [sample for sample in target_samples if sample["split"] == "SELECT"]
    for rank in RANKS:
        values = []
        for random_seed in RANDOM_SEEDS:
            generator = torch.Generator().manual_seed(10_000 * rank + random_seed)
            Q = torch.linalg.qr(torch.randn(HEAD_D, rank, generator=generator).cuda(), mode="reduced")[0]
            recovery, calls = evaluate_targets(model, states, select_targets, Q)
            random_calls += calls
            values.append(float(np.mean(recovery)))
        random_controls[str(rank)] = {
            "mean_target_recovery_by_seed": values,
            "mean_across_seeds": float(np.mean(values)),
            "passed": bool(float(np.mean(values)) < .10),
        }

    eligible = []
    for rank in RANKS:
        stable = sum(reports[str(rank)][str(seed)]["passed"] for seed in SEEDS) >= 2
        reports[str(rank)]["seed_stable"] = bool(stable)
        reports[str(rank)]["rank_eligible"] = bool(stable and random_controls[str(rank)]["passed"])
        if reports[str(rank)]["rank_eligible"]:
            eligible.append(rank)
    selected_rank = min(eligible) if eligible else None
    selected_seeds = ([seed for seed in SEEDS if reports[str(selected_rank)][str(seed)]["passed"]]
                      if selected_rank is not None else [])
    score_total = score_calls + random_calls
    instrument = bool(
        native_calls == EXPECTED_NATIVE_FORWARDS
        and score_total == EXPECTED_SCORE_SUFFIX_EVALS
        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
    )
    torch.save({"projectors": projectors, "selected_rank": selected_rank,
                "selected_seeds": selected_seeds}, BUNDLE)
    result = {
        "rung": 556, "stage": "pending_opener_target_plus_invariance_das_l13h8",
        "pred_a_exact_instrument": instrument,
        "pred_b_stable_selective_projector_exists": selected_rank is not None,
        "pred_c_random_subspaces_below_bar": all(item["passed"] for item in random_controls.values()),
        "strong_null": selected_rank is None,
        "selected_rank": selected_rank, "selected_seeds": selected_seeds,
        "fits": reports, "random_controls": random_controls,
        "native_model_forwards": native_calls,
        "gradient_suffix_evaluations": EXPECTED_GRAD_SUFFIX_EVALS,
        "no_gradient_suffix_evaluations": score_total,
        "model_forwards": native_calls + EXPECTED_GRAD_SUFFIX_EVALS + score_total,
        "model_backwards": EXPECTED_GRAD_SUFFIX_EVALS, "model_weights_updated": False,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "bundle_path": str(BUNDLE.relative_to(ROOT.parent.parent)), "bundle_sha256": sha256(BUNDLE),
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "freeze_selected_projector_then_open_FINAL_TEST_and_OOD_once"
            if selected_rank is not None else "record_linear_site_null_and_test_nonlinear_or_earlier_representation"
        ),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({
        key: value for key, value in result.items()
        if key.startswith("pred_") or key in {"strong_null", "selected_rank", "selected_seeds", "next_step"}
    }, indent=2))


if __name__ == "__main__":
    main()
