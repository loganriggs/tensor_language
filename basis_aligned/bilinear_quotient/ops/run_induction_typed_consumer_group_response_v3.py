#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT
"""A instrument; B confirmed typed multi-module mediator; C compact group."""
import hashlib
import json
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(POLY), str(OPS), str(ROOT)]
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
from circuit_fast_screen_managed_runner import atomic_create_json
from fp32_add_observation_v1 import observe
import induction_centered_fixed_geometry_rung594 as p
from induction_centered_fixed_geometry_rung594_runtime import R594ModelExecutor
import induction_live_clamp_v1 as clamp


ROWS = POLY / "INDUCTION_CONTEXTUAL_CONSUMER_RESPONSE_V1_ROWS.json"
BINDING = POLY / "INDUCTION_TYPED_CONSUMER_GROUP_RESPONSE_V3_BINDING.json"
OUT = POLY / "INDUCTION_TYPED_CONSUMER_GROUP_RESPONSE_V3_RESULT.json"
CANDIDATE_GROUPS = {
    "all_mlp": tuple(f"mlp{x}" for x in range(8, 18)),
    "all_attention": tuple(f"attn{x}" for x in range(9, 18)),
    "early_all": ("mlp8",) + tuple(name for x in range(9, 13) for name in (f"attn{x}", f"mlp{x}")),
    "late_all": tuple(name for x in range(13, 18) for name in (f"attn{x}", f"mlp{x}")),
    "early_mlp": tuple(f"mlp{x}" for x in range(8, 13)),
    "late_mlp": tuple(f"mlp{x}" for x in range(13, 18)),
    "early_attention": tuple(f"attn{x}" for x in range(9, 13)),
    "late_attention": tuple(f"attn{x}" for x in range(13, 18)),
}
CANDIDATES = tuple(CANDIDATE_GROUPS)
BATCH = 32


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lse(x):
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def module_index(name):
    kind = 0 if name.startswith("attn") else 1
    return 2 * int(name[4:] if kind == 0 else name[3:]) + kind


def load_rows():
    frozen = json.loads(ROWS.read_text())
    r585, authority = p.load_authority()
    by_id = {row["directed_id"]: row for row in authority["directions"]}
    rows = []
    for saved in frozen["rows"]:
        current = by_id[saved["directed_id"]]
        assert current == saved
        rows.append(current)
    endpoint_by_id = {row["endpoint_id"]: row for row in authority["endpoints"]}
    endpoints = [endpoint_by_id[key] for key in frozen["endpoint_ids"]]
    assert len(rows) == 96 and len(endpoints) == 96
    assert frozen["authority_direction_manifest_sha256"] == authority["direction_manifest_sha256"]
    return r585, frozen, rows, endpoints


def capture_native(runtime, endpoints):
    torch = runtime.torch
    device = next(runtime.model.parameters()).device
    all_logits, all_terms, all_writes = [], [], []
    for start in range(0, len(endpoints), BATCH):
        specs = endpoints[start:start + BATCH]
        tokens = p.fixed_tokens(specs, "token_ids")
        token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
        terms = np.zeros((len(specs), 4, p.RESIDUAL), dtype=np.float32)
        writes = np.zeros((len(specs), 36, p.RESIDUAL), dtype=np.float32)

        def attention(event):
            if event.site in (5, 7, 8):
                write, first, captured, _ = runtime.r585.factorize_attention_event(
                    event, specs, torch=torch, functional=runtime.functional,
                    induction=runtime.induction,
                )
                for local, row in enumerate(captured):
                    for index, site in enumerate(p.SITES):
                        if int(site[1]) == event.site:
                            terms[local, index] = row[site]["term"].numpy()
            else:
                write, first = event.block.attn(event.state, event.first_value)
            for local, spec in enumerate(specs):
                writes[local, 2 * event.site] = write[local, int(spec["final_position"])].float().detach().cpu().numpy()
            return write, first

        def mlp(event):
            write = event.block.mlp(event.state)
            for local, spec in enumerate(specs):
                writes[local, 2 * event.site + 1] = write[local, int(spec["final_position"])].float().detach().cpu().numpy()
            return write

        with torch.inference_mode():
            logits = runtime.facade.forward_with_dispatch(
                runtime.model, token_tensor, attention, mlp, require_production=False
            )
        all_logits.append(np.stack([
            logits[i, int(spec["final_position"])].float().cpu().numpy()
            for i, spec in enumerate(specs)
        ]))
        all_terms.append(terms)
        all_writes.append(writes)
    return np.concatenate(all_logits), np.concatenate(all_terms), np.concatenate(all_writes)


def execute(runtime, tokens, specs, target, native_writes, restore_names=()):
    torch = runtime.torch
    device = next(runtime.model.parameters()).device
    token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
    target_gpu = torch.as_tensor(target, dtype=torch.float32, device=device)
    before = np.zeros((len(specs), 3, p.RESIDUAL), dtype=np.float32)
    total = np.zeros_like(before)
    after = np.zeros_like(before)
    active = np.zeros(len(specs), dtype=bool)
    restored = np.zeros(len(specs), dtype=np.float64)
    restore_indices = {module_index(name) for name in restore_names}

    def attention(event):
        if event.site in (5, 7, 8):
            write, first, captured, _ = runtime.r585.factorize_attention_event(
                event, specs, torch=torch, functional=runtime.functional,
                induction=runtime.induction,
            )
            changed = write.clone()
            indices = [i for i, site in enumerate(p.SITES) if int(site[1]) == event.site]
            li = (5, 7, 8).index(event.site)
            for local, spec in enumerate(specs):
                query = int(spec["final_position"])
                pieces = []
                for index in indices:
                    current = captured[local][p.SITES[index]]["term"].numpy()
                    pieces.append(np.subtract(target[local, index], current, dtype=np.float32))
                delta = torch.as_tensor(np.stack(pieces), device=device).sum(dim=0)
                before[local, li] = changed[local, query].float().detach().cpu().numpy()
                total[local, li] = delta.float().detach().cpu().numpy()
                changed[local, query] += delta.to(changed.dtype)
                after[local, li] = changed[local, query].float().detach().cpu().numpy()
                active[local] |= bool(delta.float().norm().detach().cpu() > 1e-6)
            write = changed
        else:
            write, first = event.block.attn(event.state, event.first_value)
        index = 2 * event.site
        if index in restore_indices:
            write = write.clone()
            for local, spec in enumerate(specs):
                query = int(spec["final_position"])
                old = write[local, query].float()
                new = torch.as_tensor(native_writes[local, index], device=device)
                restored[local] += float((new - old).double().norm().detach().cpu()) ** 2
                write[local, query] = new.to(write.dtype)
        return write, first

    def mlp(event):
        write = event.block.mlp(event.state)
        index = 2 * event.site + 1
        if index in restore_indices:
            write = write.clone()
            for local, spec in enumerate(specs):
                query = int(spec["final_position"])
                old = write[local, query].float()
                new = torch.as_tensor(native_writes[local, index], device=device)
                restored[local] += float((new - old).double().norm().detach().cpu()) ** 2
                write[local, query] = new.to(write.dtype)
        return write

    with torch.inference_mode():
        logits = runtime.facade.forward_with_dispatch(
            runtime.model, token_tensor, attention, mlp, require_production=False
        )
    selected = np.stack([
        logits[i, int(spec["final_position"])].float().cpu().numpy()
        for i, spec in enumerate(specs)
    ])
    observation = observe(before, total, after)
    assert observation["passed"]
    return selected, active, np.sqrt(restored), observation


def metrics(native, arm, rows):
    out = []
    for z, a, row in zip(native, arm, rows):
        z = z.astype(np.float64); a = a.astype(np.float64)
        answer = int(row["recipient_answer_id"])
        other = int(row["recipient_other_answer_id"])
        out.append({
            "directed_id": row["directed_id"],
            "group_id": row["group_id"],
            "cell": "|".join(str(row[k]) for k in ("family", "variant", "recipient_condition", "direction")),
            "ce_damage": lse(a) - a[answer] - lse(z) + z[answer],
            "correct": bool(a[answer] > a[other]),
            "margin_change": float((a[answer] - a[other]) - (z[answer] - z[other])),
            "vocab_rms": float(np.sqrt(np.mean((a - z) ** 2))),
        })
    return out


def cell_report(items, groups):
    chosen = [row for row in items if row["group_id"] in groups]
    by_cell = defaultdict(list)
    for row in chosen:
        by_cell[row["cell"]].append(row)
    assert len(by_cell) == 4 and all(len(value) == 12 for value in by_cell.values())
    return {
        cell: {
            "groups": 12,
            "mean_ce_damage": float(np.mean([row["ce_damage"] for row in values])),
            "correct_fraction": float(np.mean([row["correct"] for row in values])),
            "median_vocab_rms": float(np.median([row["vocab_rms"] for row in values])),
        }
        for cell, values in sorted(by_cell.items())
    }


def passes(report, joint_report):
    reductions = {}
    for cell, value in report.items():
        denominator = joint_report[cell]["median_vocab_rms"]
        reductions[cell] = 1.0 - value["median_vocab_rms"] / denominator
    answer = all(v["mean_ce_damage"] <= 0.10 and v["correct_fraction"] >= 0.75 for v in report.values())
    return answer and all(v >= 0.25 for v in reductions.values()), reductions


def main():
    binding = json.loads(BINDING.read_text())
    assert all(digest(Path(path)) == expected for path, expected in binding["files"].items())
    r585, frozen, rows, endpoints = load_rows()
    assert len(CANDIDATES) == 8 and binding["price"] == {"forwards": 33, "sequences": 1056}
    if REQUESTED_DRY:
        print(json.dumps({
            "dryrun": True, "model_loaded": False, "gpu_accessed": False,
            "rows": len(rows), "endpoints": len(endpoints), "candidates": list(CANDIDATES),
            "price": binding["price"], "controls": clamp.controls(),
        }))
        return
    assert not OUT.exists()
    signal.alarm(600)
    runtime = R594ModelExecutor(p, r585)
    runtime.torch.set_num_threads(2)
    counts = [0, 0]
    def count_forward(_module, args, _output):
        counts[0] += 1
        counts[1] += len(args[0])

    handle = runtime.model.transformer.h[0].attn.register_forward_hook(count_forward)
    tic = time.perf_counter()
    try:
        endpoint_logits, endpoint_terms, endpoint_writes = capture_native(runtime, endpoints)
        endpoint_index = {row["endpoint_id"]: i for i, row in enumerate(endpoints)}
        recipient_index = np.array([endpoint_index[row["recipient_endpoint_id"]] for row in rows])
        donor_index = np.array([endpoint_index[row["donor_endpoint_id"]] for row in rows])
        specs = [endpoints[index] for index in recipient_index]
        tokens = p.fixed_tokens(specs, "token_ids")
        native = endpoint_logits[recipient_index]
        recipient_terms = endpoint_terms[recipient_index]
        donor_terms = endpoint_terms[donor_index]
        native_writes = endpoint_writes[recipient_index]
        arms = {}
        active = {}
        restored = {}
        observations = []
        for name, target, candidate in (
            [("self", recipient_terms, None), ("joint", donor_terms, None)]
            + [(name, donor_terms, name) for name in CANDIDATES]
        ):
            batches = []
            arm_active = []
            arm_restored = []
            for start in range(0, len(rows), BATCH):
                stop = start + BATCH
                logits, act, response, observation = execute(
                    runtime, tokens[start:stop], specs[start:stop], target[start:stop],
                    native_writes[start:stop], CANDIDATE_GROUPS.get(candidate, ()),
                )
                batches.append(logits); arm_active.append(act); arm_restored.append(response)
                observations.append(observation)
            arms[name] = np.concatenate(batches)
            active[name] = np.concatenate(arm_active)
            restored[name] = np.concatenate(arm_restored)
    finally:
        handle.remove()

    maximum_abs = float(np.max(np.abs(arms["self"].astype(np.float64) - native.astype(np.float64))))
    relative = float(np.linalg.norm(arms["self"].astype(np.float64) - native.astype(np.float64)) /
                     max(np.linalg.norm(native.astype(np.float64)), 1e-30))
    arm_metrics = {name: metrics(native, values, rows) for name, values in arms.items() if name != "self"}
    discovery = set(frozen["discovery_group_ids"]); confirm = set(frozen["confirm_group_ids"])
    discovery_reports = {name: cell_report(items, discovery) for name, items in arm_metrics.items()}
    confirm_reports = {name: cell_report(items, confirm) for name, items in arm_metrics.items()}
    eligible = []
    discovery_reductions = {}
    for name in CANDIDATES:
        passed, reductions = passes(discovery_reports[name], discovery_reports["joint"])
        discovery_reductions[name] = reductions
        if passed:
            eligible.append((len(CANDIDATE_GROUPS[name]), -min(reductions.values()), CANDIDATES.index(name), name))
    selected = min(eligible)[3] if eligible else None
    confirm_pass = False; confirm_reductions = None
    if selected is not None:
        confirm_pass, confirm_reductions = passes(confirm_reports[selected], confirm_reports["joint"])
    neighbors = []
    neighbor_passes = {}
    localized = bool(confirm_pass and selected is not None and len(CANDIDATE_GROUPS[selected]) <= 5)
    instrument = (
        counts == [33, 1056]
        and maximum_abs <= 1e-3 and relative <= 1e-5
        and bool(np.mean(active["joint"]) >= 0.75)
        and all(bool(np.mean(restored[name] > 1e-6) >= 0.75) for name in CANDIDATES)
        and all(observation["passed"] for observation in observations)
    )
    result = {
        "experiment": "induction_typed_consumer_group_response_v3",
        "terminal": "complete" if instrument else "invalid_instrument",
        "predictions": {
            "pred_a_instrument": bool(instrument),
            "pred_b_confirm_typed_group": bool(confirm_pass),
            "pred_c_compact_group": localized,
        },
        "selected_candidate": selected,
        "neighbor_candidates": neighbors,
        "neighbor_confirm_passes": neighbor_passes,
        "instrument": {
            "self_native_max_abs": maximum_abs,
            "self_native_relative": relative,
            "joint_active_fraction": float(np.mean(active["joint"])),
            "minimum_candidate_response_active_fraction": float(min(np.mean(restored[name] > 1e-6) for name in CANDIDATES)),
            "maximum_rounding_residual": float(max(o["maximum_rounding_residual"] for o in observations)),
        },
        "discovery": {
            "joint": discovery_reports["joint"],
            "eligible_candidates": [item[3] for item in sorted(eligible)],
            "fractional_vocab_rms_reduction": discovery_reductions,
            "selected": None if selected is None else discovery_reports[selected],
        },
        "confirm": {
            "joint": confirm_reports["joint"],
            "selected": None if selected is None else confirm_reports[selected],
            "fractional_vocab_rms_reduction": confirm_reductions,
        },
        "price": {"forwards": counts[0], "sequences": counts[1], "fits": 0, "backwards": 0, "weight_updates": 0},
        "claim_boundary": "Native-oracle typed multi-module response restoration on frozen FIT rows; no extracted consumer, SELECT/FINAL/OOD, or execution compression claim.",
        "rows_sha256": digest(ROWS), "binding_sha256": digest(BINDING),
        "runner_sha256": digest(RUNNER), "checkpoint_sha256": runtime.checkpoint_sha256,
        "wall_seconds": time.perf_counter() - tic,
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
