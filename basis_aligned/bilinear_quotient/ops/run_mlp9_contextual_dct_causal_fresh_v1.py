#!/usr/bin/env python3
# BQGATE:64fresh prefixes;896suffix states;180seconds;frozen install/remove test.
"""Fresh causal install/remove test of the extracted contextual DCT node."""
import importlib.util
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
PACKAGE = P / "extracted_circuits/mlp9_contextual_dct_node_v1"
EXECUTOR = PACKAGE / "execute.py"
WEIGHTS = PACKAGE / "weights_rank16_v2.pt"
sys.path[:0] = [str(HERE), str(P), str(ROOT)]

import torch
import torch.nn.functional as F

import live_crossfirst_prefix_v1 as live
import run_crossfirst_per_layer_causal_hessian_v1 as allocation
import run_mlp9_contextual_dct_downstream_reader_discovery_v1 as reader_v1
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ROWS = P / f"{STEM}_ROWS.json"
NODE_RESULT = P / "MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_RESULT.json"
SCALE_RESULT = P / "MLP9_CONTEXTUAL_DCT_FINITE_SCALE_DISCOVERY_V1_RESULT.json"
SCALE = 32.0
DIRECTION = 2
TARGET = (21215, 6165)
CONTROLS = ((23482, 23095), (44178, 36538), (30724, 30175))
RANDOM_CONTROLS = 8


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def report(expected, predicted):
    expected = torch.cat([torch.as_tensor(value).double().reshape(-1) for value in expected])
    predicted = torch.cat([torch.as_tensor(value).double().reshape(-1) for value in predicted])
    return {
        "relative_l2": float((predicted - expected).norm() / expected.norm().clamp_min(1e-30)),
        "cosine": float((expected * predicted).sum() / (expected.norm() * predicted.norm()).clamp_min(1e-30)),
        "expected_mean_absolute": float(expected.abs().mean()),
        "predicted_mean_absolute": float(predicted.abs().mean()),
        "sign_agreement": float((torch.sign(expected) == torch.sign(predicted)).double().mean()),
    }


def norm_ratio(numerator, denominator):
    numerator = torch.cat([torch.as_tensor(value).double().reshape(-1) for value in numerator])
    denominator = torch.cat([torch.as_tensor(value).double().reshape(-1) for value in denominator])
    return float(numerator.norm() / denominator.norm().clamp_min(1e-30))


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    rows = json.loads(ROWS.read_text())["rows"]
    scale = json.loads(SCALE_RESULT.read_text())
    node = json.loads(NODE_RESULT.read_text())
    if len(rows) != 64 or scale["selected_scale"] != SCALE or not all(node["predictions"].values()):
        raise ValueError("frozen program changed")
    return rows, node, torch.load(WEIGHTS, weights_only=True)


def plan():
    rows, node, program = load_bound()
    return {"schema": "mlp9_contextual_dct_causal_fresh_v1_plan", "fresh_prefixes": len(rows), "families": 4, "scale": SCALE, "direction": DIRECTION, "suffix_states": len(rows) * (6 + RANDOM_CONTROLS), "random_controls": RANDOM_CONTROLS, "rank": int(program["selected_rank"]), "prior_predictions_bound": all(node["predictions"].values()), "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    rows, node_result, cpu_program = load_bound()
    program = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu_program.items()}
    spec = importlib.util.spec_from_file_location("mlp9_contextual_dct_execute_causal", EXECUTOR)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    live_weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = program["directions"][DIRECTION].float()
    generator = torch.Generator().manual_seed(2026091644)
    records = []
    state_replay = []
    suffix_replay = []
    started = time.perf_counter()

    for row_index, row in enumerate(rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, live_weights)
        _, stages = allocation.make_stages(model, native)
        stage = stages[0]
        h0 = stage(native["z9"]).detach()
        tangent = SCALE * direction[None, None, :]
        h1 = stage(native["z9"] + tangent).detach()
        h2 = stage(native["z9"] + 2 * tangent).detach()
        additive = 2 * h1 - h0
        exact = h2 - additive
        predicted = SCALE * SCALE * executor.execute(native["z9"], program)[DIRECTION, DIRECTION].detach().float()
        randoms = []
        for _ in range(RANDOM_CONTROLS):
            random = torch.randn(predicted.shape, generator=generator, dtype=torch.float32).to("cuda")
            randoms.append(random / random.norm().clamp_min(1e-30) * predicted.norm())
        states = torch.cat((h0, h1, h2, additive, additive + predicted, h2 - predicted, *[h2 - value for value in randoms]), dim=0)
        logits = reader_v1.batched_suffix_logits(model, native, states).detach().float()[:, :50257]
        native_mixed = logits[2] - 2 * logits[1] + logits[0]
        exact_local = logits[2] - logits[3]
        installed = logits[4] - logits[3]
        remaining = logits[5] - 2 * logits[1] + logits[0]
        removed = logits[2] - logits[5]
        random_removed = [logits[2] - logits[6 + index] for index in range(RANDOM_CONTROLS)]
        state_replay.append(float((h0 - native["h9"]).norm() / native["h9"].norm().clamp_min(1e-30)))
        direct_state = h0
        for downstream in stages[1:]:
            direct_state = downstream(direct_state)
        direct_last = F.rms_norm(direct_state[:, -1], (1152,))
        direct_logits = (30 * torch.tanh(model.lm_head(direct_last) / 30))[0, :50257]
        suffix_replay.append(float((direct_logits - logits[0]).norm() / direct_logits.norm().clamp_min(1e-30)))
        def contrast(value, pair):
            return value[pair[0]] - value[pair[1]]
        records.append({"family": row["family"], "exact_state": exact.cpu(), "predicted_state": predicted.cpu(), "native_mixed": native_mixed.cpu(), "exact_local": exact_local.cpu(), "installed": installed.cpu(), "remaining": remaining.cpu(), "removed": removed.cpu(), "random_removed": [value.cpu() for value in random_removed], "target_native_mixed": float(contrast(native_mixed, TARGET)), "target_exact_local": float(contrast(exact_local, TARGET)), "target_remaining": float(contrast(remaining, TARGET)), "target_removed": float(contrast(removed, TARGET)), "target_random_removed": [float(contrast(value, TARGET)) for value in random_removed], "collateral_removed": [float(contrast(removed, pair)) for pair in CONTROLS]})

    def subset(family=None):
        return [record for record in records if family is None or record["family"] == family]
    state_overall = report([r["exact_state"] for r in records], [r["predicted_state"] for r in records])
    state_families = [report([r["exact_state"] for r in subset(f)], [r["predicted_state"] for r in subset(f)]) for f in range(4)]
    install_overall = report([r["exact_local"] for r in records], [r["installed"] for r in records])
    install_families = [report([r["exact_local"] for r in subset(f)], [r["installed"] for r in subset(f)]) for f in range(4)]
    removal_ratio = norm_ratio([r["remaining"] for r in records], [r["native_mixed"] for r in records])
    removal_families = [norm_ratio([r["remaining"] for r in subset(f)], [r["native_mixed"] for r in subset(f)]) for f in range(4)]
    target_removal_ratio = norm_ratio([r["target_remaining"] for r in records], [r["target_native_mixed"] for r in records])
    target_family_ratios = [norm_ratio([r["target_remaining"] for r in subset(f)], [r["target_native_mixed"] for r in subset(f)]) for f in range(4)]
    target_family_signs = [float(torch.sign(torch.tensor([r["target_removed"] for r in subset(f)]).mean()) == torch.sign(torch.tensor([r["target_exact_local"] for r in subset(f)]).mean())) for f in range(4)]
    random_ratios = [norm_ratio([r["target_random_removed"][i] for r in records], [r["target_removed"] for r in records]) for i in range(RANDOM_CONTROLS)]
    random_median = float(torch.tensor(random_ratios).median())
    collateral_ratios = [norm_ratio([r["collateral_removed"][i] for r in records], [r["target_removed"] for r in records]) for i in range(len(CONTROLS))]
    pred_a = bool(max(state_replay) <= 2e-6 and max(suffix_replay) <= 2e-6)
    pred_b = bool(pred_a and state_overall["relative_l2"] <= .02 and state_overall["cosine"] >= .999 and max(value["relative_l2"] for value in state_families) <= .02)
    pred_c = bool(pred_a and install_overall["relative_l2"] <= .03 and install_overall["cosine"] >= .999 and max(value["relative_l2"] for value in install_families) <= .05)
    pred_d = bool(pred_a and removal_ratio <= .10 and max(removal_families) <= .15 and target_removal_ratio <= .10 and max(target_family_ratios) <= .15 and min(target_family_signs) == 1.0)
    pred_e = bool(pred_a and random_median <= .25 and max(random_ratios) <= .50 and max(collateral_ratios) <= .50)
    pred_f = bool(node_result["package"]["activation_ports"] == ["native pre-MLP9 z9"] and node_result["package"]["isolated_check"]["relative_l2"] <= 2e-6 and all(node_result["predictions"].values()))
    predictions = {"pred_a_instrument": pred_a, "pred_b_ood_local_prediction": pred_b, "pred_c_ood_causal_installation": pred_c, "pred_d_removal": pred_d, "pred_e_specificity_and_collateral": pred_e, "pred_f_extraction_and_reuse_bound": pred_f}
    terminal = "mlp9_contextual_dct_four_trait_synthetic_primitive" if all(predictions.values()) else "valid_mlp9_contextual_dct_causal_fresh_null" if pred_a else "invalid"
    result = {"schema": "mlp9_contextual_dct_causal_fresh_v1_result", "terminal": terminal, "predictions": predictions, "state_prediction_overall": state_overall, "state_prediction_families": state_families, "causal_installation_overall": install_overall, "causal_installation_families": install_families, "all_logit_remaining_mixed_ratio": removal_ratio, "all_logit_remaining_mixed_family_ratios": removal_families, "target_remaining_mixed_ratio": target_removal_ratio, "target_remaining_mixed_family_ratios": target_family_ratios, "target_family_aggregate_signs": target_family_signs, "matched_random_target_effect_ratios": random_ratios, "matched_random_target_effect_median_ratio": random_median, "collateral_over_target_removal_ratios": collateral_ratios, "maximum_state_replay_relative_l2": max(state_replay), "maximum_suffix_replay_relative_l2": max(suffix_replay), "frozen": {"scale": SCALE, "direction": DIRECTION, "target": TARGET, "controls": CONTROLS, "random_seed": 2026091644}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "fits": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Prospective fresh-context install/remove test of an induced MLP9 local interaction. This is a synthetic causal primitive with the exact native suffix live, not a natural-language behavior, naturally occurring feature removal, or complete sparse circuit."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "state": state_overall, "install": install_overall, "removal_ratio": removal_ratio, "removal_families": removal_families, "target_removal_ratio": target_removal_ratio, "target_families": target_family_ratios, "random_ratios": random_ratios, "collateral_ratios": collateral_ratios, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
