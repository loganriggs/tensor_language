#!/usr/bin/env python3
# BQGATE:64opened prefixes;21suffix states;180seconds;structure-matched specificity audit.
"""Adversarial specificity controls for the passing contextual DCT removal."""
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

STEM = "MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_SPECIFICITY_AUDIT"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ROWS = P / "MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_ROWS.json"
CAUSAL_RESULT = P / "MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_RESULT.json"
SCALE = 32.0
TARGET = (21215, 6165)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    rows = json.loads(ROWS.read_text())["rows"]
    causal = json.loads(CAUSAL_RESULT.read_text())
    if len(rows) != 64 or not all(causal["predictions"].values()):
        raise ValueError("causal receipt changed")
    return rows, torch.load(WEIGHTS, weights_only=True)


def plan():
    rows, program = load_bound()
    return {"schema": "mlp9_contextual_dct_causal_fresh_v1_specificity_audit_plan", "opened_prefixes": len(rows), "rank": int(program["selected_rank"]), "isotropic_controls": 8, "subspace_controls": 8, "alternate_pair_controls": 3, "suffix_states_per_prefix": 21, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def match_position_norm(candidate, target):
    target_norm = target.float().norm(dim=-1, keepdim=True)
    return candidate.float() / candidate.float().norm(dim=-1, keepdim=True).clamp_min(1e-30) * target_norm


def control_reports(target_effects, control_effects, target_logits, control_logits):
    target_effects = torch.tensor(target_effects, dtype=torch.float64)
    reports = []
    for index in range(len(control_effects[0])):
        effect = torch.tensor([row[index] for row in control_effects], dtype=torch.float64)
        target_flat = torch.cat([row.reshape(-1).double() for row in target_logits])
        control_flat = torch.cat([row[index].reshape(-1).double() for row in control_logits])
        reports.append({
            "target_effect_ratio": float(effect.norm() / target_effects.norm().clamp_min(1e-30)),
            "target_effect_cosine": float((effect * target_effects).sum() / (effect.norm() * target_effects.norm()).clamp_min(1e-30)),
            "all_logit_effect_ratio": float(control_flat.norm() / target_flat.norm().clamp_min(1e-30)),
            "all_logit_effect_cosine": float((control_flat * target_flat).sum() / (control_flat.norm() * target_flat.norm()).clamp_min(1e-30)),
        })
    return reports


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

    rows, cpu_program = load_bound()
    program = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu_program.items()}
    spec = importlib.util.spec_from_file_location("mlp9_contextual_dct_execute_audit", EXECUTOR)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    live_weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = program["directions"][2].float()
    basis = program["basis"].float()
    generator = torch.Generator().manual_seed(2026091645)
    target_effects, target_logits = [], []
    isotropic_effects, isotropic_logits = [], []
    subspace_effects, subspace_logits = [], []
    alternate_effects, alternate_logits = [], []
    state_replay = []
    suffix_replay = []
    started = time.perf_counter()

    for row in rows:
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, live_weights)
        _, stages = allocation.make_stages(model, native)
        stage = stages[0]
        h0 = stage(native["z9"]).detach()
        tangent = SCALE * direction[None, None, :]
        h1 = stage(native["z9"] + tangent).detach()
        h2 = stage(native["z9"] + 2 * tangent).detach()
        table = SCALE * SCALE * executor.execute(native["z9"], program).detach().float()
        target = table[2, 2]
        isotropic = []
        subspace = []
        for _ in range(8):
            candidate = torch.randn(target.shape, generator=generator, dtype=torch.float32).to("cuda")
            isotropic.append(match_position_norm(candidate, target))
            coefficients = torch.randn((*target.shape[:-1], basis.shape[1]), generator=generator, dtype=torch.float32).to("cuda")
            subspace.append(match_position_norm(coefficients @ basis.T, target))
        alternate = [match_position_norm(table[index, index], target) for index in (0, 1, 3)]
        removals = isotropic + subspace + alternate
        states = torch.cat((h2, h2 - target, *[h2 - value for value in removals]), dim=0)
        logits = reader_v1.batched_suffix_logits(model, native, states).detach().float()[:, :50257]
        target_logit = logits[0] - logits[1]
        controls = [logits[0] - logits[2 + index] for index in range(len(removals))]
        def contrast(value):
            return float(value[TARGET[0]] - value[TARGET[1]])
        target_effects.append(contrast(target_logit))
        target_logits.append(target_logit.cpu())
        isotropic_effects.append([contrast(value) for value in controls[:8]])
        isotropic_logits.append([value.cpu() for value in controls[:8]])
        subspace_effects.append([contrast(value) for value in controls[8:16]])
        subspace_logits.append([value.cpu() for value in controls[8:16]])
        alternate_effects.append([contrast(value) for value in controls[16:]])
        alternate_logits.append([value.cpu() for value in controls[16:]])
        state_replay.append(float((h0 - native["h9"]).norm() / native["h9"].norm().clamp_min(1e-30)))
        direct_state = h2
        for downstream in stages[1:]:
            direct_state = downstream(direct_state)
        direct_last = F.rms_norm(direct_state[:, -1], (1152,))
        direct_logits = (30 * torch.tanh(model.lm_head(direct_last) / 30))[0, :50257]
        suffix_replay.append(float((direct_logits - logits[0]).norm() / direct_logits.norm().clamp_min(1e-30)))

    isotropic_reports = control_reports(target_effects, isotropic_effects, target_logits, isotropic_logits)
    subspace_reports = control_reports(target_effects, subspace_effects, target_logits, subspace_logits)
    alternate_reports = control_reports(target_effects, alternate_effects, target_logits, alternate_logits)
    pred_a = bool(max(state_replay) <= 2e-6 and max(suffix_replay) <= 2e-6)
    pred_b = bool(pred_a and max(value["target_effect_ratio"] for value in isotropic_reports) <= .50)
    pred_c = bool(pred_a and max(value["target_effect_ratio"] for value in subspace_reports) <= .50)
    pred_d = bool(pred_a and max(value["target_effect_ratio"] for value in alternate_reports) <= .50)
    predictions = {"pred_a_instrument": pred_a, "pred_b_position_matched_isotropic_specificity": pred_b, "pred_c_same_subspace_specificity": pred_c, "pred_d_alternate_dct_pair_specificity": pred_d}
    terminal = "mlp9_contextual_dct_specificity_survives_audit" if all(predictions.values()) else "valid_mlp9_contextual_dct_specificity_limitation" if pred_a else "invalid"
    result = {"schema": "mlp9_contextual_dct_causal_fresh_v1_specificity_audit_result", "terminal": terminal, "predictions": predictions, "position_matched_isotropic": isotropic_reports, "same_subspace": subspace_reports, "alternate_dct_pairs": alternate_reports, "alternate_pair_order": [[0, 0], [1, 1], [3, 3]], "maximum_state_replay_relative_l2": max(state_replay), "maximum_suffix_replay_relative_l2": max(suffix_replay), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "new_prefixes": 0, "fits": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Post-outcome adversarial specificity audit on the opened causal panel; it can narrow specificity but cannot add fresh OOD evidence."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "isotropic_target_ratios": [v["target_effect_ratio"] for v in isotropic_reports], "subspace_target_ratios": [v["target_effect_ratio"] for v in subspace_reports], "alternate_target_ratios": [v["target_effect_ratio"] for v in alternate_reports], "alternate_all_logit_ratios": [v["all_logit_effect_ratio"] for v in alternate_reports], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
