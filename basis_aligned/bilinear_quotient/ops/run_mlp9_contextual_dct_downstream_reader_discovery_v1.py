#!/usr/bin/env python3
# BQGATE:64opened prefixes;32-way batched suffix JVP;180seconds;reader discovery only.
"""Discover a downstream logit reader for the extracted MLP9 DCT node."""
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
import run_mlp9_contextual_dct_node_fresh_v1 as node_v1
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_CONTEXTUAL_DCT_DOWNSTREAM_READER_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ROWS = P / "MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_ROWS.json"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def aggregate(expected, predicted):
    expected = expected.double()
    predicted = predicted.double()
    expected_sq = expected.square().sum()
    predicted_sq = predicted.square().sum()
    dot = (expected * predicted).sum()
    return {
        "relative_l2": float((predicted - expected).norm() / expected.norm().clamp_min(1e-30)),
        "cosine": float(dot / torch.sqrt(expected_sq * predicted_sq).clamp_min(1e-30)),
        "expected_rms": float(expected.square().mean().sqrt()),
        "predicted_rms": float(predicted.square().mean().sqrt()),
    }


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    rows = json.loads(ROWS.read_text())["rows"]
    if len(rows) != 64:
        raise ValueError("expected 64 discovery rows")
    return rows, torch.load(WEIGHTS, weights_only=True)


def plan():
    rows, program = load_bound()
    return {
        "schema": "mlp9_contextual_dct_downstream_reader_discovery_v1_plan",
        "opened_prefixes": len(rows),
        "ordered_pairs": 16,
        "suffix_tangents_per_prefix": 32,
        "rank": int(program["selected_rank"]),
        "new_text": 0,
        "finite_interventions": 0,
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
    }


def batched_suffix_logits(model, native, value):
    batch = value.shape[0]
    x0 = native["x0"].expand(batch, -1, -1)
    v1 = native["v1"].expand(batch, *native["v1"].shape[1:])
    for block in model.transformer.h[10:]:
        value, v1 = block(value, v1, x0)
    last = F.rms_norm(value[:, -1], (1152,))
    return 30 * torch.tanh(model.lm_head(last) / 30)


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
    import tiktoken

    rows, cpu_program = load_bound()
    program = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu_program.items()}
    spec = importlib.util.spec_from_file_location("mlp9_contextual_dct_execute", EXECUTOR)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    live_weights = live.assembled.load_weights(model.state_dict(), "cuda")
    directions = program["directions"].float()
    expected_rows = []
    predicted_rows = []
    state_replay = []
    suffix_replay = []
    started = time.perf_counter()

    for row in rows:
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, live_weights)
        _, stages = allocation.make_stages(model, native)
        expected = node_v1.stage_responses(stages[0], native["z9"], directions).detach().float().reshape(16, *native["z9"].shape[1:])
        predicted = executor.execute(native["z9"], program).detach().float().reshape_as(expected)
        h9 = stages[0](native["z9"]).detach()
        state_replay.append(float((h9 - native["h9"]).norm() / native["h9"].norm().clamp_min(1e-30)))
        point = h9.expand(32, -1, -1).contiguous()
        tangents = torch.cat((expected, predicted), dim=0).contiguous()
        base_logits = batched_suffix_logits(model, native, point)
        _, tangent_logits = torch.func.jvp(
            lambda value: batched_suffix_logits(model, native, value),
            (point,),
            (tangents,),
        )
        direct_state = h9
        for downstream in stages[1:]:
            direct_state = downstream(direct_state)
        direct_last = F.rms_norm(direct_state[:, -1], (1152,))
        direct = (30 * torch.tanh(model.lm_head(direct_last) / 30))[0].detach()
        suffix_replay.append(float((base_logits[0] - direct).norm() / direct.norm().clamp_min(1e-30)))
        expected_rows.append(tangent_logits[:16, :50257].detach().cpu())
        predicted_rows.append(tangent_logits[16:, :50257].detach().cpu())

    expected = torch.stack(expected_rows)
    predicted = torch.stack(predicted_rows)
    overall = aggregate(expected, predicted)
    mean = expected.mean(0)
    rms = expected.square().mean(0).sqrt().clamp_min(1e-30)
    consistency = mean.abs() / rms
    candidates = []
    for pair in range(16):
        positive = int(torch.argmax(mean[pair] / rms[pair]))
        negative = int(torch.argmin(mean[pair] / rms[pair]))
        contrast = expected[:, pair, positive] - expected[:, pair, negative]
        predicted_contrast = predicted[:, pair, positive] - predicted[:, pair, negative]
        report = aggregate(contrast, predicted_contrast)
        sign = torch.sign(contrast.mean())
        sign_agreement = float((torch.sign(contrast) == sign).float().mean())
        candidates.append({
            "pair_flat": pair,
            "first": pair // 4,
            "second": pair % 4,
            "positive_token_id": positive,
            "negative_token_id": negative,
            "mean_exact_contrast": float(contrast.mean()),
            "mean_absolute_exact_contrast": float(contrast.abs().mean()),
            "exact_contrast_rms": float(contrast.square().mean().sqrt()),
            "sign_agreement": sign_agreement,
            "prediction": report,
            "selection_score": float(contrast.mean().abs() / contrast.square().mean().sqrt().clamp_min(1e-30)),
        })
    selected = max(candidates, key=lambda value: (value["selection_score"], value["mean_absolute_exact_contrast"]))
    encoder = tiktoken.get_encoding("gpt2")
    selected["positive_token_text"] = encoder.decode([selected["positive_token_id"]])
    selected["negative_token_text"] = encoder.decode([selected["negative_token_id"]])
    pred_a = bool(max(state_replay) <= 2e-6 and max(suffix_replay) <= 2e-6)
    pred_b = bool(pred_a and overall["relative_l2"] <= .02 and overall["cosine"] >= .999)
    pred_c = bool(pred_a and selected["sign_agreement"] >= .90 and selected["mean_absolute_exact_contrast"] >= .005)
    pred_d = bool(pred_a and selected["prediction"]["relative_l2"] <= .03 and selected["prediction"]["cosine"] >= .999 and selected["mean_exact_contrast"] * (predicted[:, selected["pair_flat"], selected["positive_token_id"]] - predicted[:, selected["pair_flat"], selected["negative_token_id"]]).mean().item() > 0)
    pred_e = bool(selected["positive_token_id"] < 50257 and selected["negative_token_id"] < 50257 and selected["positive_token_id"] != selected["negative_token_id"])
    predictions = {"pred_a_instrument": pred_a, "pred_b_downstream_prediction": pred_b, "pred_c_stable_reader_candidate": pred_c, "pred_d_package_predicts_reader": pred_d, "pred_e_no_padded_token_artifact": pred_e}
    terminal = "mlp9_contextual_dct_reader_candidate" if all(predictions.values()) else "valid_mlp9_contextual_dct_reader_null" if pred_a else "invalid"
    result = {
        "schema": "mlp9_contextual_dct_downstream_reader_discovery_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "overall_logit_response": overall,
        "selected_reader": selected,
        "pair_candidates": candidates,
        "maximum_state_replay_relative_l2": max(state_replay),
        "maximum_suffix_replay_relative_l2": max(suffix_replay),
        "runner_sha256": digest(RUNNER),
        "binding_sha256": digest(BINDING),
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "seconds": time.perf_counter() - started,
        "price": planned | {"checkpoint_loads": 1, "suffix_jvp_calls": len(rows)},
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Reader discovery on an already opened response panel; exact native suffix remains live; no fresh-context, finite-intervention, selective-removal, or complete-circuit claim.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "overall": overall, "selected": selected, "state_replay_max": max(state_replay), "suffix_replay_max": max(suffix_replay), "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
