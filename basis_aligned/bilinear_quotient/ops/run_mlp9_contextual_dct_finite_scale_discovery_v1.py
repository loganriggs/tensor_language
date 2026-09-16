#!/usr/bin/env python3
# BQGATE:64opened prefixes;6scales;2688suffix states;180seconds;finite-scale discovery.
"""Calibrate finite causal installation/removal for the MLP9 DCT node."""
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

STEM = "MLP9_CONTEXTUAL_DCT_FINITE_SCALE_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ROWS = P / "MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_ROWS.json"
READER = P / "MLP9_CONTEXTUAL_DCT_DOWNSTREAM_READER_DISCOVERY_V1_RESULT.json"
SCALES = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metrics(expected, predicted):
    expected = torch.as_tensor(expected, dtype=torch.float64)
    predicted = torch.as_tensor(predicted, dtype=torch.float64)
    return {
        "relative_l2": float((predicted - expected).norm() / expected.norm().clamp_min(1e-30)),
        "cosine": float((expected * predicted).sum() / (expected.norm() * predicted.norm()).clamp_min(1e-30)),
        "expected_mean_absolute": float(expected.abs().mean()),
        "predicted_mean_absolute": float(predicted.abs().mean()),
        "sign_agreement": float((torch.sign(expected) == torch.sign(predicted)).double().mean()),
    }


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    rows = json.loads(ROWS.read_text())["rows"]
    reader = json.loads(READER.read_text())["selected_reader"]
    if len(rows) != 64 or (reader["first"], reader["second"], reader["positive_token_id"], reader["negative_token_id"]) != (2, 2, 21215, 6165):
        raise ValueError("reader or rows changed")
    return rows, reader, torch.load(WEIGHTS, weights_only=True)


def plan():
    rows, _, _ = load_bound()
    return {"schema": "mlp9_contextual_dct_finite_scale_discovery_v1_plan", "opened_prefixes": len(rows), "scales": list(SCALES), "suffix_states": len(rows) * len(SCALES) * 7, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


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

    rows, reader, cpu_program = load_bound()
    program = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu_program.items()}
    spec = importlib.util.spec_from_file_location("mlp9_contextual_dct_execute_scale", EXECUTOR)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    live_weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = program["directions"][2].float()
    positive, negative = reader["positive_token_id"], reader["negative_token_id"]
    records = {scale: {name: [] for name in ("exact_state", "predicted_state", "native_mixed", "exact_local_effect", "predicted_install_effect", "remaining_after_removal", "random_removal_effect")} for scale in SCALES}
    state_replay = []
    suffix_replay = []
    generator = torch.Generator().manual_seed(2026091643)
    started = time.perf_counter()

    for row in rows:
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, live_weights)
        _, stages = allocation.make_stages(model, native)
        stage = stages[0]
        h0 = stage(native["z9"]).detach()
        state_replay.append(float((h0 - native["h9"]).norm() / native["h9"].norm().clamp_min(1e-30)))
        unit_response = executor.execute(native["z9"], program)[2, 2].detach().float()
        for scale in SCALES:
            tangent = scale * direction[None, None, :]
            h1 = stage(native["z9"] + tangent).detach()
            h2 = stage(native["z9"] + 2 * tangent).detach()
            additive = 2 * h1 - h0
            exact = h2 - additive
            predicted = scale * scale * unit_response
            random = torch.randn(predicted.shape, generator=generator, dtype=torch.float32).to("cuda")
            random = random / random.norm().clamp_min(1e-30) * predicted.norm()
            states = torch.cat((h0, h1, h2, additive, additive + predicted, h2 - predicted, h2 - random), dim=0)
            logits = reader_v1.batched_suffix_logits(model, native, states).detach().float()
            contrast = logits[:, positive] - logits[:, negative]
            native_mixed = contrast[2] - 2 * contrast[1] + contrast[0]
            exact_local = contrast[2] - contrast[3]
            predicted_install = contrast[4] - contrast[3]
            remaining = contrast[5] - 2 * contrast[1] + contrast[0]
            random_effect = contrast[2] - contrast[6]
            rec = records[scale]
            rec["exact_state"].append(exact.cpu())
            rec["predicted_state"].append(predicted.cpu())
            rec["native_mixed"].append(float(native_mixed))
            rec["exact_local_effect"].append(float(exact_local))
            rec["predicted_install_effect"].append(float(predicted_install))
            rec["remaining_after_removal"].append(float(remaining))
            rec["random_removal_effect"].append(float(random_effect))
            if scale == SCALES[0]:
                direct_state = h0
                for downstream in stages[1:]:
                    direct_state = downstream(direct_state)
                direct_last = F.rms_norm(direct_state[:, -1], (1152,))
                direct_logits = (30 * torch.tanh(model.lm_head(direct_last) / 30))[0]
                batched_logits = reader_v1.batched_suffix_logits(model, native, h0)[0]
                suffix_replay.append(float((direct_logits - batched_logits).norm() / direct_logits.norm().clamp_min(1e-30)))

    reports = []
    for scale in SCALES:
        rec = records[scale]
        exact_state = torch.cat([value.reshape(-1) for value in rec["exact_state"]])
        predicted_state = torch.cat([value.reshape(-1) for value in rec["predicted_state"]])
        native_mixed = torch.tensor(rec["native_mixed"])
        exact_local = torch.tensor(rec["exact_local_effect"])
        predicted_install = torch.tensor(rec["predicted_install_effect"])
        remaining = torch.tensor(rec["remaining_after_removal"])
        random_effect = torch.tensor(rec["random_removal_effect"])
        state_report = metrics(exact_state, predicted_state)
        install_report = metrics(exact_local, predicted_install)
        remaining_ratio = float(remaining.norm() / native_mixed.norm().clamp_min(1e-30))
        exact_local_fraction = float(exact_local.norm() / native_mixed.norm().clamp_min(1e-30))
        random_specificity = float(random_effect.norm() / exact_local.norm().clamp_min(1e-30))
        eligible = bool(state_report["relative_l2"] <= .15 and install_report["relative_l2"] <= .20 and remaining_ratio <= .35)
        reports.append({"scale": scale, "eligible": eligible, "state_prediction": state_report, "reader_install_prediction": install_report, "native_mixed_mean_absolute": float(native_mixed.abs().mean()), "native_mixed_sign_agreement": float((torch.sign(native_mixed) == torch.sign(native_mixed.mean())).float().mean()), "exact_local_over_native_mixed_norm": exact_local_fraction, "remaining_after_packaged_removal_over_native_mixed_norm": remaining_ratio, "random_removal_over_exact_local_norm": random_specificity})
    eligible = [report for report in reports if report["eligible"]]
    selected = max(eligible, key=lambda report: report["scale"]) if eligible else None
    predictions = {
        "pred_a_instrument": bool(max(state_replay) <= 2e-6 and max(suffix_replay) <= 2e-6),
        "pred_b_finite_scale_exists": bool(selected is not None),
        "pred_c_registered_selection_rule": bool(len(reports) == len(SCALES) and (selected is None or selected["scale"] == max(report["scale"] for report in eligible))),
    }
    result = {"schema": "mlp9_contextual_dct_finite_scale_discovery_v1_result", "terminal": "mlp9_contextual_dct_finite_scale_candidate" if all(predictions.values()) else "valid_mlp9_contextual_dct_finite_scale_null" if predictions["pred_a_instrument"] else "invalid", "predictions": predictions, "selected_scale": None if selected is None else selected["scale"], "selected_report": selected, "scale_reports": reports, "maximum_state_replay_relative_l2": max(state_replay), "maximum_suffix_replay_relative_l2": max(suffix_replay), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Finite-scale calibration on the opened reader-discovery panel; no fresh-context, semantic behavior, selective collateral, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": predictions, "selected_scale": result["selected_scale"], "reports": reports, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
