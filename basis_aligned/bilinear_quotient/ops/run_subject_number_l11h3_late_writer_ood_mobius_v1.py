#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_discovery_compact pred_c_context_ood pred_d_lexical_ood pred_e_joint_ood pred_f_writer_specificity
"""Exact late-writer Möbius graph with frozen lexical/context OOD evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as base
import run_subject_number_l11h3_subject_value_upstream_mobius_v3 as parent
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = base.DECODER; RANK1 = base.RANK1
PARENT_RESULT = POLY / "SUBJECT_NUMBER_L11H3_SUBJECT_VALUE_UPSTREAM_MOBIUS_V3_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_L11H3_LATE_WRITER_OOD_MOBIUS_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_L11H3_LATE_WRITER_OOD_MOBIUS_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_L11H3_LATE_WRITER_OOD_MOBIUS_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128; NULLS, SEED, STEPS = 16, 20260927, 6
PORTS = ("upstream_0_7", "attn_8", "mlp_8", "attn_9", "mlp_9", "attn_10", "mlp_10", "first_value_bus")
BARS = {"maximum_absolute_closure_error": 1e-4, "maximum_closed_reconstruction_error": 1e-10,
        "maximum_gauge_correction_relative_l2": 1e-5, "maximum_native_value_replay_error": 1e-4,
        "maximum_relative_closure_error": 1e-5, "maximum_synthetic_error": 1e-12,
        "maximum_panel_relative_l2": .25, "minimum_aligned_recovery": 0.,
        "minimum_random_median_advantage": .10}
PRICE = {"partial_forwards": 4, "sequences": 256, "corners": 256, "mobius_terms": 255,
         "candidate_terms": 36, "readouts": 17, "greedy_steps": 6, "behavior_logits": 0,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_instrument": None, "pred_b_discovery_compact": None,
    "pred_c_context_ood": None, "pred_d_lexical_ood": None, "pred_e_joint_ood": None,
    "pred_f_writer_specificity": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "source_primitive": Path(source_factor.__file__),
             "parent_runner": Path(parent.__file__), "parent_result": PARENT_RESULT, "decoder": DECODER,
             "rank1": RANK1, "preregistration": PREREG}
    if binding["files"] != {k: sha(v) for k, v in paths.items()} or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["nulls"] != NULLS or binding["null_seed"] != SEED \
            or binding["steps"] != STEPS or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, parent_result = (json.loads(p.read_text()) for p in (DECODER, RANK1, PARENT_RESULT))
    if decoder["terminal"] != "embedding_number_decoder_frozen" or rank["terminal"] != "rank1_frozen_weights_only" \
            or parent_result["terminal"] != "valid_subject_value_upstream_decomposition":
        raise ValueError("parent status changed")
    return binding, decoder, rank, parent_result, authority.build_rows()

def panel_indices(rows):
    return {
        "discovery": [i for i, r in enumerate(rows) if r["pair_index"] < 8 and r["template_id"] in ("near", "behind")],
        "context_ood": [i for i, r in enumerate(rows) if r["pair_index"] < 8 and r["template_id"] in ("under", "above")],
        "lexical_ood": [i for i, r in enumerate(rows) if r["pair_index"] >= 8 and r["template_id"] in ("near", "behind")],
        "joint_ood": [i for i, r in enumerate(rows) if r["pair_index"] >= 8 and r["template_id"] in ("under", "above")],
    }

def plan():
    _, decoder, rank, parent_result, rows = load_bound(); panels = panel_indices(rows)
    if any(len(v) != 32 for v in panels.values()) or len(set().union(*map(set, panels.values()))) != 128:
        raise ValueError("panel partition failed")
    return {"schema": "subject_number_l11h3_late_writer_ood_mobius_v1_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "rows": len(rows), "panel_sizes": {k: len(v) for k, v in panels.items()},
            "ports": list(PORTS), "eligible_orders": [1, 2], "steps": STEPS,
            "parent_terminal": parent_result["terminal"], "rank": rank["rank"], "authority_sha256": authority.canonical(rows),
            "bars": BARS, "price": PRICE, "binding_sha256": sha(BINDING),
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"]}

def select_terms(target, atoms, candidates):
    residual = target.copy(); selected = []; available = list(candidates); curve = []
    denom = max(float(np.linalg.norm(target)), 1e-30)
    for _ in range(STEPS):
        choice = min(available, key=lambda i: float(np.linalg.norm(residual - atoms[i])))
        residual -= atoms[choice]; available.remove(choice); selected.append(choice)
        curve.append(float(np.linalg.norm(residual) / denom))
    return selected, curve

def evaluate(target, atoms, selected, ids):
    y = target[ids]; prediction = atoms[selected][:, ids].sum(0)
    denom = max(float(np.linalg.norm(y)), 1e-30)
    return {"rows": len(ids), "target_rms": float(np.sqrt(np.mean(y ** 2))),
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
            "relative_l2": float(np.linalg.norm(y - prediction) / denom),
            "aligned_recovery": float(prediction @ y / max(float(y @ y), 1e-30))}

def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(900); binding, decoder, rank, parent_result, rows = load_bound(); panels = panel_indices(rows)
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"], dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"]); unit = decoder_axis / decoder_axis.norm()
    writer = torch.tensor(rank["axis"], dtype=torch.float32, device=device); writer /= writer.norm()
    rng = np.random.default_rng(SEED); axes = [writer]
    for _ in range(NULLS):
        v = torch.tensor(rng.standard_normal(model.config.n_embd), dtype=torch.float32, device=device)
        v -= (v @ writer) * writer; axes.append(v / v.norm())
    axes = torch.stack(axes)
    counts = {"partial_forwards": 0, "sequences": 0, "corners": 256, "mobius_terms": 255,
              "candidate_terms": 36, "readouts": 17, "greedy_steps": 6, "behavior_logits": 0,
              "fits": 0, "backwards": 0, "parameter_updates": 0}
    reconstruction = value_replay = closure_abs = closure_rel = correction_abs = correction_rel = 0.
    all_atoms = np.zeros((17, 255, len(rows)), dtype=np.float32); all_target = np.zeros((17, len(rows)), dtype=np.float32)

    def capture(initial, positions):
        counts["partial_forwards"] += 1; counts["sequences"] += len(initial); x = initial; x0 = initial; first = None
        components = {("embedding", -1): initial.clone()}
        with torch.no_grad():
            for layer, block in enumerate(model.transformer.h):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                for key in components: components[key] = components[key] * block.lambdas[0]
                components[("embedding", -1)] += block.lambdas[1] * x0; state = F.rms_norm(x, (x.shape[-1],))
                if layer == LAYER:
                    _write, factors = source_factor.replay_attention_with_source_factors(
                        state, first, block.attn, positions, HEAD, torch, F, include_qk_factors=True)
                    zero = torch.zeros_like(x)
                    explicit = [sum((v for (kind, idx), v in components.items() if kind == "embedding" or idx <= 7), zero)]
                    explicit.extend(components[(kind, idx)] for idx in range(8, 11) for kind in ("attn", "mlp"))
                    groups = [v.double() for v in explicit]; prefix = sum(groups[:-1]); closed = x.double() - prefix
                    correction = closed - groups[-1]; groups[-1] = closed
                    return x, groups, first, factors, correction, explicit[-1].double()
                attention, first = block.attn(state, first); x = x + attention; components[("attn", layer)] = attention
                mlp = block.mlp(F.rms_norm(x, (x.shape[-1],))); x = x + mlp; components[("mlp", layer)] = mlp
        raise RuntimeError("layer not reached")

    batch_ids = [[i for i, r in enumerate(rows) if r["template_id"] in names]
                 for names in (("near", "behind"), ("under", "above"))]
    for ids in batch_ids:
        batch_rows = [rows[i] for i in ids]; tokens = torch.tensor([r["token_ids"] for r in batch_rows], dtype=torch.long, device=device)
        pos = torch.full((len(ids),), 5, device=device, dtype=torch.long); b = torch.arange(len(ids), device=device)
        with torch.no_grad(): base_input = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
        subject = base_input[b, pos].double(); projection = subject @ unit; target_projection = threshold / float(decoder_axis.norm())
        orthogonal = subject - projection[:, None] * unit
        scale = torch.sqrt((subject.square().sum(1) - target_projection ** 2) / orthogonal.square().sum(1))
        removed_subject = target_projection * unit + scale[:, None] * orthogonal
        removed_input = base_input.clone(); removed_input[b, pos] = removed_subject.float()
        raw_b, groups_b, first_b, factors_b, corr_b, explicit_b = capture(base_input, pos)
        raw_r, groups_r, first_r, factors_r, corr_r, explicit_r = capture(removed_input, pos)
        reconstruction = max(reconstruction, float((sum(groups_b) - raw_b.double()).abs().max()),
                             float((sum(groups_r) - raw_r.double()).abs().max()))
        correction_abs = max(correction_abs, float(corr_b.abs().max()), float(corr_r.abs().max()))
        correction_rel = max(correction_rel, float(corr_b.norm() / explicit_b.norm().clamp_min(1e-30)),
                             float(corr_r.norm() / explicit_r.norm().clamp_min(1e-30)))
        attention = model.transformer.h[LAYER].attn; head_slice = attention.c_proj.weight[:, HEAD*HEAD_WIDTH:(HEAD+1)*HEAD_WIDTH]
        p_subject = factors_b["p"][:, 5]
        def corner(mask):
            raw = sum((groups_r if mask & (1 << i) else groups_b)[i] for i in range(7)).float()
            state = F.rms_norm(raw, (raw.shape[-1],)); current = attention.c_v(state).view(len(ids), 6, 9, 128)[:, :, HEAD]
            first = (first_r if mask & (1 << 7) else first_b).view(len(ids), 6, 9, 128)[:, :, HEAD]
            effective = (1 - attention.lamb) * current + attention.lamb * first
            u = F.linear(effective[:, 5], head_slice); return p_subject[:, None] * (u @ axes.T)
        values = {mask: corner(mask) for mask in range(256)}; dividends = {}
        for mask in range(256):
            div = values[mask].clone(); sub = (mask - 1) & mask
            while sub: div -= dividends[sub]; sub = (sub - 1) & mask
            if mask: div -= dividends[0]
            dividends[mask] = div
        target = values[0] - values[255]; closure = -sum(dividends[m] for m in range(1, 256))
        closure_abs = max(closure_abs, float((closure - target).abs().max()))
        closure_rel = max(closure_rel, float((closure - target).norm() / target.norm().clamp_min(1e-30)))
        direct = (factors_b["p"][:, 5, None] * (factors_b["u"][:, 5] - factors_r["u"][:, 5])) @ axes.T
        value_replay = max(value_replay, float((direct - target).abs().max()))
        stacked = torch.stack([-dividends[m] for m in range(1, 256)]).cpu().numpy()
        all_atoms[:, :, ids] = stacked.transpose(2, 0, 1); all_target[:, ids] = target.cpu().numpy().T

    names = ["*".join(PORTS[i] for i in range(8) if m & (1 << i)) for m in range(1, 256)]
    candidate_indices = [m - 1 for m in range(1, 256) if m.bit_count() <= 2]
    reports = []
    for ai in range(17):
        target = all_target[ai]; atoms = all_atoms[ai]; discovery = panels["discovery"]
        selected, curve = select_terms(target[discovery], atoms[:, discovery], candidate_indices)
        panel_report = {name: evaluate(target, atoms, selected, ids) for name, ids in panels.items()}
        reports.append({"selected_indices": selected, "selected_masks": [i + 1 for i in selected],
                        "selected_terms": [names[i] for i in selected], "discovery_curve": curve, "panels": panel_report})
    target_report = reports[0]; random_joint = np.asarray([r["panels"]["joint_ood"]["relative_l2"] for r in reports[1:]])
    advantage = float(np.median(random_joint) - target_report["panels"]["joint_ood"]["relative_l2"])
    term_details = []
    for idx in target_report["selected_indices"]:
        term_details.append({"mask": idx + 1, "name": names[idx], "ports": [PORTS[i] for i in range(8) if (idx + 1) & (1 << i)],
            "order": (idx + 1).bit_count(), "all_rows_rms": float(np.sqrt(np.mean(all_atoms[0, idx] ** 2))),
            "all_rows_aligned_recovery": float(all_atoms[0, idx] @ all_target[0] / max(float(all_target[0] @ all_target[0]), 1e-30))})
    target_report["selected_term_details"] = term_details
    synthetic = base.parent.synthetic_fixture()
    finite = bool(np.isfinite(np.asarray([reconstruction, correction_abs, correction_rel, value_replay, closure_abs, closure_rel,
                                          synthetic, advantage, *random_joint])).all())
    pred_a = bool(finite and reconstruction <= BARS["maximum_closed_reconstruction_error"]
        and correction_rel <= BARS["maximum_gauge_correction_relative_l2"]
        and value_replay <= BARS["maximum_native_value_replay_error"] and closure_abs <= BARS["maximum_absolute_closure_error"]
        and closure_rel <= BARS["maximum_relative_closure_error"] and synthetic <= BARS["maximum_synthetic_error"]
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    def panel_pass(name):
        p = target_report["panels"][name]
        return bool(p["relative_l2"] <= BARS["maximum_panel_relative_l2"] and p["aligned_recovery"] > BARS["minimum_aligned_recovery"])
    pred_b = bool(pred_a and panel_pass("discovery")); pred_c = bool(pred_b and panel_pass("context_ood"))
    pred_d = bool(pred_b and panel_pass("lexical_ood")); pred_e = bool(pred_c and pred_d and panel_pass("joint_ood"))
    pred_f = bool(pred_e and advantage >= BARS["minimum_random_median_advantage"])
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)))
    terminal = "specific_late_writer_ood_graph" if pred_f else "valid_late_writer_ood_decomposition" if pred_a else "invalid"
    result = {"schema": "subject_number_l11h3_late_writer_ood_mobius_v1_result", "terminal": terminal, "predictions": predictions,
        "instrument": {"finite": finite, "closed_reconstruction_max_abs_error": reconstruction,
            "gauge_correction_max_abs": correction_abs, "gauge_correction_max_relative_l2": correction_rel,
            "native_subject_value_replay_max_abs_error": value_replay, "mobius_closure_max_abs_error": closure_abs,
            "mobius_closure_max_relative_l2": closure_rel, "synthetic_fixture_error": synthetic, "counts": counts},
        "selection": {"eligible_orders": [1, 2], "candidate_count": len(candidate_indices), "steps": STEPS,
                      "coefficient_fits": 0, "selected_on": "discovery_only"},
        "writer_axis": target_report, "random_readout_specificity": {"count": NULLS, "seed": SEED,
            "joint_ood_relative_l2": random_joint.tolist(), "median_joint_ood_relative_l2": float(np.median(random_joint)),
            "target_joint_ood_relative_l2": target_report["panels"]["joint_ood"]["relative_l2"], "median_advantage": advantage},
        "bars": BARS, "price": PRICE, "ports": list(PORTS), "authority_sha256": authority.canonical(rows),
        "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER), "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "wall_seconds": time.perf_counter() - started,
        "scope": "Exact eight-port late-writer decomposition; six unscaled order<=2 terms frozen on discovery and tested on context, lexical, and joint OOD panels."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "instrument", "selection", "writer_axis", "random_readout_specificity")}, indent=2, sort_keys=True))
    assert pred_a

if __name__ == "__main__": main()
