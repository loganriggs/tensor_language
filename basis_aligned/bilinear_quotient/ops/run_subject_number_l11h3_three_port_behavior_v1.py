#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument_and_capability pred_b_three_port_removal_ood pred_c_selective_removal pred_d_behavioral_composition pred_e_three_port_rescue
"""Behavioral removal, rescue, and composition test of the frozen three-port graph."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_late_writer_ood_mobius_v1 as graph
import run_subject_number_l11h3_late_writer_specificity_audit_v2 as parent
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as base
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = base.DECODER
PARENT_RESULT = POLY / "SUBJECT_NUMBER_L11H3_LATE_WRITER_SPECIFICITY_AUDIT_V2_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_L11H3_THREE_PORT_BEHAVIOR_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_L11H3_THREE_PORT_BEHAVIOR_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_L11H3_THREE_PORT_BEHAVIOR_V1_RESULT.json"
LAYER = 11; NULLS, SEED = 16, 20260929
PORTS = ("upstream_0_7", "mlp_8", "mlp_10"); GROUP_INDICES = (0, 2, 6)
BARS = {"maximum_closed_reconstruction_error": 1e-10, "maximum_control_fraction": .50,
        "maximum_native_suffix_replay_error": 1e-5, "maximum_random_norm_error": 1e-5,
        "maximum_gauge_correction_relative_l2": 1e-5, "maximum_composition_relative_l2": .25,
        "minimum_composition_cosine": .90, "minimum_native_accuracy": .75,
        "minimum_panel_damage_positive_fraction": .60, "minimum_panel_damage_rms": .10,
        "minimum_random_median_ratio": 2., "minimum_rescue_cosine": .50,
        "minimum_aligned_recovery": 0.}
PRICE = {"partial_forwards": 4, "partial_sequences": 256, "native_forwards": 4,
         "native_sequences": 256, "suffix_forwards": 46, "suffix_sequences": 2944,
         "random_controls": 16, "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_instrument_and_capability": None,
    "pred_b_three_port_removal_ood": None, "pred_c_selective_removal": None,
    "pred_d_behavioral_composition": None, "pred_e_three_port_rescue": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual); pn = np.linalg.norm(predicted); an = np.linalg.norm(actual)
    return {"cosine": float(predicted @ actual / max(pn * an, 1e-30)),
            "relative_l2": float(np.linalg.norm(predicted - actual) / max(an, 1e-30)),
            "aligned_recovery": float(predicted @ actual / max(actual @ actual, 1e-30)),
            "predicted_rms": float(np.sqrt(np.mean(predicted ** 2))),
            "actual_rms": float(np.sqrt(np.mean(actual ** 2)))}

def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "parent_runner": Path(parent.__file__),
             "parent_result": PARENT_RESULT, "decoder": DECODER, "preregistration": PREREG}
    if binding["files"] != {k: sha(v) for k, v in paths.items()} or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["nulls"] != NULLS or binding["null_seed"] != SEED \
            or binding["ports"] != list(PORTS) or binding["group_indices"] != list(GROUP_INDICES) or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, parent_result = (json.loads(p.read_text()) for p in (DECODER, PARENT_RESULT))
    if decoder["terminal"] != "embedding_number_decoder_frozen" or parent_result["terminal"] != "valid_late_writer_specificity_audit" \
            or not parent_result["predictions"]["pred_c_three_term_pruning"] \
            or parent_result["target_writer"]["expected_masks"][:3] != [4, 1, 64]:
        raise ValueError("parent status changed")
    return binding, decoder, parent_result, authority.build_rows()

def plan():
    _, decoder, parent_result, rows = load_bound(); panels = graph.panel_indices(rows)
    return {"schema": "subject_number_l11h3_three_port_behavior_v1_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "rows": len(rows), "panel_sizes": {k: len(v) for k, v in panels.items()},
            "ports": list(PORTS), "group_indices": list(GROUP_INDICES), "parent_terminal": parent_result["terminal"],
            "authority_sha256": authority.canonical(rows), "bars": BARS, "price": PRICE,
            "binding_sha256": sha(BINDING), "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"]}

def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(1200); binding, decoder, parent_result, rows = load_bound(); panels = graph.panel_indices(rows)
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"], dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"]); unit = decoder_axis / decoder_axis.norm()
    rng = np.random.default_rng(SEED)
    random_axes = [torch.tensor(rng.standard_normal(model.config.n_embd), dtype=torch.float64, device=device) for _ in range(NULLS)]
    counts = {"partial_forwards": 0, "partial_sequences": 0, "native_forwards": 0, "native_sequences": 0,
              "suffix_forwards": 0, "suffix_sequences": 0, "random_controls": NULLS,
              "fits": 0, "backwards": 0, "parameter_updates": 0}
    reconstruction = correction_rel = suffix_replay = random_norm_error = 0.; outputs = []
    answer_ids = {"singular": authority.task14.ENCODING.encode(" is")[0], "plural": authority.task14.ENCODING.encode(" are")[0]}
    control_ids = [authority.task14.ENCODING.encode(" can")[0], authority.task14.ENCODING.encode(" will")[0]]

    def capture(initial):
        nonlocal reconstruction, correction_rel
        counts["partial_forwards"] += 1; counts["partial_sequences"] += len(initial)
        x = initial; x0 = initial; first = None; components = {("embedding", -1): initial.clone()}
        with torch.no_grad():
            for layer, block in enumerate(model.transformer.h):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                for key in components: components[key] = components[key] * block.lambdas[0]
                components[("embedding", -1)] += block.lambdas[1] * x0; state = F.rms_norm(x, (x.shape[-1],))
                if layer == LAYER:
                    zero = torch.zeros_like(x)
                    explicit = [sum((v for (kind, idx), v in components.items() if kind == "embedding" or idx <= 7), zero)]
                    explicit.extend(components[(kind, idx)] for idx in range(8, 11) for kind in ("attn", "mlp"))
                    groups = [v.double() for v in explicit]; closed = x.double() - sum(groups[:-1])
                    correction_rel = max(correction_rel, float((closed - groups[-1]).norm() / groups[-1].norm().clamp_min(1e-30)))
                    groups[-1] = closed; reconstruction = max(reconstruction, float((sum(groups) - x.double()).abs().max()))
                    return x, x0, first, groups
                attention, first = block.attn(state, first); x = x + attention
                components[("attn", layer)] = attention
                mlp = block.mlp(F.rms_norm(x, (x.shape[-1],))); x = x + mlp; components[("mlp", layer)] = mlp
        raise RuntimeError("layer not reached")

    def readout(x, positions, selected):
        n = len(x); batch = torch.arange(n, device=device)
        logits = (30. * torch.tanh(model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.)).float()
        return logits[batch[:, None], positions[:, None], selected].cpu().numpy()

    def native(initial, positions, selected):
        counts["native_forwards"] += 1; counts["native_sequences"] += len(initial)
        with torch.no_grad():
            x = initial; x0 = initial; first = None
            for block in model.transformer.h:
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first); x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            return readout(x, positions, selected)

    def suffix(raw, x0, first, positions, selected):
        counts["suffix_forwards"] += 1; counts["suffix_sequences"] += len(raw)
        with torch.no_grad():
            x = raw
            for layer in range(LAYER, len(model.transformer.h)):
                block = model.transformer.h[layer]
                if layer > LAYER: x = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first); x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            return readout(x, positions, selected)

    def process(ids):
        nonlocal suffix_replay, random_norm_error
        batch_rows = [rows[i] for i in ids]; n = len(ids); batch = torch.arange(n, device=device)
        tokens = torch.tensor([r["token_ids"] for r in batch_rows], dtype=torch.long, device=device)
        positions = torch.tensor([r["subject_position"] for r in batch_rows], dtype=torch.long, device=device)
        selected = torch.tensor([[r["native_answer_id"], answer_ids["plural" if r["number"] == "singular" else "singular"], *control_ids]
                                 for r in batch_rows], dtype=torch.long, device=device)
        with torch.no_grad(): base_input = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
        subject = base_input[batch, positions].double(); projection = subject @ unit; target_projection = threshold / float(decoder_axis.norm())
        orthogonal = subject - projection[:, None] * unit
        scale = torch.sqrt((subject.square().sum(1) - target_projection ** 2) / orthogonal.square().sum(1))
        removed_subject = target_projection * unit + scale[:, None] * orthogonal
        removed_input = base_input.clone(); removed_input[batch, positions] = removed_subject.float()
        raw_b, x0_b, first_b, groups_b = capture(base_input); raw_r, x0_r, first_r, groups_r = capture(removed_input)
        native_b = native(base_input, positions, selected); native_r = native(removed_input, positions, selected)
        suffix_b = suffix(raw_b, x0_b, first_b, positions, selected); suffix_r = suffix(raw_r, x0_r, first_r, positions, selected)
        suffix_replay = max(suffix_replay, float(np.max(np.abs(native_b - suffix_b))), float(np.max(np.abs(native_r - suffix_r))))
        deltas = [(groups_r[i] - groups_b[i])[batch, positions] for i in GROUP_INDICES]; joint_delta = sum(deltas)
        joint_raw = raw_b.clone(); joint_raw[batch, positions] += joint_delta.float()
        joint = suffix(joint_raw, x0_b, first_b, positions, selected)
        rescue_raw = raw_r.clone(); rescue_raw[batch, positions] -= joint_delta.float()
        rescue = suffix(rescue_raw, x0_r, first_r, positions, selected)
        individual = []
        for delta in deltas:
            edited = raw_b.clone(); edited[batch, positions] += delta.float()
            individual.append(suffix(edited, x0_b, first_b, positions, selected))
        random_values = []
        target_norm = joint_delta.norm(dim=1)
        for axis in random_axes:
            direction = axis[None, :].expand(n, -1)
            direction = direction - ((direction * joint_delta).sum(1) / joint_delta.square().sum(1).clamp_min(1e-30))[:, None] * joint_delta
            random_delta = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-30) * target_norm[:, None]
            random_norm_error = max(random_norm_error, float(((random_delta.norm(dim=1) - target_norm).abs() / target_norm.clamp_min(1e-30)).max()))
            edited = raw_b.clone(); edited[batch, positions] += random_delta.float()
            random_values.append(suffix(edited, x0_b, first_b, positions, selected))
        return {"native_base": native_b, "native_removed": native_r, "base": suffix_b, "removed": suffix_r,
                "joint": joint, "rescue": rescue, "individual": np.asarray(individual), "random": np.asarray(random_values),
                "target_edit_norm": target_norm.cpu().numpy()}

    batch_ids = [[i for i, r in enumerate(rows) if r["template_id"] in names]
                 for names in (("near", "behind"), ("under", "above"))]
    outputs = [process(ids) for ids in batch_ids]
    def combine(key):
        sample = outputs[0][key]
        if sample.ndim == 2: result = np.empty((len(rows), sample.shape[1]), dtype=sample.dtype)
        elif sample.ndim == 1: result = np.empty(len(rows), dtype=sample.dtype)
        else: result = np.empty((sample.shape[0], len(rows), sample.shape[2]), dtype=sample.dtype)
        for ids, out in zip(batch_ids, outputs):
            if sample.ndim <= 2: result[ids] = out[key]
            else: result[:, ids] = out[key]
        return result
    native_base, native_removed, base_values, removed_values, joint_values, rescue_values = \
        (combine(k) for k in ("native_base", "native_removed", "base", "removed", "joint", "rescue"))
    individual_values, random_values, target_edit_norm = combine("individual"), combine("random"), combine("target_edit_norm")
    base_margin = base_values[:, 0] - base_values[:, 1]; removed_margin = removed_values[:, 0] - removed_values[:, 1]
    joint_margin = joint_values[:, 0] - joint_values[:, 1]; rescue_margin = rescue_values[:, 0] - rescue_values[:, 1]
    damage = base_margin - joint_margin; full_damage = base_margin - removed_margin; rescue_effect = rescue_margin - removed_margin
    individual_damage = base_margin[None, :] - (individual_values[:, :, 0] - individual_values[:, :, 1])
    composed_damage = individual_damage.sum(0)
    random_damage = base_margin[None, :] - (random_values[:, :, 0] - random_values[:, :, 1])
    control_change = (joint_values[:, 2] - joint_values[:, 3]) - (base_values[:, 2] - base_values[:, 3])
    panel_damage = {}; panel_composition = {}
    for name, ids in panels.items():
        panel_damage[name] = {"rows": len(ids), "rms": float(np.sqrt(np.mean(damage[ids] ** 2))),
                              "positive_fraction": float(np.mean(damage[ids] > 0)),
                              "full_embedding_damage_rms": float(np.sqrt(np.mean(full_damage[ids] ** 2)))}
        panel_composition[name] = metrics(composed_damage[ids], damage[ids])
    damage_rms = float(np.sqrt(np.mean(damage ** 2))); random_rms = np.sqrt(np.mean(random_damage ** 2, axis=1))
    control_rms = float(np.sqrt(np.mean(control_change ** 2))); rescue_report = metrics(rescue_effect, full_damage)
    capability = {name: float(np.mean(base_margin[ids] > 0)) for name, ids in panels.items()}
    finite = bool(np.isfinite(np.asarray([*damage, *full_damage, *rescue_effect, *individual_damage.ravel(),
                                          *random_damage.ravel(), *target_edit_norm])).all())
    pred_a = bool(finite and suffix_replay <= BARS["maximum_native_suffix_replay_error"]
        and reconstruction <= BARS["maximum_closed_reconstruction_error"] and correction_rel <= BARS["maximum_gauge_correction_relative_l2"]
        and random_norm_error <= BARS["maximum_random_norm_error"] and all(v >= BARS["minimum_native_accuracy"] for v in capability.values())
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    pred_b = bool(pred_a and all(p["rms"] >= BARS["minimum_panel_damage_rms"]
        and p["positive_fraction"] >= BARS["minimum_panel_damage_positive_fraction"] for p in panel_damage.values()))
    pred_c = bool(pred_b and damage_rms / max(float(np.median(random_rms)), 1e-30) >= BARS["minimum_random_median_ratio"]
        and control_rms / max(damage_rms, 1e-30) <= BARS["maximum_control_fraction"])
    pred_d = bool(pred_b and all(p["relative_l2"] <= BARS["maximum_composition_relative_l2"]
        and p["cosine"] >= BARS["minimum_composition_cosine"] and p["aligned_recovery"] > BARS["minimum_aligned_recovery"]
        for p in panel_composition.values()))
    pred_e = bool(pred_b and rescue_report["cosine"] >= BARS["minimum_rescue_cosine"]
        and rescue_report["aligned_recovery"] > BARS["minimum_aligned_recovery"])
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d, pred_e)))
    terminal = "three_port_behavioral_graph" if all(predictions.values()) else "valid_three_port_behavior_test" if pred_a else "invalid"
    result = {"schema": "subject_number_l11h3_three_port_behavior_v1_result", "terminal": terminal, "predictions": predictions,
        "instrument": {"finite": finite, "native_suffix_replay_max_abs_error": suffix_replay,
            "closed_reconstruction_max_abs_error": reconstruction, "gauge_correction_max_relative_l2": correction_rel,
            "random_edit_norm_max_relative_error": random_norm_error, "native_capability": capability, "counts": counts},
        "extraction": {"native_port_count": 3, "ports": list(PORTS), "site": "pre_L11_attention_subject_position",
                       "fitted_coefficients": 0, "native_upstream_state_required": True},
        "removal": {"overall_damage_rms": damage_rms, "panels": panel_damage,
            "control_change_rms": control_rms, "control_fraction": control_rms / max(damage_rms, 1e-30),
            "target_edit_norm_rms": float(np.sqrt(np.mean(target_edit_norm ** 2)))},
        "equal_l2_same_site_controls": {"count": NULLS, "seed": SEED, "damage_rms": random_rms.tolist(),
            "median_damage_rms": float(np.median(random_rms)), "maximum_damage_rms": float(np.max(random_rms)),
            "target_to_median_ratio": damage_rms / max(float(np.median(random_rms)), 1e-30)},
        "composition": {"individual_damage_rms": np.sqrt(np.mean(individual_damage ** 2, axis=1)).tolist(),
                        "panels": panel_composition},
        "rescue_to_full_embedding_removal_damage": rescue_report,
        "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows), "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER), "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "wall_seconds": time.perf_counter() - started,
        "scope": "Frozen three-native-port pre-L11 behavioral removal, rescue, equal-L2 null, and component-relative composition test."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "instrument", "extraction", "removal", "equal_l2_same_site_controls", "composition", "rescue_to_full_embedding_removal_damage")}, indent=2, sort_keys=True))
    assert pred_a

if __name__ == "__main__": main()
