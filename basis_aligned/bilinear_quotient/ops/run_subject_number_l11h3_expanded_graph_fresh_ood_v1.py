#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_fresh_instrument pred_b_frozen_graph_fresh_ood pred_c_fresh_removal pred_d_fresh_selectivity
"""Fresh noun/template/length OOD test of the frozen nine-edge native-port graph."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_expanded_behavioral_graph_v2 as parent
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as base
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = base.DECODER
PARENT_RESULT = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V2_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_GRAPH_FRESH_OOD_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_GRAPH_FRESH_OOD_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_GRAPH_FRESH_OOD_V1_RESULT.json"
LAYER, NULLS, SEED = 11, 16, 20260930
PORTS = ("embedding_recurrence", "early_writes_0_3", "middle_writes_4_7", "mlp_8", "mlp_10")
EXPECTED_MASKS = (8, 1, 4, 2, 24, 12, 16, 9, 20)
BARS = {"maximum_absolute_closure_error": 1e-10, "maximum_aggregation_error": 1e-10,
        "maximum_composition_relative_l2": .15, "maximum_control_fraction": .50,
        "maximum_gauge_correction_relative_l2": 1e-5, "maximum_native_replay_error": 1e-5,
        "maximum_random_norm_error": 1e-5, "maximum_relative_closure_error": 1e-10,
        "minimum_aligned_recovery": 0., "minimum_composition_cosine": .98,
        "minimum_native_accuracy": .75, "minimum_panel_damage_positive_fraction": .60,
        "minimum_panel_damage_rms": .10, "minimum_random_median_ratio": 2.}
PRICE = {"partial_forwards": 4, "partial_sequences": 64, "native_forwards": 2,
         "native_sequences": 32, "suffix_forwards": 96, "suffix_sequences": 1536,
         "corners": 32, "mobius_terms": 31, "frozen_terms": 9, "random_controls": 16,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_fresh_instrument": None,
    "pred_b_frozen_graph_fresh_ood": None, "pred_c_fresh_removal": None,
    "pred_d_fresh_selectivity": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cells(rows):
    result = {"overall": list(range(len(rows)))}
    for direction in ("singular_to_plural", "plural_to_singular"):
        for template in ("behind_beside", "among_behind"):
            result[f"{direction}|{template}"] = [i for i, r in enumerate(rows)
                if r["direction_id"] == direction and r["template_id"] == template]
    return result

def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "parent_runner": Path(parent.__file__),
             "parent_result": PARENT_RESULT, "decoder": DECODER, "preregistration": PREREG}
    if binding["files"] != {k: sha(v) for k, v in paths.items()} or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["ports"] != list(PORTS) \
            or binding["expected_masks"] != list(EXPECTED_MASKS) or binding["nulls"] != NULLS \
            or binding["null_seed"] != SEED or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, parent_result = (json.loads(p.read_text()) for p in (DECODER, PARENT_RESULT))
    if decoder["terminal"] != "embedding_number_decoder_frozen" or parent_result["terminal"] != "expanded_sparse_behavioral_graph" \
            or parent_result["graph"]["selected_masks"][:9] != list(EXPECTED_MASKS):
        raise ValueError("parent status or masks changed")
    rows = authority.build_rows()
    return binding, decoder, parent_result, rows

def plan():
    _, decoder, parent_result, rows = load_bound(); groups = cells(rows)
    return {"schema": "subject_number_l11h3_expanded_graph_fresh_ood_v1_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "rows": len(rows), "cell_sizes": {k: len(v) for k, v in groups.items()},
            "ports": list(PORTS), "frozen_masks": list(EXPECTED_MASKS), "parent_terminal": parent_result["terminal"],
            "authority_sha256": authority.canonical_sha256(rows), "bars": BARS, "price": PRICE,
            "binding_sha256": sha(BINDING), "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"]}

def metric(prediction, target):
    yn = np.linalg.norm(target); pn = np.linalg.norm(prediction)
    return {"rows": len(target), "target_rms": float(np.sqrt(np.mean(target ** 2))),
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
            "relative_l2": float(np.linalg.norm(target - prediction) / max(yn, 1e-30)),
            "cosine": float(prediction @ target / max(pn * yn, 1e-30)),
            "aligned_recovery": float(prediction @ target / max(target @ target, 1e-30))}

def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(1200); binding, decoder, parent_result, rows = load_bound(); groups = cells(rows)
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"], dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"]); unit = decoder_axis / decoder_axis.norm()
    rng = np.random.default_rng(SEED)
    random_axes = [torch.tensor(rng.standard_normal(model.config.n_embd), dtype=torch.float64, device=device) for _ in range(NULLS)]
    counts = {"partial_forwards": 0, "partial_sequences": 0, "native_forwards": 0, "native_sequences": 0,
              "suffix_forwards": 0, "suffix_sequences": 0, "corners": 32, "mobius_terms": 31,
              "frozen_terms": 9, "random_controls": NULLS, "fits": 0, "backwards": 0, "parameter_updates": 0}
    aggregation_error = gauge_rel = native_replay = random_norm_error = closure_abs = closure_rel = 0.
    atoms = np.zeros((31, len(rows)), dtype=np.float64); target = np.zeros(len(rows), dtype=np.float64)
    native_correct = np.zeros(len(rows), dtype=bool); control_change = np.zeros(len(rows), dtype=np.float64)
    random_damage = np.zeros((NULLS, len(rows)), dtype=np.float64)

    def capture(initial):
        nonlocal aggregation_error, gauge_rel
        counts["partial_forwards"] += 1; counts["partial_sequences"] += len(initial)
        x = initial; x0 = initial; first = None; components = {("embedding", -1): initial.clone()}
        with torch.no_grad():
            for layer, block in enumerate(model.transformer.h):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                for key in components: components[key] = components[key] * block.lambdas[0]
                components[("embedding", -1)] += block.lambdas[1] * x0; state = F.rms_norm(x, (x.shape[-1],))
                if layer == LAYER:
                    zero = torch.zeros_like(x)
                    upstream = sum((v for (kind, idx), v in components.items() if kind == "embedding" or idx <= 7), zero).double()
                    embedding = components[("embedding", -1)].double()
                    early = sum((v for (kind, idx), v in components.items() if kind != "embedding" and idx in range(0, 4)), zero).double()
                    middle_explicit = sum((v for (kind, idx), v in components.items() if kind != "embedding" and idx in range(4, 8)), zero).double()
                    middle = upstream - embedding - early
                    gauge_rel = max(gauge_rel, float((middle - middle_explicit).norm() / middle_explicit.norm().clamp_min(1e-30)))
                    explicit8 = [upstream]
                    explicit8.extend(components[(kind, idx)].double() for idx in range(8, 11) for kind in ("attn", "mlp"))
                    mlp10 = x.double() - sum(explicit8[:-1])
                    gauge_rel = max(gauge_rel, float((mlp10 - explicit8[-1]).norm() / explicit8[-1].norm().clamp_min(1e-30)))
                    expanded = [embedding, early, middle, explicit8[2], mlp10]
                    aggregation_error = max(aggregation_error, float((sum(expanded[:3]) - upstream).abs().max()))
                    return x, x0, first, expanded
                attention, first = block.attn(state, first); x = x + attention; components[("attn", layer)] = attention
                mlp = block.mlp(F.rms_norm(x, (x.shape[-1],))); x = x + mlp; components[("mlp", layer)] = mlp
        raise RuntimeError("layer not reached")

    def readout(x, positions, selected):
        n = len(x); batch = torch.arange(n, device=device)
        logits = 30. * torch.tanh(model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.)
        return logits[batch[:, None], positions[:, None], selected].double()

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

    batch_ids = [[i for i, r in enumerate(rows) if r["template_id"] == template]
                 for template in ("behind_beside", "among_behind")]
    can_id = authority.old_task14.ENCODING.encode(" can")[0]; will_id = authority.old_task14.ENCODING.encode(" will")[0]
    for ids in batch_ids:
        batch_rows = [rows[i] for i in ids]; endpoints = [r["endpoints"]["recipient"] for r in batch_rows]
        n = len(ids); batch = torch.arange(n, device=device)
        tokens = torch.tensor([e["ids"] for e in endpoints], dtype=torch.long, device=device)
        positions = torch.tensor([r["subject_position"] for r in batch_rows], dtype=torch.long, device=device)
        selected = torch.tensor([[e["answer_id"], e["foil_id"], can_id, will_id] for e in endpoints], dtype=torch.long, device=device)
        with torch.no_grad(): base_input = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
        subject = base_input[batch, positions].double(); projection = subject @ unit; target_projection = threshold / float(decoder_axis.norm())
        orthogonal = subject - projection[:, None] * unit
        scale = torch.sqrt((subject.square().sum(1) - target_projection ** 2) / orthogonal.square().sum(1))
        removed_subject = target_projection * unit + scale[:, None] * orthogonal
        removed_input = base_input.clone(); removed_input[batch, positions] = removed_subject.float()
        raw_b, x0_b, first_b, groups_b = capture(base_input); _raw_r, _x0_r, _first_r, groups_r = capture(removed_input)
        deltas = [(groups_r[i] - groups_b[i])[batch, positions] for i in range(5)]; joint_delta = sum(deltas)
        values = {}
        for mask in range(32):
            edited = raw_b.clone(); delta = sum(deltas[i] for i in range(5) if mask & (1 << i)) if mask else torch.zeros_like(deltas[0])
            edited[batch, positions] += delta.float(); values[mask] = suffix(edited, x0_b, first_b, positions, selected)
        native_values = native(base_input, positions, selected)
        native_replay = max(native_replay, float((native_values - values[0]).abs().max()))
        native_correct[ids] = ((native_values[:, 0] - native_values[:, 1]) > 0).cpu().numpy()
        margins = {mask: values[mask][:, 0] - values[mask][:, 1] for mask in range(32)}; dividends = {}
        for mask in range(32):
            div = margins[mask].clone(); sub = (mask - 1) & mask
            while sub: div -= dividends[sub]; sub = (sub - 1) & mask
            if mask: div -= dividends[0]
            dividends[mask] = div
        batch_target = margins[0] - margins[31]; batch_closure = -sum(dividends[m] for m in range(1, 32))
        closure_abs = max(closure_abs, float((batch_closure - batch_target).abs().max()))
        closure_rel = max(closure_rel, float((batch_closure - batch_target).norm() / batch_target.norm().clamp_min(1e-30)))
        target[ids] = batch_target.cpu().numpy(); atoms[:, ids] = torch.stack([-dividends[m] for m in range(1, 32)]).cpu().numpy()
        control_change[ids] = ((values[31][:, 2] - values[31][:, 3]) - (values[0][:, 2] - values[0][:, 3])).cpu().numpy()
        target_norm = joint_delta.norm(dim=1)
        for ri, axis in enumerate(random_axes):
            direction = axis[None, :].expand(n, -1)
            direction = direction - ((direction * joint_delta).sum(1) / joint_delta.square().sum(1).clamp_min(1e-30))[:, None] * joint_delta
            random_delta = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-30) * target_norm[:, None]
            random_norm_error = max(random_norm_error, float(((random_delta.norm(dim=1) - target_norm).abs() / target_norm.clamp_min(1e-30)).max()))
            edited = raw_b.clone(); edited[batch, positions] += random_delta.float()
            random_values = suffix(edited, x0_b, first_b, positions, selected)
            random_damage[ri, ids] = ((values[0][:, 0] - values[0][:, 1]) - (random_values[:, 0] - random_values[:, 1])).cpu().numpy()

    selected_indices = [m - 1 for m in EXPECTED_MASKS]; prediction = atoms[selected_indices].sum(0)
    composition = {name: metric(prediction[ids], target[ids]) for name, ids in groups.items()}
    removal = {name: {"rows": len(ids), "damage_rms": float(np.sqrt(np.mean(target[ids] ** 2))),
                      "positive_fraction": float(np.mean(target[ids] > 0))} for name, ids in groups.items()}
    random_rms = np.sqrt(np.mean(random_damage ** 2, axis=1)); damage_rms = removal["overall"]["damage_rms"]
    control_rms = float(np.sqrt(np.mean(control_change ** 2)))
    finite = bool(np.isfinite(np.asarray([aggregation_error, gauge_rel, native_replay, random_norm_error,
                                          closure_abs, closure_rel, *target, *atoms.ravel(), *random_damage.ravel()])).all())
    pred_a = bool(finite and list(parent_result["graph"]["selected_masks"][:9]) == list(EXPECTED_MASKS)
        and aggregation_error <= BARS["maximum_aggregation_error"] and gauge_rel <= BARS["maximum_gauge_correction_relative_l2"]
        and native_replay <= BARS["maximum_native_replay_error"] and random_norm_error <= BARS["maximum_random_norm_error"]
        and closure_abs <= BARS["maximum_absolute_closure_error"] and closure_rel <= BARS["maximum_relative_closure_error"]
        and all(float(np.mean(native_correct[ids])) >= BARS["minimum_native_accuracy"] for ids in groups.values())
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    pred_b = bool(pred_a and all(p["relative_l2"] <= BARS["maximum_composition_relative_l2"]
        and p["cosine"] >= BARS["minimum_composition_cosine"] and p["aligned_recovery"] > BARS["minimum_aligned_recovery"]
        for p in composition.values()))
    pred_c = bool(pred_a and all(p["damage_rms"] >= BARS["minimum_panel_damage_rms"]
        and p["positive_fraction"] >= BARS["minimum_panel_damage_positive_fraction"] for p in removal.values()))
    pred_d = bool(pred_c and damage_rms / max(float(np.median(random_rms)), 1e-30) >= BARS["minimum_random_median_ratio"]
        and control_rms / max(damage_rms, 1e-30) <= BARS["maximum_control_fraction"])
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d)))
    terminal = "expanded_graph_fresh_ood_held" if all(predictions.values()) else "valid_expanded_graph_fresh_ood_test" if pred_a else "invalid"
    result = {"schema": "subject_number_l11h3_expanded_graph_fresh_ood_v1_result", "terminal": terminal,
        "predictions": predictions, "instrument": {"finite": finite, "native_replay_max_abs_error": native_replay,
            "expanded_to_parent_aggregation_max_abs_error": aggregation_error, "gauge_correction_max_relative_l2": gauge_rel,
            "random_edit_norm_max_relative_error": random_norm_error, "mobius_closure_max_abs_error": closure_abs,
            "mobius_closure_max_relative_l2": closure_rel,
            "native_capability": {name: float(np.mean(native_correct[ids])) for name, ids in groups.items()}, "counts": counts},
        "frozen_graph": {"ports": list(PORTS), "masks": list(EXPECTED_MASKS),
            "terms": [parent_result["graph"]["selected_terms"][i] for i in range(9)], "coefficient_fits": 0,
            "composition": composition}, "removal": removal,
        "equal_l2_same_site_controls": {"count": NULLS, "seed": SEED, "damage_rms": random_rms.tolist(),
            "median_damage_rms": float(np.median(random_rms)), "maximum_damage_rms": float(np.max(random_rms)),
            "target_to_median_ratio": damage_rms / max(float(np.median(random_rms)), 1e-30)},
        "collateral": {"can_will_change_rms": control_rms, "fraction_of_target_damage": control_rms / max(damage_rms, 1e-30)},
        "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical_sha256(rows), "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER), "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "wall_seconds": time.perf_counter() - started,
        "scope": "Frozen nine-edge graph on disjoint nouns, new two-attractor templates, longer sequence, and new subject position; no reselection or fitting."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "instrument", "frozen_graph", "removal", "equal_l2_same_site_controls", "collateral")}, indent=2, sort_keys=True))
    assert pred_a

if __name__ == "__main__": main()
