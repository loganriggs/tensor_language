#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_expanded_instrument pred_b_six_term_composition_ood pred_c_eight_term_composition_ood
"""Open upstream_0_7 into three bands and fit an OOD sparse behavioral graph."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_behavioral_mobius_v1 as parent
import run_subject_number_l11h3_late_writer_ood_mobius_v1 as graph
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as base
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = base.DECODER
PARENT_RESULT = POLY / "SUBJECT_NUMBER_L11H3_BEHAVIORAL_MOBIUS_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V1_RESULT.json"
LAYER, STEPS = 11, 8
PORTS = ("embedding_recurrence", "early_writes_0_3", "middle_writes_4_7", "mlp_8", "mlp_10")
BARS = {"maximum_absolute_closure_error": 1e-10, "maximum_aggregation_error": 1e-10,
        "maximum_composition_relative_l2": .10, "maximum_gauge_correction_relative_l2": 1e-5,
        "maximum_native_replay_error": 1e-5, "maximum_relative_closure_error": 1e-10,
        "minimum_aligned_recovery": 0., "minimum_composition_cosine": .99,
        "minimum_native_accuracy": .75}
PRICE = {"partial_forwards": 4, "partial_sequences": 256, "native_forwards": 2,
         "native_sequences": 128, "suffix_forwards": 64, "suffix_sequences": 4096,
         "corners": 32, "mobius_terms": 31, "candidate_terms": 15, "greedy_steps": 8,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_expanded_instrument": None,
    "pred_b_six_term_composition_ood": None, "pred_c_eight_term_composition_ood": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "parent_runner": Path(parent.__file__),
             "parent_result": PARENT_RESULT, "decoder": DECODER, "preregistration": PREREG}
    if binding["files"] != {k: sha(v) for k, v in paths.items()} or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["ports"] != list(PORTS) or binding["steps"] != STEPS or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, parent_result = (json.loads(p.read_text()) for p in (DECODER, PARENT_RESULT))
    if decoder["terminal"] != "embedding_number_decoder_frozen" or parent_result["terminal"] != "sparse_behavioral_mobius_graph" \
            or not parent_result["predictions"]["pred_c_two_correction_composition_ood"]:
        raise ValueError("parent status changed")
    return binding, decoder, parent_result, authority.build_rows()

def plan():
    _, decoder, parent_result, rows = load_bound(); panels = graph.panel_indices(rows)
    return {"schema": "subject_number_l11h3_expanded_behavioral_graph_v1_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "rows": len(rows), "panel_sizes": {k: len(v) for k, v in panels.items()},
            "ports": list(PORTS), "eligible_orders": [1, 2], "steps": STEPS, "parent_terminal": parent_result["terminal"],
            "authority_sha256": authority.canonical(rows), "bars": BARS, "price": PRICE,
            "binding_sha256": sha(BINDING), "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"]}

def evaluate(target, atoms, selected, ids):
    y = target[ids]; prediction = atoms[selected][:, ids].sum(0); yn = np.linalg.norm(y); pn = np.linalg.norm(prediction)
    return {"rows": len(ids), "target_rms": float(np.sqrt(np.mean(y ** 2))),
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
            "relative_l2": float(np.linalg.norm(y - prediction) / max(yn, 1e-30)),
            "cosine": float(prediction @ y / max(pn * yn, 1e-30)),
            "aligned_recovery": float(prediction @ y / max(y @ y, 1e-30))}

def term_rows(atoms, ids):
    out = atoms[:, ids]
    return out.T if out.shape[0] == len(ids) else out

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
    counts = {"partial_forwards": 0, "partial_sequences": 0, "native_forwards": 0, "native_sequences": 0,
              "suffix_forwards": 0, "suffix_sequences": 0, "corners": 32, "mobius_terms": 31,
              "candidate_terms": 15, "greedy_steps": STEPS, "fits": 0, "backwards": 0, "parameter_updates": 0}
    aggregation_error = gauge_rel = native_replay = closure_abs = closure_rel = 0.
    atoms = np.zeros((31, len(rows)), dtype=np.float64); target = np.zeros(len(rows), dtype=np.float64); native_correct = np.zeros(len(rows), dtype=bool)

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

    def margin(x, positions, answer_pairs):
        n = len(x); batch = torch.arange(n, device=device)
        logits = 30. * torch.tanh(model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30.)
        selected = logits[batch[:, None], positions[:, None], answer_pairs]
        return (selected[:, 0] - selected[:, 1]).double()

    def native(initial, positions, answer_pairs):
        counts["native_forwards"] += 1; counts["native_sequences"] += len(initial)
        with torch.no_grad():
            x = initial; x0 = initial; first = None
            for block in model.transformer.h:
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first); x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            return margin(x, positions, answer_pairs)

    def suffix(raw, x0, first, positions, answer_pairs):
        counts["suffix_forwards"] += 1; counts["suffix_sequences"] += len(raw)
        with torch.no_grad():
            x = raw
            for layer in range(LAYER, len(model.transformer.h)):
                block = model.transformer.h[layer]
                if layer > LAYER: x = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, first = block.attn(F.rms_norm(x, (x.shape[-1],)), first); x = x + attention
                x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            return margin(x, positions, answer_pairs)

    batch_ids = [[i for i, r in enumerate(rows) if r["template_id"] in names]
                 for names in (("near", "behind"), ("under", "above"))]
    answer_ids = {"singular": authority.task14.ENCODING.encode(" is")[0], "plural": authority.task14.ENCODING.encode(" are")[0]}
    for ids in batch_ids:
        batch_rows = [rows[i] for i in ids]; n = len(ids); batch = torch.arange(n, device=device)
        tokens = torch.tensor([r["token_ids"] for r in batch_rows], dtype=torch.long, device=device)
        positions = torch.tensor([r["subject_position"] for r in batch_rows], dtype=torch.long, device=device)
        answer_pairs = torch.tensor([[r["native_answer_id"], answer_ids["plural" if r["number"] == "singular" else "singular"]]
                                     for r in batch_rows], dtype=torch.long, device=device)
        with torch.no_grad(): base_input = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
        subject = base_input[batch, positions].double(); projection = subject @ unit; target_projection = threshold / float(decoder_axis.norm())
        orthogonal = subject - projection[:, None] * unit
        scale = torch.sqrt((subject.square().sum(1) - target_projection ** 2) / orthogonal.square().sum(1))
        removed_subject = target_projection * unit + scale[:, None] * orthogonal
        removed_input = base_input.clone(); removed_input[batch, positions] = removed_subject.float()
        raw_b, x0_b, first_b, groups_b = capture(base_input); _raw_r, _x0_r, _first_r, groups_r = capture(removed_input)
        deltas = [(groups_r[i] - groups_b[i])[batch, positions] for i in range(5)]; values = {}
        for mask in range(32):
            edited = raw_b.clone(); delta = sum(deltas[i] for i in range(5) if mask & (1 << i)) if mask else torch.zeros_like(deltas[0])
            edited[batch, positions] += delta.float(); values[mask] = suffix(edited, x0_b, first_b, positions, answer_pairs)
        native_values = native(base_input, positions, answer_pairs)
        native_replay = max(native_replay, float((native_values - values[0]).abs().max())); native_correct[ids] = (native_values > 0).cpu().numpy()
        dividends = {}
        for mask in range(32):
            div = values[mask].clone(); sub = (mask - 1) & mask
            while sub: div -= dividends[sub]; sub = (sub - 1) & mask
            if mask: div -= dividends[0]
            dividends[mask] = div
        batch_target = values[0] - values[31]; batch_closure = -sum(dividends[m] for m in range(1, 32))
        closure_abs = max(closure_abs, float((batch_closure - batch_target).abs().max()))
        closure_rel = max(closure_rel, float((batch_closure - batch_target).norm() / batch_target.norm().clamp_min(1e-30)))
        target[ids] = batch_target.cpu().numpy(); atoms[:, ids] = torch.stack([-dividends[m] for m in range(1, 32)]).cpu().numpy()

    names = ["*".join(PORTS[i] for i in range(5) if m & (1 << i)) for m in range(1, 32)]
    candidates = [m - 1 for m in range(1, 32) if m.bit_count() <= 2]; discovery = panels["discovery"]
    residual = target[discovery].copy(); available = list(candidates); selected = []; curve = []
    discovery_atoms = term_rows(atoms, discovery); denom = max(float(np.linalg.norm(target[discovery])), 1e-30)
    for _ in range(STEPS):
        choice = min(available, key=lambda i: float(np.linalg.norm(residual - discovery_atoms[i])))
        residual -= discovery_atoms[choice]; available.remove(choice); selected.append(choice); curve.append(float(np.linalg.norm(residual) / denom))
    prefix_reports = {str(k): {name: evaluate(target, atoms, selected[:k], ids) for name, ids in panels.items()} for k in range(1, STEPS + 1)}
    complete_reports = {name: evaluate(target, atoms, list(range(31)), ids) for name, ids in panels.items()}
    term_reports = [{"mask": i + 1, "name": names[i], "order": (i + 1).bit_count(),
                     "rms": float(np.sqrt(np.mean(atoms[i] ** 2))),
                     "aligned_recovery": float(atoms[i] @ target / max(target @ target, 1e-30))} for i in range(31)]
    finite = bool(np.isfinite(np.asarray([aggregation_error, gauge_rel, native_replay, closure_abs, closure_rel, *target, *atoms.ravel()])).all())
    pred_a = bool(finite and aggregation_error <= BARS["maximum_aggregation_error"]
        and gauge_rel <= BARS["maximum_gauge_correction_relative_l2"] and native_replay <= BARS["maximum_native_replay_error"]
        and closure_abs <= BARS["maximum_absolute_closure_error"] and closure_rel <= BARS["maximum_relative_closure_error"]
        and all(float(np.mean(native_correct[ids])) >= BARS["minimum_native_accuracy"] for ids in panels.values())
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    def passes(report):
        return all(p["relative_l2"] <= BARS["maximum_composition_relative_l2"]
                   and p["cosine"] >= BARS["minimum_composition_cosine"]
                   and p["aligned_recovery"] > BARS["minimum_aligned_recovery"] for p in report.values())
    pred_b = bool(pred_a and passes(prefix_reports["6"])); pred_c = bool(pred_a and passes(prefix_reports["8"]))
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c)))
    terminal = "expanded_sparse_behavioral_graph" if pred_b or pred_c else "valid_expanded_behavioral_decomposition" if pred_a else "invalid"
    result = {"schema": "subject_number_l11h3_expanded_behavioral_graph_v1_result", "terminal": terminal, "predictions": predictions,
        "instrument": {"finite": finite, "native_replay_max_abs_error": native_replay,
            "expanded_to_parent_aggregation_max_abs_error": aggregation_error, "gauge_correction_max_relative_l2": gauge_rel,
            "mobius_closure_max_abs_error": closure_abs, "mobius_closure_max_relative_l2": closure_rel,
            "native_capability": {name: float(np.mean(native_correct[ids])) for name, ids in panels.items()}, "counts": counts},
        "graph": {"ports": list(PORTS), "eligible_orders": [1, 2], "candidate_count": len(candidates),
            "selected_masks": [i + 1 for i in selected], "selected_terms": [names[i] for i in selected],
            "discovery_curve": curve, "coefficient_fits": 0, "selected_on": "discovery_only",
            "prefix_panels": prefix_reports, "complete_panels": complete_reports, "term_reports": term_reports},
        "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows), "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER), "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "wall_seconds": time.perf_counter() - started,
        "scope": "Exact five-port expansion of upstream_0_7; main/pair terms selected on discovery and frozen across three OOD panels."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "instrument", "graph")}, indent=2, sort_keys=True))
    assert pred_a

if __name__ == "__main__": main()
