#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_behavioral_mobius pred_b_one_correction_composition_ood pred_c_two_correction_composition_ood
"""Exact behavioral Möbius graph over the frozen three native ports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_late_writer_ood_mobius_v1 as graph
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as base
import run_subject_number_l11h3_three_port_behavior_v1 as parent
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = base.DECODER
PARENT_RESULT = POLY / "SUBJECT_NUMBER_L11H3_THREE_PORT_BEHAVIOR_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_L11H3_BEHAVIORAL_MOBIUS_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_L11H3_BEHAVIORAL_MOBIUS_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_L11H3_BEHAVIORAL_MOBIUS_V1_RESULT.json"
LAYER = 11; PORTS = parent.PORTS; GROUP_INDICES = parent.GROUP_INDICES
BARS = {"maximum_absolute_closure_error": 1e-10, "maximum_closed_reconstruction_error": 1e-10,
        "maximum_composition_relative_l2": .10, "maximum_gauge_correction_relative_l2": 1e-5,
        "maximum_native_replay_error": 1e-5, "maximum_relative_closure_error": 1e-10,
        "minimum_aligned_recovery": 0., "minimum_composition_cosine": .99,
        "minimum_native_accuracy": .75}
PRICE = {"partial_forwards": 4, "partial_sequences": 256, "native_forwards": 2,
         "native_sequences": 128, "suffix_forwards": 16, "suffix_sequences": 1024,
         "corners": 8, "mobius_terms": 7, "candidate_corrections": 4,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_behavioral_mobius": None,
    "pred_b_one_correction_composition_ood": None, "pred_c_two_correction_composition_ood": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "parent_runner": Path(parent.__file__),
             "parent_result": PARENT_RESULT, "decoder": DECODER, "preregistration": PREREG}
    if binding["files"] != {k: sha(v) for k, v in paths.items()} or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["ports"] != list(PORTS) \
            or binding["group_indices"] != list(GROUP_INDICES) or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, parent_result = (json.loads(p.read_text()) for p in (DECODER, PARENT_RESULT))
    if decoder["terminal"] != "embedding_number_decoder_frozen" or parent_result["terminal"] != "three_port_behavioral_graph" \
            or not all(parent_result["predictions"].values()):
        raise ValueError("parent status changed")
    return binding, decoder, parent_result, authority.build_rows()

def plan():
    _, decoder, parent_result, rows = load_bound(); panels = graph.panel_indices(rows)
    return {"schema": "subject_number_l11h3_behavioral_mobius_v1_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "rows": len(rows), "panel_sizes": {k: len(v) for k, v in panels.items()},
            "ports": list(PORTS), "mandatory_masks": [1, 2, 4], "candidate_correction_masks": [3, 5, 6, 7],
            "parent_terminal": parent_result["terminal"], "authority_sha256": authority.canonical(rows),
            "bars": BARS, "price": PRICE, "binding_sha256": sha(BINDING),
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"]}

def evaluate(target, atoms, selected, ids):
    y = target[ids]; prediction = atoms[selected][:, ids].sum(0); yn = np.linalg.norm(y); pn = np.linalg.norm(prediction)
    return {"rows": len(ids), "target_rms": float(np.sqrt(np.mean(y ** 2))),
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
            "relative_l2": float(np.linalg.norm(y - prediction) / max(yn, 1e-30)),
            "cosine": float(prediction @ y / max(pn * yn, 1e-30)),
            "aligned_recovery": float(prediction @ y / max(y @ y, 1e-30))}

def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(900); binding, decoder, parent_result, rows = load_bound(); panels = graph.panel_indices(rows)
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"], dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"]); unit = decoder_axis / decoder_axis.norm()
    counts = {"partial_forwards": 0, "partial_sequences": 0, "native_forwards": 0, "native_sequences": 0,
              "suffix_forwards": 0, "suffix_sequences": 0, "corners": 8, "mobius_terms": 7,
              "candidate_corrections": 4, "fits": 0, "backwards": 0, "parameter_updates": 0}
    reconstruction = correction_rel = native_replay = closure_abs = closure_rel = 0.
    atoms = np.zeros((7, len(rows)), dtype=np.float64); target = np.zeros(len(rows), dtype=np.float64); native_correct = np.zeros(len(rows), dtype=bool)

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
        raw_b, x0_b, first_b, groups_b = capture(base_input); raw_r, _x0_r, _first_r, groups_r = capture(removed_input)
        deltas = [(groups_r[i] - groups_b[i])[batch, positions] for i in GROUP_INDICES]
        values = {}
        for mask in range(8):
            edited = raw_b.clone(); delta = sum(deltas[i] for i in range(3) if mask & (1 << i)) if mask else torch.zeros_like(deltas[0])
            edited[batch, positions] += delta.float(); values[mask] = suffix(edited, x0_b, first_b, positions, answer_pairs)
        native_values = native(base_input, positions, answer_pairs)
        native_replay = max(native_replay, float((native_values - values[0]).abs().max()))
        native_correct[ids] = (native_values > 0).cpu().numpy()
        dividends = {}
        for mask in range(8):
            div = values[mask].clone(); sub = (mask - 1) & mask
            while sub: div -= dividends[sub]; sub = (sub - 1) & mask
            if mask: div -= dividends[0]
            dividends[mask] = div
        batch_target = values[0] - values[7]; batch_closure = -sum(dividends[m] for m in range(1, 8))
        closure_abs = max(closure_abs, float((batch_closure - batch_target).abs().max()))
        closure_rel = max(closure_rel, float((batch_closure - batch_target).norm() / batch_target.norm().clamp_min(1e-30)))
        target[ids] = batch_target.cpu().numpy()
        atoms[:, ids] = torch.stack([-dividends[m] for m in range(1, 8)]).cpu().numpy()

    names = ["*".join(PORTS[i] for i in range(3) if m & (1 << i)) for m in range(1, 8)]
    mains = [0, 1, 3]; candidates = [2, 4, 5, 6]; discovery = panels["discovery"]
    residual = target[discovery] - atoms[mains][:, discovery].sum(0); selected_corrections = []
    for _ in range(2):
        choice = min(candidates, key=lambda i: float(np.linalg.norm(residual - atoms[i, discovery])))
        selected_corrections.append(choice); candidates.remove(choice); residual -= atoms[choice, discovery]
    one = mains + selected_corrections[:1]; two = mains + selected_corrections
    reports = {"main_only": {name: evaluate(target, atoms, mains, ids) for name, ids in panels.items()},
               "one_correction": {name: evaluate(target, atoms, one, ids) for name, ids in panels.items()},
               "two_corrections": {name: evaluate(target, atoms, two, ids) for name, ids in panels.items()},
               "complete": {name: evaluate(target, atoms, list(range(7)), ids) for name, ids in panels.items()}}
    term_reports = [{"mask": i + 1, "name": names[i], "order": (i + 1).bit_count(),
                     "rms": float(np.sqrt(np.mean(atoms[i] ** 2))),
                     "aligned_recovery": float(atoms[i] @ target / max(target @ target, 1e-30))} for i in range(7)]
    finite = bool(np.isfinite(np.asarray([reconstruction, correction_rel, native_replay, closure_abs, closure_rel, *target, *atoms.ravel()])).all())
    pred_a = bool(finite and reconstruction <= BARS["maximum_closed_reconstruction_error"]
        and correction_rel <= BARS["maximum_gauge_correction_relative_l2"] and native_replay <= BARS["maximum_native_replay_error"]
        and closure_abs <= BARS["maximum_absolute_closure_error"] and closure_rel <= BARS["maximum_relative_closure_error"]
        and all(float(np.mean(native_correct[ids])) >= BARS["minimum_native_accuracy"] for ids in panels.values())
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    def passes(report):
        return all(p["relative_l2"] <= BARS["maximum_composition_relative_l2"]
                   and p["cosine"] >= BARS["minimum_composition_cosine"]
                   and p["aligned_recovery"] > BARS["minimum_aligned_recovery"] for p in report.values())
    pred_b = bool(pred_a and passes(reports["one_correction"])); pred_c = bool(pred_a and passes(reports["two_corrections"]))
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c)))
    terminal = "sparse_behavioral_mobius_graph" if pred_b or pred_c else "valid_behavioral_mobius_decomposition" if pred_a else "invalid"
    result = {"schema": "subject_number_l11h3_behavioral_mobius_v1_result", "terminal": terminal, "predictions": predictions,
        "instrument": {"finite": finite, "native_replay_max_abs_error": native_replay,
            "closed_reconstruction_max_abs_error": reconstruction, "gauge_correction_max_relative_l2": correction_rel,
            "mobius_closure_max_abs_error": closure_abs, "mobius_closure_max_relative_l2": closure_rel,
            "native_capability": {name: float(np.mean(native_correct[ids])) for name, ids in panels.items()}, "counts": counts},
        "graph": {"mandatory_main_masks": [1, 2, 4], "mandatory_main_terms": [names[i] for i in mains],
            "selected_correction_masks": [i + 1 for i in selected_corrections],
            "selected_correction_terms": [names[i] for i in selected_corrections], "coefficient_fits": 0,
            "selected_on": "discovery_only", "term_reports": term_reports, "panel_reports": reports},
        "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows), "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER), "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "wall_seconds": time.perf_counter() - started,
        "scope": "Exact eight-corner behavioral Mobius graph; correction edges selected on discovery and frozen for three OOD panels."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "instrument", "graph")}, indent=2, sort_keys=True))
    assert pred_a

if __name__ == "__main__": main()
