#!/usr/bin/env python3
"""Prospective OOD, activation-noise, and output-KL red-team of frozen response projectors."""
# BQGATE: EXPERIMENT pred_a_authority_hashes_disjointness_finiteness_and_price pred_b_both_frozen_projectors_transfer_cleanly pred_c_transfer_survives_activation_noise pred_d_controls_are_output_inert pred_e_crossfit_effects_agree_ood
from datetime import datetime, timezone
import hashlib, json, math, os, statistics, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as fresh_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as fresh_i
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_frozen_response_ood_regularization_v1.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TB = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
TC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
IB = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
IC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
GENERIC = ROOT / "circuits/followups/temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1_result.json"
CONTROL = ROOT / "circuits/followups/temporal_five_mlp_matched_control_upstream_atlas_v2_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_frozen_response_ood_regularization_v1_result.json"
EXPECTED = {
    "prior": "e83cf0ccd8260558c04ba92cbf30960201bb33e0a2a5c8c1027b1bc2b68d8a2f",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "temporal_builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "temporal_capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "iswas_builder": "fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec",
    "iswas_capability": "6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
    "generic": "57402478b86e88237bb745824e7aa8e6d56c17336753cee3d1b4e9359b5febe3",
    "control": "b75a4b81b8d2407ea3661f21be8b4a3bf967bc8a242fc08d562aa6579b97bb69",
}
NOISE_FRACTION, NOISE_SEEDS = 0.10, (1701, 1702, 1703)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def run_project(backend, batch, base, donor, bases, *, noise_fraction=0.0, seed=0):
    """Apply a frozen rank-8 projector, optionally after sitewise relative Gaussian delta jitter."""
    torch = backend.torch
    width = backend.model.config.n_embd // backend.model.config.n_head
    gen = torch.Generator(device=backend.device).manual_seed(seed)
    handles = []

    def jitter(delta):
        if noise_fraction == 0:
            return delta
        scale = delta.float().square().mean().sqrt().clamp_min(1e-12) * noise_fraction
        return delta + torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=gen) * scale

    for site in comp.SITES:
        kind, layer, head = atlasrun.site_parts(site)
        q = bases[site][:, :8]
        if kind == "attn":
            b, d = base["attention"][layer], donor["attention"][layer]
            def hook(_m, args, b=b, d=d, q=q, head=head):
                x = args[0].clone(); a, z = head * width, (head + 1) * width
                for i, pos in enumerate(batch.semantic_positions):
                    n = int(pos) + 1
                    delta = jitter((d[i, :n, a:z] - b[i, :n, a:z]).to(x).float())
                    x[i, :n, a:z] = b[i, :n, a:z].to(x) + ((delta @ q) @ q.T).to(x)
                return (x,) + tuple(args[1:])
            handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
        else:
            b, d = base["mlp"][layer], donor["mlp"][layer]
            def hook(_m, _a, out, b=b, d=d, q=q):
                x = out.clone()
                for i, pos in enumerate(batch.semantic_positions):
                    n = int(pos) + 1
                    delta = jitter((d[i, :n] - b[i, :n]).to(x).float())
                    x[i, :n] = b[i, :n].to(x) + ((delta @ q) @ q.T).to(x)
                return x
            handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(hook))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles: handle.remove()


def target_report(backend, rows, base_state, full_state, state, reader, temporal_n):
    torch = backend.torch
    full_behavior = comp.margins(backend, full_state, rows) - comp.margins(backend, base_state, rows)
    behavior = comp.margins(backend, state, rows) - comp.margins(backend, base_state, rows)
    full_modes, modes = (full_state - base_state) @ reader, (state - base_state) @ reader
    task_indices = {"temporal": slice(0, temporal_n), "iswas": slice(temporal_n, len(rows))}
    cells, projection = {}, {}
    for task, ix in task_indices.items():
        cells[f"{task}_behavior"] = float((behavior[ix] - full_behavior[ix]).square().sum() / full_behavior[ix].square().sum().clamp_min(1e-30))
        projection[task] = float(behavior[ix] @ full_behavior[ix] / full_behavior[ix].square().sum().clamp_min(1e-30))
        for mode in range(2):
            cells[f"{task}_mode{mode + 1}"] = float((modes[ix, mode] - full_modes[ix, mode]).square().sum() / full_modes[ix, mode].square().sum().clamp_min(1e-30))
    return {"worst_target_residual": max(cells.values()), "cells": cells, "behavior_signed_projection": projection}


def control_report(backend, rows, base_state, state, scales):
    torch, F = backend.torch, backend.F
    delta = comp.margins(backend, state, rows) - comp.margins(backend, base_state, rows)
    names = ["temporal" if r["task_id"] == matched.temporal.TASK_ID else "iswas" for r in rows]
    fractions = {}
    for name in ("temporal", "iswas"):
        ix = torch.as_tensor([j for j, x in enumerate(names) if x == name], device=backend.device)
        fractions[name] = float(delta[ix].square().mean().sqrt()) / scales[name]
    lb, lp = das.head_logits(backend, base_state).float(), das.head_logits(backend, state).float()
    logb, logp = F.log_softmax(lb, -1), F.log_softmax(lp, -1)
    kl = (logb.exp() * (logb - logp)).sum(-1)
    return {"margin_rms_fraction": fractions, "median_full_vocab_kl_nats": float(kl.median()),
            "max_full_vocab_kl_nats": float(kl.max()), "top1_flip_fraction": float((lb.argmax(-1) != lp.argmax(-1)).float().mean())}


def main():
    paths = {"prior": PRIOR, "interface": INTERFACE, "temporal_builder": TB,
             "temporal_capability": TC, "iswas_builder": IB, "iswas_capability": IC,
             "generic": GENERIC, "control": CONTROL}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("frozen OOD authority changed")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_frozen_response_ood_regularization_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 27, "fit_updates": 0, "model_updates": 0,
           "transformer_backwards": 0, "noise_fraction": NOISE_FRACTION, "noise_seeds": NOISE_SEEDS}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)

    iface = json.loads(INTERFACE.read_text())
    tcap, icap = json.loads(TC.read_text()), json.loads(IC.read_text())
    fresh_tr = sum((population.capable_rows(fresh_t, tcap, panel, 12) for panel in ("A1", "A2")), [])
    fresh_ir = sum((population.capable_rows(fresh_i, icap, panel, 12) for panel in ("A1", "A2")), [])
    fresh_targets = fresh_tr + fresh_ir
    fresh_p = [r for r in fresh_t.build_rows() if r["transform_id"] == "P"][:16]
    controls = matched.control_rows()
    old_validation = controls["validation"]

    old_tcap, old_icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text())
    old_tr, old_ir, *_ = greedy.rows_and_controls(old_tcap, old_icap)
    target_even, target_odd = old_tr[::2] + old_ir[::2], old_tr[1::2] + old_ir[1::2]
    control_even, control_odd = controls["discovery"], controls["validation"]
    fit_ids = {r["row_id"] for r in target_even + target_odd}
    authority_ok = (iface["terminal"] == "stable_weight_readable_response_program"
                    and len(fresh_targets) == 48 and len(fresh_p) == 16
                    and not fit_ids.intersection(r["row_id"] for r in fresh_targets)
                    and all(r["base_semantic_position"] == r["donor_semantic_position"] for r in fresh_targets + fresh_p)
                    and all(r["base_answer_id"] == r["donor_answer_id"] for r in fresh_p))
    if not authority_ok: raise RuntimeError("fresh population or disjointness changed")

    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    ceb, _, ce0, _, ce1 = v1.cap(backend, control_even)
    cob, _, co0, _, co1 = v1.cap(backend, control_odd)
    teb, _, te0, _, te1 = v1.cap(backend, target_even)
    tob, _, to0, _, to1 = v1.cap(backend, target_odd)
    cbe, _ = comp.fit_bases(backend, ceb, ce0, ce1); cbo, _ = comp.fit_bases(backend, cob, co0, co1)
    qe, _ = v1.fit_target(backend, teb, te0, te1, cbe); qo, _ = v1.fit_target(backend, tob, to0, to1, cbo)
    projectors = {"even_fit": qe, "odd_fit": qo}
    hash_replay = {site: [interface.thash(qe[site]), interface.thash(qo[site])] for site in comp.SITES}
    hashes_ok = all(hash_replay[site] == iface["records"][site]["fit_basis_sha256"] for site in comp.SITES)

    ftb, ftbo, ft0, _, ft1 = v1.cap(backend, fresh_targets)
    fpb, fpbo, fp0, _, fp1 = v1.cap(backend, fresh_p)
    ovb, ovbo, ov0, _, ov1 = v1.cap(backend, old_validation)
    base_target = atlasrun.states(torch, backend, ftbo, fresh_targets)
    full_target_out, _ = atlasrun.run_patch(backend, ftb, ft1, comp.SITES)
    full_target = atlasrun.states(torch, backend, full_target_out, fresh_targets)
    base_fp = atlasrun.states(torch, backend, fpbo, fresh_p)
    base_ov = atlasrun.states(torch, backend, ovbo, old_validation)
    scales = json.loads(GENERIC.read_text())["target_behavior_rms_scales"]
    reports = {}
    for label, bases in projectors.items():
        clean = run_project(backend, ftb, ft0, ft1, bases)
        clean_state = atlasrun.states(torch, backend, clean, fresh_targets)
        target = target_report(backend, fresh_targets, base_target, full_target, clean_state, reader, len(fresh_tr))
        fp = run_project(backend, fpb, fp0, fp1, bases)
        ov = run_project(backend, ovb, ov0, ov1, bases)
        combined_rows = fresh_p + old_validation
        combined_base = torch.cat([base_fp, base_ov])
        combined_state = torch.cat([atlasrun.states(torch, backend, fp, fresh_p), atlasrun.states(torch, backend, ov, old_validation)])
        control = control_report(backend, combined_rows, combined_base, combined_state, scales)
        noisy = []
        for seed in NOISE_SEEDS:
            out = run_project(backend, ftb, ft0, ft1, bases, noise_fraction=NOISE_FRACTION, seed=seed)
            noisy.append({"seed": seed, **target_report(backend, fresh_targets, base_target, full_target,
                          atlasrun.states(torch, backend, out, fresh_targets), reader, len(fresh_tr))})
        reports[label] = {"clean_target": target, "control": control, "noisy_target": noisy}

    finite_values = []
    for item in reports.values():
        finite_values += list(item["clean_target"]["cells"].values()) + list(item["clean_target"]["behavior_signed_projection"].values())
        finite_values += list(item["control"]["margin_rms_fraction"].values()) + [item["control"]["median_full_vocab_kl_nats"], item["control"]["max_full_vocab_kl_nats"], item["control"]["top1_flip_fraction"]]
        for noisy in item["noisy_target"]: finite_values += list(noisy["cells"].values()) + list(noisy["behavior_signed_projection"].values())
    pa = authority_ok and hashes_ok and reader_ok and orientation <= 1e-6 and all(math.isfinite(x) for x in finite_values)
    pb = all(x["clean_target"]["worst_target_residual"] <= .15 and min(x["clean_target"]["behavior_signed_projection"].values()) >= .70 for x in reports.values())
    pc = all(n["worst_target_residual"] <= .25 and min(n["behavior_signed_projection"].values()) >= .60 for x in reports.values() for n in x["noisy_target"])
    pd = all(max(x["control"]["margin_rms_fraction"].values()) <= .10 and x["control"]["median_full_vocab_kl_nats"] <= .02 and x["control"]["top1_flip_fraction"] <= .05 for x in reports.values())
    pe = all(abs(reports["even_fit"]["clean_target"]["behavior_signed_projection"][task] - reports["odd_fit"]["clean_target"]["behavior_signed_projection"][task]) <= .15 for task in ("temporal", "iswas"))
    predictions = {"pred_a_authority_hashes_disjointness_finiteness_and_price": bool(pa),
                   "pred_b_both_frozen_projectors_transfer_cleanly": bool(pb),
                   "pred_c_transfer_survives_activation_noise": bool(pc),
                   "pred_d_controls_are_output_inert": bool(pd),
                   "pred_e_crossfit_effects_agree_ood": bool(pe)}
    terminal = "invalid" if not pa else "stable_reusable_response_program" if all(predictions.values()) else "memorization_fragility_null" if pb and not (pc and pd) else "ood_transfer_null"
    summary = {label: {"clean_worst": x["clean_target"]["worst_target_residual"],
                       "clean_projection": x["clean_target"]["behavior_signed_projection"],
                       "noisy_worst_max": max(n["worst_target_residual"] for n in x["noisy_target"]),
                       "noisy_projection_min": min(min(n["behavior_signed_projection"].values()) for n in x["noisy_target"]),
                       "control_margin_max": max(x["control"]["margin_rms_fraction"].values()),
                       "control_median_kl": x["control"]["median_full_vocab_kl_nats"],
                       "control_top1_flip": x["control"]["top1_flip_fraction"]} for label, x in reports.items()}
    result = {"schema": "temporal_five_mlp_frozen_response_ood_regularization_result_v1",
              "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter() - tic,
              "authority_sha256": EXPECTED, "projector_hash_replay": hash_replay,
              "noise_fraction": NOISE_FRACTION, "noise_seeds": NOISE_SEEDS, "reports": reports,
              "summary": summary, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 27, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
