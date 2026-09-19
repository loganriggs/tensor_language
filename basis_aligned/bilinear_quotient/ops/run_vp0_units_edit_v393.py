#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_vp0_units_cut_the_agreement_axis pred_c_beats_random_sets pred_d_class_logit_contrast_falls pred_e_pronoun_margin_barely_moves
"""Edit of the agreement-axis writers (v393). v391 / v392: at the final token the plural-class agreement axis VP0 is written by MLPs 12-17, led by units 701 (MLP 17),
2483 (MLP 16), 3173 (MLP 15), 1330 (MLP 13), 1131 (MLP 12). Edits decide on the v76 rows: zero the five at every position; readouts (i) the plural - singular
final-state difference's projection on VP0, (ii) the class-logit contrast -- mean logit of the 256 plural nouns minus mean logit of their singulars at the final
token, plural rows minus singular rows -- (iii) the they - he margin; null: 12 layer-matched random 5-sets.
PREDICTIONS (scored as written; failures preserved; priors from v391 / v392)
    pred_a_baseline_replays            the unedited margin replays 2.048 within 1e-3
    pred_b_vp0_units_cut_the_agreement_axis  the five zeroed cut the final-state difference's VP0 projection by >= 0.20 relative
    pred_c_beats_random_sets           that cut exceeds 3x the largest |relative change| among the 12 random 5-sets
    pred_d_class_logit_contrast_falls  the plural-vs-singular class-logit contrast falls under the edit (the agreement axis feeds the noun-class logits)
    pred_e_pronoun_margin_barely_moves |they - he margin change| <= 0.03 of native (the two axes are written by disjoint units). Prior: unsure.
PRICE (registered maximum): 3 row batches x (1 baseline + 1 edit + 12 null) = 42 forwards + one SVD; 0 backwards; 0 fits. Bar <= 46.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/vp0_units_edit_v393_result.json"
CANDIDATE_ID = "subspace.vp0_units_edit_v393"
CHAIN = {12: (1131,), 13: (1330,), 15: (3173,), 16: (2483,), 17: (701,)}; N_NULL, SEED, BATCH = 12, 393, 32
NATIVE_M, M_TOL, CUT_MIN, NULL_FACTOR, MARGIN_MAX = 2.0481, 1e-3, 0.20, 3.0, 0.03
FORWARDS_MAX = 46
import run_mlp1_token_table_scaling_v287 as v287
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_vp0_units_cut_the_agreement_axis": ">= 0.20 relative", "pred_c_beats_random_sets": "> 3x null", "pred_d_class_logit_contrast_falls": "falls", "pred_e_pronoun_margin_barely_moves": "<= 0.03 of native"}


def margins_multi(backend, fw, rows, edits, readers):
    """they - he margins with `edits` = {layer: units} zeroed at every position (several layers in one forward)."""
    torch, F, model = backend.torch, backend.F, backend.model; tokens = fw._tokens(rows)
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if edits and l in edits:
                h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    pf = torch.tensor([row.final for row in rows], device=x.device); idx = torch.arange(len(rows), device=x.device)
    return [{name: float(logits[i, row.final, a] - logits[i, row.final, b]) for name, (a, b) in readers.items()} for i, row in enumerate(rows)], logits[idx, pf].float().cpu(), x[idx, pf].float().cpu()


def main() -> None:
    rows, he, she, agents, objects = g.build()
    rng = random.Random(SEED); layers = [12, 13, 15, 16, 17]; named = {u for v in CHAIN.values() for u in v}
    null_sets = [{l: (rng.choice([j for j in range(4608) if j not in named]),) for l in layers} for _ in range(N_NULL)]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": {str(k): list(v) for k, v in CHAIN.items()}, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "cut_min": CUT_MIN, "null_factor": NULL_FACTOR, "margin_max": MARGIN_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; fw = L.ManualForward(backend); forwards = 0; model = backend.model
    WU = model.lm_head.weight.detach().float().cpu(); pairs = [(a_, b_) for _, a_, b_ in v287.SPEC["noun_pairs_vocab"]][:256]; P = torch.tensor([b_ for _, b_ in pairs]); S = torch.tensor([a_ for a_, _ in pairs])
    UP = WU[P]; vp0 = torch.linalg.svd(UP - UP.mean(0), full_matrices=False).Vh[0]
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    readers = {"they_he": (L._single(" they"), L._single(" he"))}
    def run(edits):
        marg, LG, XF = [], [], []
        for start in range(0, len(rows), BATCH):
            m_, lg_, xf_ = margins_multi(backend, fw, rows[start:start + BATCH], edits, readers); marg += m_; LG.append(lg_); XF.append(xf_)
        LG, XF = torch.cat(LG), torch.cat(XF); margin = sum((marg[i]["they_he"] if row.present else -marg[i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
        diff = (XF[plural] - XF[sing]).mean(0); proj = float(diff @ vp0); cls = float(((LG[:, P].mean(1) - LG[:, S].mean(1))[plural] - (LG[:, P].mean(1) - LG[:, S].mean(1))[sing]).mean())
        return margin, proj, cls
    m0, p0, c0 = run(None); forwards += 3; m1, p1, c1 = run(CHAIN); forwards += 3; nulls = []
    for ns in null_sets:
        mn, pn_, cn = run(ns); forwards += 3; nulls.append({"margin": mn - m0, "vp0_rel": (pn_ - p0) / p0, "class": cn - c0})
    report = {"native_margin": m0, "replay_gap": abs(m0 - NATIVE_M), "vp0_projection_native": p0, "vp0_projection_edit": p1, "vp0_rel_change": (p1 - p0) / p0, "class_logit_contrast_native": c0, "class_logit_contrast_edit": c1, "margin_change_rel": (m1 - m0) / m0,
              "null_vp0_rel_max_abs": max(abs(n["vp0_rel"]) for n in nulls), "null_class_change_max_abs": max(abs(n["class"]) for n in nulls), "null_margin_max_abs": max(abs(n["margin"]) for n in nulls)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_vp0_units_cut_the_agreement_axis": (p0 - p1) / p0 >= CUT_MIN, "pred_c_beats_random_sets": abs((p1 - p0) / p0) > NULL_FACTOR * report["null_vp0_rel_max_abs"], "pred_d_class_logit_contrast_falls": c1 < c0, "pred_e_pronoun_margin_barely_moves": abs(report["margin_change_rel"]) <= MARGIN_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "vp0_units_edit_result_v393", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
