#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_census_closure pred_c_cut_grows_with_k pred_d_every_k_beats_null pred_e_pronoun_margin_stays_within_003
"""Dose-response of the agreement-axis population (v394). v393: the five leading units of MLPs 12-17 along VP0 cut the agreement-axis output 12% and leave the
pronoun margin alone. Here the census is redone with the lambda-chain scale (each layer's unit term scaled to the final residual, closing exactly), then the top
k = 1 / 5 / 10 / 20 units per layer (MLPs 12-17) are zeroed on the v76 rows; readouts the VP0 projection of the plural - singular final-state difference and
the they - he margin; 3 layer-matched random k-sets per k as null.
PREDICTIONS (scored as written; failures preserved; priors from v392 / v393)
    pred_a_baseline_replays         the unedited margin replays 2.048 within 1e-3
    pred_b_census_closure           the lambda-scaled per-unit terms sum to each MLP's final-token projection within relative 1e-3 on every row (the instrument v392 lacked)
    pred_c_cut_grows_with_k         the VP0 cut increases monotonically over k = 1, 5, 10, 20
    pred_d_every_k_beats_null       the cut exceeds 3x the largest random cut, every k
    pred_e_pronoun_margin_stays_within_003  |they - he margin change| <= 0.03 of native at every k (the agreement population is disjoint from the pronoun circuit)
PRICE (registered maximum): 3 census batches + 3 row batches x (1 baseline + 4 edits + 12 null) = 54 forwards + one SVD; 0 backwards; 0 fits. Bar <= 58.
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
OUT = ROOT / "circuits/followups/vp0_dose_response_v394_result.json"
CANDIDATE_ID = "subspace.vp0_dose_response_v394"
LATE = (12, 13, 14, 15, 16, 17); N_NULL, SEED, BATCH, KS = 3, 394, 32, (1, 5, 10, 20)
NATIVE_M, M_TOL, CENSUS_TOL, NULL_FACTOR, MARGIN_MAX = 2.0481, 1e-3, 1e-3, 3.0, 0.03
FORWARDS_MAX = 58
import run_mlp1_token_table_scaling_v287 as v287
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_census_closure": "<= 1e-3", "pred_c_cut_grows_with_k": "monotone", "pred_d_every_k_beats_null": "> 3x null x 4", "pred_e_pronoun_margin_stays_within_003": "<= 0.03 x 4"}


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
    rng = random.Random(SEED)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "late": list(LATE), "ks": list(KS), "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "census_tol": CENSUS_TOL, "null_factor": NULL_FACTOR, "margin_max": MARGIN_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F = backend.torch, backend.F; fw = L.ManualForward(backend); forwards = 0; model = backend.model; blocks = model.transformer.h; D = model.config.n_embd
    WU = model.lm_head.weight.detach().float().cpu(); pairs = [(a_, b_) for _, a_, b_ in v287.SPEC["noun_pairs_vocab"]][:256]; P = torch.tensor([b_ for _, b_ in pairs]); S = torch.tensor([a_ for a_, _ in pairs])
    UP = WU[P]; vp0 = torch.linalg.svd(UP - UP.mean(0), full_matrices=False).Vh[0]
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    lam = [float(b_.lambdas[0]) for b_ in blocks]
    # census pass: lambda-scaled per-unit terms of MLPs 12-17 at the final token along vp0
    H = {l: [] for l in LATE}; M = {l: [] for l in LATE}; XF = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pf = torch.tensor([r_.final for r_ in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; writes = {}
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in writes: writes[k] = block.lambdas[0] * writes[k]
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; xin = F.rms_norm(x, (D,)); m = block.mlp(xin); x = x + m
                if l in LATE: H[l].append(dod_units.hidden(model, block.mlp, xin)[idx, pf].float().cpu()); writes[l] = m.clone()
            for l in LATE: M[l].append(writes[l][idx, pf].float().cpu())
            XF.append(x[idx, pf].float().cpu()); forwards += 1
    H = {l: torch.cat(v) for l, v in H.items()}; M = {l: torch.cat(v) for l, v in M.items()}; census_clos = 0.0; top = {}
    for l in LATE:
        scale = 1.0
        for j in range(l + 1, 18): scale *= lam[j]
        Dw, b_ = blocks[l].mlp.Down.weight.detach().float().cpu(), blocks[l].mlp.Down_bias.detach().float().cpu(); unit = scale * H[l] * (Dw.T @ vp0).unsqueeze(0); full = M[l] @ vp0
        census_clos = max(census_clos, float(((unit.sum(1) + scale * float(b_ @ vp0) - full).abs() / full.abs().clamp_min(1e-6)).max()))
        pooled = (unit[plural] - unit[sing]).mean(0); top[l] = tuple(torch.argsort(pooled.abs(), descending=True)[:max(KS)].tolist())
    readers = {"they_he": (L._single(" they"), L._single(" he"))}
    def run(edits):
        marg, XFe = [], []
        for start in range(0, len(rows), BATCH):
            m_, lg_, xf_ = margins_multi(backend, fw, rows[start:start + BATCH], edits, readers); marg += m_; XFe.append(xf_)
        XFe = torch.cat(XFe); margin = sum((marg[i]["they_he"] if row.present else -marg[i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
        return margin, float(((XFe[plural] - XFe[sing]).mean(0)) @ vp0)
    m0, p0 = run(None); forwards += 3; per = {}
    for k in KS:
        m1, p1 = run({l: top[l][:k] for l in LATE}); forwards += 3; nulls = []
        for _ in range(N_NULL):
            rs = {l: tuple(rng.sample([j for j in range(4608) if j not in top[l]], k)) for l in LATE}; mn, pn_ = run(rs); forwards += 3; nulls.append(abs((pn_ - p0) / p0))
        per[str(k)] = {"vp0_cut_rel": (p0 - p1) / p0, "margin_change_rel": (m1 - m0) / m0, "null_cut_max": max(nulls)}
    cuts = [per[str(k)]["vp0_cut_rel"] for k in KS]
    report = {"native_margin": m0, "replay_gap": abs(m0 - NATIVE_M), "census_closure": census_clos, "vp0_projection_native": p0, "per_k": per, "top5_per_layer": {str(l): list(v[:5]) for l, v in top.items()}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_census_closure": census_clos <= CENSUS_TOL, "pred_c_cut_grows_with_k": all(cuts[i + 1] > cuts[i] for i in range(len(cuts) - 1)), "pred_d_every_k_beats_null": all(per[str(k)]["vp0_cut_rel"] > NULL_FACTOR * per[str(k)]["null_cut_max"] for k in KS),
                   "pred_e_pronoun_margin_stays_within_003": all(abs(per[str(k)]["margin_change_rel"]) <= MARGIN_MAX for k in KS)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "vp0_dose_response_result_v394", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
