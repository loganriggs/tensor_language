#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_frames_replay pred_b_114_leads_the_mirror_census pred_c_mirror_population_is_spread pred_d_mirror_edit_moves_the_prediction pred_e_mirror_edit_lowers_singular_preference
"""The mirror violation: "The X is" with plural X (v354). v351 named the MLP-3 population flagging "singular X + are" (3040, 114, 565 lead; zeroing the top ten
costs 0.094 nats and 0.73 log-odds of plural continuation). v352: 114 fires +1098 for the mirror "plural X + is". Same census and population edit on the
mirror cell: write-weighted median contrast "plural X + is" vs "plural X + are" across 4,608 units; zero the top 10 / 50 / 200 on the mirror cell; KL and the
change in the singular-vs-plural continuation log-odds; 3 random sets per k.
PREDICTIONS (scored as written; failures preserved; priors from v351 / v352)
    pred_a_baseline_frames_replay        the native pass reproduces v349's 3465 medians for the two plural cells (is|p -87.8, are|p -49.0) within 5%
    pred_b_114_leads_the_mirror_census   114 ranks first by |contrast| . ||Down_j|| on the mirror cell
    pred_c_mirror_population_is_spread   the top 50 carry <= 0.60 of the summed |contrast| . ||Down_j||
    pred_d_mirror_edit_moves_the_prediction  KL under the top-10 zero-edit on the mirror cell >= 0.03 nats and > 3x the largest random 10-set
    pred_e_mirror_edit_lowers_singular_preference  the top-10 edit lowers the singular-vs-plural continuation log-odds on the mirror cell (the flag lets the model follow the singular verb over the plural noun). Prior: unsure.
PRICE (registered maximum): 2 native cells + mirror cell x (3 edits + 9 nulls) = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mirror_violation_population_v354_result.json"
CANDIDATE_ID = "pronoun_number.mirror_violation_population_v354"
LAYER, KS, N_NULL, SEED = 3, (10, 50, 200), 3, 351
PLURAL = (" they", " them", " these", " those", " all", " both"); SINGULAR = (" it", " him", " her", " this", " that", " one")
V349 = {"is|p": -87.8, "are|p": -49.0}; REPLAY, TOP50_MAX, KL_MIN, NULL_FACTOR = 0.05, 0.60, 0.03, 3.0
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_baseline_frames_replay": "within 5%", "pred_b_114_leads_the_mirror_census": "rank 1", "pred_c_mirror_population_is_spread": "top-50 <= 0.60", "pred_d_mirror_edit_moves_the_prediction": ">= 0.03 nats, > 3x null", "pred_e_mirror_edit_lowers_singular_preference": "singular log-odds fall"}


def run(backend, ids, units):
    """full forward with the given MLP-3 units zeroed at the last position; returns log-probs at the last position and unit 3465 / 493 activations there."""
    torch, F, model = backend.torch, backend.F, backend.model
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None; acts = None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == LAYER:
                h = dod_units.hidden(model, block.mlp, xin); acts = h[:, -1].float().cpu()
                if units: h[:, -1, torch.tensor(list(units), device=h.device)] = 0
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else: m = block.mlp(xin)
            x = x + m
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return torch.log_softmax(logits[:, -1].float(), -1).cpu(), acts


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    the, is_, are = L._single("The"), L._single(" is"), L._single(" are"); pl = [L._single(t) for t in PLURAL]; sg = [L._single(t) for t in SINGULAR]
    cells = {"is|s": [[the, a, is_] for a, _ in pairs], "is|p": [[the, b, is_] for _, b in pairs], "are|s": [[the, a, are] for a, _ in pairs], "are|p": [[the, b, are] for _, b in pairs]}
    rng = random.Random(SEED); pool = [j for j in range(4608) if j not in (3465, 493)]; nulls = [(rng.choice(pool),) for _ in range(N_NULL)]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "cells": list(cells), "ks": list(KS), "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay": REPLAY, "top50_max": TOP50_MAX, "kl_min": KL_MIN, "null_factor": NULL_FACTOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; forwards = 0
    Dn = backend.model.transformer.h[LAYER].mlp.Down.weight.detach().float().cpu().norm(dim=0)
    lpv, actv = run(backend, torch.tensor(cells["is|p"], device="cuda"), ()); forwards += 1; lpg, actg = run(backend, torch.tensor(cells["are|p"], device="cuda"), ()); forwards += 1
    med3465 = {"is|p": float(actv[:, 3465].median()), "are|p": float(actg[:, 3465].median())}; replay = all(abs(med3465[c] - V349[c]) <= REPLAY * abs(V349[c]) for c in V349)
    contrast = (actv.median(0).values - actg.median(0).values) * Dn; order = torch.argsort(contrast.abs(), descending=True); rank114 = int((contrast.abs() > contrast.abs()[114]).sum()) + 1
    shares = {str(k): float(contrast.abs()[order[:k]].sum() / contrast.abs().sum()) for k in (10, 50, 200, 500)}
    rng = random.Random(SEED); pool = list(range(4608)); ids = torch.tensor(cells["is|p"], device="cuda")
    def kl_lo(lp_):
        return float((lpv.exp() * (lpv - lp_)).sum(-1).mean()), float((torch.logsumexp(lp_[:, pl], -1) - torch.logsumexp(lp_[:, sg], -1)).mean())
    lo_native = float((torch.logsumexp(lpv[:, pl], -1) - torch.logsumexp(lpv[:, sg], -1)).mean()); edits = {}
    for k in KS:
        lp_, _ = run(backend, ids, tuple(order[:k].tolist())); forwards += 1; klk, lok = kl_lo(lp_)
        nulls = []
        for _ in range(N_NULL):
            rs = tuple(rng.sample(pool, k)); lpn, _ = run(backend, ids, rs); forwards += 1; nulls.append(kl_lo(lpn)[0])
        edits[str(k)] = {"kl": klk, "delta_logodds": lok - lo_native, "null_kl_max": max(nulls), "top_units": order[:min(k, 12)].tolist()}
    report = {"native_3465_medians": med3465, "rank_114": rank114, "top12_units": [(int(j), float(contrast[j])) for j in order[:12]], "top_shares": shares, "edits": edits, "logodds_plural_vs_singular_native_is_p": lo_native}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_frames_replay": replay, "pred_b_114_leads_the_mirror_census": rank114 == 1, "pred_c_mirror_population_is_spread": shares["50"] <= TOP50_MAX,
                   "pred_d_mirror_edit_moves_the_prediction": edits["10"]["kl"] >= KL_MIN and edits["10"]["kl"] > NULL_FACTOR * max(edits["10"]["null_kl_max"], 1e-9), "pred_e_mirror_edit_lowers_singular_preference": edits["10"]["delta_logodds"] > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mirror_violation_population_result_v354", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
