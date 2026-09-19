#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_factor_closure pred_c_3465_contrast_is_on_the_noun_factor pred_d_829_contrast_is_on_the_noun_factor pred_e_licensing_factor_is_constant_across_pairs
"""The agreement units' contrast on the pronoun rows, split by factor (v339). v338 (vocabulary pairs in four frames): unit 3465's L factor reads the noun's
number and its R factor the preceding word's licensing; 829 the same with sides swapped; 493 reads the noun on R. On the v76 rows every noun follows
"The", so the licensing factor should be (nearly) the same for the plural and singular member of each aligned pair and the units' plural - singular
contrast should live entirely on the noun factor. Exact class-wise split (the conventions' gradient x shift law): for aligned pairs (p, s),
h_p - h_s = (L_p - L_s) x (R_p + R_s)/2 + (L_p + L_s)/2 x (R_p - R_s), computed at the noun for 3465, 493, 1036, 829; the share of the pooled contrast on
each term, and the pair-wise spread of the licensing factor.
PREDICTIONS (scored as written; failures preserved; priors from v338)
    pred_a_closure                              the manual forward reproduces the model's they - he margin (2.048) within 1e-3 (instrument)
    pred_b_factor_closure                       the two terms sum to h_p - h_s within relative 1e-3 on every pair and unit
    pred_c_3465_contrast_is_on_the_noun_factor  for 3465, the L-contrast term carries >= 0.80 of the pooled contrast
    pred_d_829_contrast_is_on_the_noun_factor   for 829, the R-contrast term carries >= 0.70 of the pooled contrast (its noun factor is R)
    pred_e_licensing_factor_is_constant_across_pairs  for 3465, the median |R_p - R_s| / |(R_p + R_s)/2| over pairs is <= 0.25 (the licensing read of "The" barely depends on which noun follows). Prior: unsure.
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/agreement_factors_on_rows_v339_result.json"
CANDIDATE_ID = "pronoun_number.agreement_factors_on_rows_v339"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829,)}; BATCH = 32
CLOSURE_TOL, NATIVE_M, FACTOR_TOL, NOUN_3465, NOUN_829, LIC_MAX = 1e-3, 2.0481, 1e-3, 0.80, 0.70, 0.25
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_factor_closure": "<= 1e-3", "pred_c_3465_contrast_is_on_the_noun_factor": ">= 0.80 on L", "pred_d_829_contrast_is_on_the_noun_factor": ">= 0.70 on R", "pred_e_licensing_factor_is_constant_across_pairs": "<= 0.25"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "factor_tol": FACTOR_TOL, "noun_3465": NOUN_3465, "noun_829": NOUN_829, "lic_max": LIC_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    FLc, FRc, Hc = {u: [] for l in UNITS for u in UNITS[l]}, {u: [] for l in UNITS for u in UNITS[l]}, {u: [] for l in UNITS for u in UNITS[l]}; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    lf, rf = block.mlp.Left(xin).float().cpu(), block.mlp.Right(xin).float().cpu(); hh = dod_units.hidden(model, block.mlp, xin).float().cpu()
                    for u in UNITS[l]: FLc[u].append(lf[idx, pn, u]); FRc[u].append(rf[idx, pn, u]); Hc[u].append(hh[idx, pn, u])
                x = x + block.mlp(xin)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    per, fclos = {}, 0.0
    for u in FLc:
        Lf, Rf, H = torch.cat(FLc[u]), torch.cat(FRc[u]), torch.cat(Hc[u]); Lp, Ls, Rp, Rs = Lf[plural], Lf[sing], Rf[plural], Rf[sing]
        tL = (Lp - Ls) * (Rp + Rs) / 2; tR = (Lp + Ls) / 2 * (Rp - Rs); dh = H[plural] - H[sing]
        fclos = max(fclos, float(((tL + tR - dh).abs() / dh.abs().clamp_min(1e-6)).max())); tot = float(dh.sum())
        per[str(u)] = {"pooled_contrast": tot, "share_L_contrast_term": float(tL.sum()) / tot, "share_R_contrast_term": float(tR.sum()) / tot, "L_pair_rel_spread_median": float(((Lp - Ls).abs() / ((Lp + Ls) / 2).abs().clamp_min(1e-6)).median()),
                      "R_pair_rel_spread_median": float(((Rp - Rs).abs() / ((Rp + Rs) / 2).abs().clamp_min(1e-6)).median()), "L_mean": float(Lf.mean()), "R_mean": float(Rf.mean())}
    report = {"closure_margin_gap": closure, "factor_closure_max": fclos, "per_unit": per}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_factor_closure": fclos <= FACTOR_TOL, "pred_c_3465_contrast_is_on_the_noun_factor": per["3465"]["share_L_contrast_term"] >= NOUN_3465,
                   "pred_d_829_contrast_is_on_the_noun_factor": per["829"]["share_R_contrast_term"] >= NOUN_829, "pred_e_licensing_factor_is_constant_across_pairs": per["3465"]["R_pair_rel_spread_median"] <= LIC_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "agreement_factors_on_rows_result_v339", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
