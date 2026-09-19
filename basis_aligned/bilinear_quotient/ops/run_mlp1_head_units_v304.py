#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_factor_closure pred_b_head_units_have_a_token_constant_factor pred_c_head_units_context_factor_tracks_share pred_d_head_units_write_the_common_lookup_direction pred_e_head_units_are_gain_units_on_text
"""MLP 1: what the head units 3289 and 624 read (v304). v303: the self-cancellation is layer-wide, but units 3289 and 624 carry ~30% of it (each ~9% of
the lookup along the token's entry and ~15% of the cut). A unit u_j = (L_j . n)(R_j . n) is a literal GAIN unit if one factor is (nearly) constant across
tokens (a bias-like direction: it reads "there is a token here") while the other reads how much context was mixed in. For the top-4 cancelling units
(v303 order: 3289, 624, 1715, 3804) at phrase A, lengths 1 / 8 / 64, 224 targets: (i) each factor's coefficient of variation across the 224
single-token inputs (token-constant if <= 0.25); (ii) each factor's correlation across rows with the own-key attention share (context-reading if
|r| >= 0.60); (iii) cosine of the unit's write direction D_j with the mean single-token lookup direction (the common component of the table);
(iv) on the 2,944 natural positions, the unit's activation h_j vs the position's alpha (Pearson).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_factor_closure                             sum over the 4 units of their per-unit cross terms equals the same computed by v303's split within 1e-3 (instrument)
    pred_b_head_units_have_a_token_constant_factor    for each of 3289 and 624, one factor has CV <= 0.25 across the 224 single tokens
    pred_c_head_units_context_factor_tracks_share     for each of 3289 and 624, the other factor's |r| with the own-key share across the 672 in-context rows >= 0.60
    pred_d_head_units_write_the_common_lookup_direction  |cos(D_j, mean unit lookup direction)| >= 0.50 for 3289 and 624
    pred_e_head_units_are_gain_units_on_text          on the natural positions, |r(h_j, alpha)| >= 0.40 for 3289 and 624
PRICE (registered maximum): 1 table batch + 3 x (length batch + self-share pass) + 2 natural + <= 12 natural-token table batches = 21 forwards; 0 backwards; 0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289
import dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_head_units_v304_result.json"
CANDIDATE_ID = "mlp1.token_table.head_units_v304"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
UNITS = (3289, 624, 1715, 3804); HEAD = (3289, 624)
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, CV_MAX, R_MIN, COS_MIN, TEXT_R_MIN = 1e-3, 0.25, 0.60, 0.50, 0.40
FORWARDS_MAX = 24
PREDICTIONS = {"pred_a_factor_closure": "<= 1e-3", "pred_b_head_units_have_a_token_constant_factor": "CV <= 0.25 x 2", "pred_c_head_units_context_factor_tracks_share": "|r| >= 0.60 x 2", "pred_d_head_units_write_the_common_lookup_direction": ">= 0.50 x 2", "pred_e_head_units_are_gain_units_on_text": "|r| >= 0.40 x 2"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cv_max": CV_MAX, "r_min": R_MIN, "cos_min": COS_MIN, "text_r_min": TEXT_R_MIN}, "units": list(UNITS)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    def pearson(a, b): a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
    That = T / T.norm(dim=1, keepdim=True); mean_dir = That.mean(0) / That.mean(0).norm()
    uidx = torch.tensor(UNITS); Lu, Ru, Du = Lw[uidx], Rw[uidx], Dw[:, uidx]            # [4, D], [4, D], [D, 4]
    fL_tab, fR_tab = n_tab @ Lu.T, n_tab @ Ru.T                                          # factors on the single-token inputs [224, 4]
    cv = {"L": (fL_tab.std(0) / fL_tab.mean(0).abs()), "R": (fR_tab.std(0) / fR_tab.mean(0).abs())}
    closure, rowsL, rowsR, rowsS, rowsA = 0.0, [], [], [], []
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); pos = torch.full((len(targets),), k, dtype=torch.long, device="cuda"); c = v289.capture(backend, toks, pos); forwards += 1
        sh = dod_units.attention_self_share(backend, toks, pos); forwards += 1
        W = c["mlp1"]; n = F.rms_norm(c["x1"], (T.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
        Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T; DT = That @ Dw
        cross_j_all = DT * (Lt2 * Rc + Lc * Rt2); cross_u = (That @ Du) * ((tp @ Lu.T) * (cc @ Ru.T) + (cc @ Lu.T) * (tp @ Ru.T))
        closure = max(closure, float(((cross_u.sum(1) - cross_j_all[:, uidx].sum(1)).abs() / cross_j_all[:, uidx].sum(1).abs().clamp_min(1e-6)).max()))
        rowsL.append(n @ Lu.T); rowsR.append(n @ Ru.T); rowsS.append(sh); rowsA.append((W * T).sum(1) / (T * T).sum(1))
    fL, fR, S = torch.cat(rowsL), torch.cat(rowsR), torch.cat(rowsS)
    r_share = {"L": [pearson(fL[:, i], S) for i in range(len(UNITS))], "R": [pearson(fR[:, i], S) for i in range(len(UNITS))]}
    cos_common = (Du.T @ mean_dir) / Du.norm(dim=0)
    # natural text: unit activations vs alpha per position
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]; nat = torch.tensor([r["ids"] for r in recs], device="cuda"); H, Wn, tokn = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == 1: hh = (mlp.Left(xin) * mlp.Right(xin))[:, 1:, uidx.to("cuda")].reshape(-1, len(UNITS)).float().cpu()
                m = block.mlp(xin); x = x + m
            forwards += 1; H.append(hh); Wn.append(m[:, 1:].reshape(-1, m.shape[-1]).float().cpu()); tokn.append(chunk[:, 1:].reshape(-1).cpu())
    H, Wn, tokn = torch.cat(H), torch.cat(Wn), torch.cat(tokn)
    uniq = sorted(set(tokn.tolist())); tabn = {}
    for s0 in range(0, len(uniq), 256):
        ids2 = torch.tensor(uniq[s0:s0 + 256], device="cuda").unsqueeze(1); c2 = v289.capture(backend, ids2, torch.zeros(len(ids2), dtype=torch.long, device="cuda")); forwards += 1
        tabn.setdefault("mlp1", []).append(c2["mlp1"])
    Tn_ = torch.cat(tabn["mlp1"]); tin = {t: i for i, t in enumerate(uniq)}; Tn_ = Tn_[torch.tensor([tin[t] for t in tokn.tolist()])]
    alpha_text = (Wn * Tn_).sum(1) / (Tn_ * Tn_).sum(1); r_text = [pearson(H[:, i], alpha_text) for i in range(len(UNITS))]
    per_unit = {str(u): {"cv_L": float(cv["L"][i]), "cv_R": float(cv["R"][i]), "mean_L_tab": float(fL_tab[:, i].mean()), "mean_R_tab": float(fR_tab[:, i].mean()), "r_L_share": r_share["L"][i], "r_R_share": r_share["R"][i], "cos_D_common": float(cos_common[i]), "r_h_alpha_text": r_text[i],
                         "h_text_median": float(H[:, i].median())} for i, u in enumerate(UNITS)}
    report = {"closure_max": closure, "per_unit": per_unit, "alpha_text_median": float(alpha_text.median())}
    print(json.dumps(report, indent=1))
    ok_const = lambda u: min(per_unit[str(u)]["cv_L"], per_unit[str(u)]["cv_R"]) <= CV_MAX
    ok_ctx = lambda u: max(abs(per_unit[str(u)]["r_L_share"]), abs(per_unit[str(u)]["r_R_share"])) >= R_MIN
    predictions = {"pred_a_factor_closure": closure <= CLOSURE_TOL, "pred_b_head_units_have_a_token_constant_factor": all(ok_const(u) for u in HEAD), "pred_c_head_units_context_factor_tracks_share": all(ok_ctx(u) for u in HEAD),
                   "pred_d_head_units_write_the_common_lookup_direction": all(abs(per_unit[str(u)]["cos_D_common"]) >= COS_MIN for u in HEAD), "pred_e_head_units_are_gain_units_on_text": all(abs(per_unit[str(u)]["r_h_alpha_text"]) >= TEXT_R_MIN for u in HEAD)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_head_units_result_v304", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
