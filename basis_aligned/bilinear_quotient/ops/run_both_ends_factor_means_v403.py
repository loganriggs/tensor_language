#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_12_4_second_factor_falls_on_plural pred_c_9_6_factors_flat pred_d_value_copy_is_current_state_not_token pred_e_12_4_net_positive
"""Both-ends degree expansion, follow-up (v403): 12.4's pattern-value cancellation and the value branch's origin.
v401 / v402: the readers' number read is a value copy (9.6 / 15.1: value-only 0.92-1.13, pattern terms <= 0.19); 12.4 alone shows a large cancellation
(value-only 1.7-2.1 against s2 x value -1.1 to -1.3): its second squared-attention factor s2 = (q2_f . k2_n)/d FALLS on plural sites while the value rises.
Here, per reader and site (noun, post-noun): the plural and singular means of s1, s2 and w; and the value-only term split by value branch
(v_n = (1 - lamb) v_cur + lamb v1_token: the current-state part vs the token-only block-0 part), pooled over the aligned pairs.
PREDICTIONS (scored as written; failures preserved; priors from v401 / v402)
    pred_a_closure                            recomputed head writes match the module's within 2e-2; value-branch split closes within 1e-4
    pred_b_12_4_second_factor_falls_on_plural  for 12.4 the plural-site mean of s2 is below the singular-site mean at both sites (its pattern weight drops on plural)
    pred_c_9_6_factors_flat                   for 9.6 |mean s1_P - mean s1_S| and |mean s2_P - mean s2_S| are each <= 0.10 of the respective singular means, both sites
    pred_d_value_copy_is_current_state_not_token  for 9.6 the current-state value branch carries >= 0.80 of the value-only term at the noun (the number state is MLP-written, v82 / v168). Prior: likely
    pred_e_12_4_net_positive                  12.4's noun-source contrast is positive at both sites despite the cancellation
PRICE (registered maximum): 3 panel batches = 3 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/both_ends_factor_means_v403_result.json"
CANDIDATE_ID = "both_ends.factor_means_v403"
N_HEAD, BATCH = 9, 32
CLOSURE_TOL, INSTR_TOL, FLAT_MAX, CUR_MIN = 1e-4, 2e-2, 0.10, 0.80
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_closure": "<= 2e-2 / 1e-4", "pred_b_12_4_second_factor_falls_on_plural": "s2_P < s2_S x 2 sites", "pred_c_9_6_factors_flat": "<= 0.10 x 4", "pred_d_value_copy_is_current_state_not_token": ">= 0.80", "pred_e_12_4_net_positive": "> 0 x 2"}
SITES = {"noun": 0, "postnoun": 1}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "instr_tol": INSTR_TOL, "flat_max": FLAT_MAX, "cur_min": CUR_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    layers = {l for l, _ in READERS}; forwards = 0
    # per row and reader: factors at the noun source and the total head write on u
    fac = {(r, st): {} for r in READERS for st in SITES}; instr = 0.0; closure = 0.0
    from jacclust.tt_model import apply_rotary_emb  # noqa
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); B_ = tokens.shape[0]; idx = torch.arange(B_)
            nn_ = torch.tensor([noun_of(r_) for r_ in chunk]); ff = torch.tensor([r_.final for r_ in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if l in layers:
                    captured = {}; hook = attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1n = attn(xin, v1_)
                    finally: hook.remove()
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    vcur = attn.c_v(xin).view(B_, -1, N_HEAD, hd); vtok = v1_.view_as(vcur); v = (1 - attn.lamb) * vcur + attn.lamb * vtok
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    Wp = attn.c_proj.weight.detach().float(); y = captured["y"].float()
                    for (ll, h) in READERS:
                        if ll != l: continue
                        uO = (u @ Wp[:, h * hd:(h + 1) * hd])                                  # (hd,)  u . O_h
                        qf, q2f = q[idx, ff, h].float(), q2[idx, ff, h].float()                  # (B, hd)
                        kh, k2h, vh = k[:, :, h].float(), k2[:, :, h].float(), v[:, :, h].float()  # (B, T, hd)
                        wc_all = (float(1 - attn.lamb) * vcur[:, :, h].float()) @ uO; wt_all = (float(attn.lamb) * vtok[:, :, h].float()) @ uO
                        s1_all = (qf[:, None, :] * kh).sum(-1) / hd; s2_all = (q2f[:, None, :] * k2h).sum(-1) / hd; w_all = vh @ uO   # (B, T)
                        T_ = tokens.shape[1]; mask = (torch.arange(T_, device=tokens.device)[None, :] <= ff.to(tokens.device)[:, None]).float()
                        total = (s1_all * s2_all * w_all * mask).sum(1)                           # recomputed head write on u
                        captured_total = (y[idx, ff, h * hd:(h + 1) * hd] @ uO)
                        instr = max(instr, float(((total - captured_total).abs() / captured_total.abs().clamp_min(1e-3)).max()))
                        for i, row in enumerate(chunk):
                            for st, off in SITES.items():
                                n = int(nn_[i]) + off
                                closure = max(closure, abs(float(wc_all[i, n] + wt_all[i, n] - w_all[i, n])) / max(abs(float(w_all[i, n])), 1e-6))
                                fac[((ll, h), st)][start + i] = {"s1": float(s1_all[i, n]), "s2": float(s2_all[i, n]), "w": float(w_all[i, n]), "wc": float(wc_all[i, n]), "wt": float(wt_all[i, n]), "total": float(total[i]), "n": n, "f": int(ff[i])}
                    v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    pairs = [(i, partner[(r_.construction, r_.group, False)]) for i, r_ in enumerate(rows) if r_.present]
    report = {"instrument_max_rel": instr, "closure_max": closure, "readers": {}}
    for r in READERS:
        R_ = {}
        for st in SITES:
            F_ = fac[(r, st)]; mean = lambda key, idx: sum(F_[i][key] for i in idx) / len(idx)
            P_, S_ = [p for p, _ in pairs], [s_ for _, s_ in pairs]
            m = {key: {"plural": mean(key, P_), "singular": mean(key, S_)} for key in ("s1", "s2", "w", "wc", "wt")}
            nc = sum(F_[p]["s1"] * F_[p]["s2"] * F_[p]["w"] - F_[s_]["s1"] * F_[s_]["s2"] * F_[s_]["w"] for p, s_ in pairs)
            val_cur = sum(F_[s_]["s1"] * F_[s_]["s2"] * (F_[p]["wc"] - F_[s_]["wc"]) for p, s_ in pairs); val_tok = sum(F_[s_]["s1"] * F_[s_]["s2"] * (F_[p]["wt"] - F_[s_]["wt"]) for p, s_ in pairs)
            R_[st] = {"factor_means": m, "noun_source_contrast": nc, "value_only_current_state": val_cur, "value_only_token_only": val_tok, "current_share_of_value_only": val_cur / (val_cur + val_tok) if abs(val_cur + val_tok) > 1e-9 else None,
                      "rel_change": {key: (m[key]["plural"] - m[key]["singular"]) / max(abs(m[key]["singular"]), 1e-9) for key in ("s1", "s2", "w")}}
        report["readers"][f"{r[0]}.{r[1]}"] = R_
    print(json.dumps(report, indent=1))
    R = report["readers"]; r124, r96 = R["12.4"], R["9.6"]
    predictions = {"pred_a_closure": instr <= INSTR_TOL and closure <= CLOSURE_TOL, "pred_b_12_4_second_factor_falls_on_plural": all(r124[st]["factor_means"]["s2"]["plural"] < r124[st]["factor_means"]["s2"]["singular"] for st in SITES),
                   "pred_c_9_6_factors_flat": all(abs(r96[st]["rel_change"][key]) <= FLAT_MAX for st in SITES for key in ("s1", "s2")), "pred_d_value_copy_is_current_state_not_token": (r96["noun"]["current_share_of_value_only"] or 0) >= CUR_MIN,
                   "pred_e_12_4_net_positive": all(r124[st]["noun_source_contrast"] > 0 for st in SITES)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "both_ends_factor_means_result_v403", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
