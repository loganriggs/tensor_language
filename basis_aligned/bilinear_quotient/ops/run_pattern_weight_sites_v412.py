#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_9_6_weight_falls_with_distance pred_c_noun_and_postnoun_weights_covary pred_d_two_site_sum_narrower pred_e_two_site_sum_sign_constant
"""What sets the readers' pattern weight (v412). v411: p = s1 * s2 at (answer, noun) is number-blind but varies as much as its mean on natural text
(9.6: CV 1.1, one sign on 0.84 of rows). The readers read two sites about equally (v381 / v402): the noun and the token after it. Here, on the 244 natural
sequences: p at the noun vs the answer - noun distance; p at the noun vs p at the post-noun site; and whether the two-site sum p_n + p_n1 is the steadier
gate (a head that attends to the noun PHRASE, spreading its weight between the two tokens row by row, would show a narrow sum and a wide split).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_instrument                       recomputed head writes match the module's within 2e-2
    pred_b_9_6_weight_falls_with_distance   corr(p_noun, answer - noun distance) <= -0.20 for 9.6 (recency). Prior: unsure
    pred_c_noun_and_postnoun_weights_covary corr(p_noun, p_postnoun) >= 0.30 for 9.6. Prior: unsure
    pred_d_two_site_sum_narrower            CV(p_noun + p_postnoun) <= 0.80 x CV(p_noun) for 9.6 (the sum is the steadier gate). Prior: unsure
    pred_e_two_site_sum_sign_constant       p_noun + p_postnoun has one sign on >= 0.90 of the natural sequences for 9.6
PRICE (registered maximum): 4 text batches = 4 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_pronoun_number_dod_battery_v76 as g

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pattern_weight_sites_v412_result.json"
CANDIDATE_ID = "both_ends.pattern_weight_sites_v412"
N_HEAD, BATCH = 9, 64
NATURAL = [dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
INSTR_TOL, DIST_CORR_MAX, COVARY_MIN, CV_RATIO_MAX, SIGN_MIN = 2e-2, -0.20, 0.30, 0.80, 0.90
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_instrument": "<= 2e-2", "pred_b_9_6_weight_falls_with_distance": "corr <= -0.20", "pred_c_noun_and_postnoun_weights_covary": "corr >= 0.30", "pred_d_two_site_sum_narrower": "CV ratio <= 0.80", "pred_e_two_site_sum_sign_constant": ">= 0.90"}
TERMS = ("value_only", "s1_only", "s2_only", "s1_s2", "s1_value", "s2_value", "s1_s2_value", "query_change")


def main() -> None:
    E = L.ENCODING
    def partner(tok):
        t = E.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E.encode(c)) == 1: return E.encode(c)[0]
        return None
    recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    class Row:  # minimal stand-in for the panel row: ids, noun position, final position, construction = cue half, present = plural noun
        def __init__(self, ids, n, cue, present, group): self.ids, self.n, self.final, self.construction, self.present, self.group = ids, n, len(ids) - 1, cue, present, group
    rows, dropped = [], 0
    for gi, r_ in enumerate(recs):
        c = r_["cue_offset"]; alt = partner(r_["ids"][c])
        if alt is None: dropped += 1; continue
        swapped = list(r_["ids"]); swapped[c] = alt; plural_first = r_["cue"] == "plural"
        rows.append(Row(r_["ids"], c, r_["cue"], plural_first, gi)); rows.append(Row(swapped, c, r_["cue"], not plural_first, gi))
    prow, he, she, agents, objects = g.build(); nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    n_text = len(rows)
    noun_of = lambda row: row.n
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "dropped_records": dropped, "natural_records": len(recs), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"instr_tol": INSTR_TOL, "dist_corr_max": DIST_CORR_MAX, "covary_min": COVARY_MIN, "cv_ratio_max": CV_RATIO_MAX, "sign_min": SIGN_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    layers = {l for l, _ in READERS}; forwards = 0
    # per row and reader: factors at the noun source and the total head write on u
    fac = {r: {} for r in READERS}; instr = 0.0
    from jacclust.tt_model import apply_rotary_emb  # noqa
    with torch.no_grad():
        for start in range(0, n_text, BATCH):
            chunk = rows[start:min(start + BATCH, n_text)] if start < n_text else rows[start:start + 32]; tokens = fw._tokens(chunk); B_ = tokens.shape[0]; idx = torch.arange(B_)   # v411 crashed twice here: panel rows vary in length (pad), and a comment swallowed idx
            nn_ = torch.tensor([noun_of(r_) for r_ in chunk]); ff = torch.tensor([r_.final for r_ in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if l in layers:
                    captured = {}; hook = attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1n = attn(xin, v1_)
                    finally: hook.remove()
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd); v = (1 - attn.lamb) * v + attn.lamb * v1_.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    Wp = attn.c_proj.weight.detach().float(); y = captured["y"].float()
                    for (ll, h) in READERS:
                        if ll != l: continue
                        uO = (u @ Wp[:, h * hd:(h + 1) * hd])                                  # (hd,)  u . O_h
                        qf, q2f = q[idx, ff, h].float(), q2[idx, ff, h].float()                  # (B, hd)
                        kh, k2h, vh = k[:, :, h].float(), k2[:, :, h].float(), v[:, :, h].float()  # (B, T, hd)
                        s1_all = (qf[:, None, :] * kh).sum(-1) / hd; s2_all = (q2f[:, None, :] * k2h).sum(-1) / hd; w_all = vh @ uO   # (B, T)
                        T_ = tokens.shape[1]; mask = (torch.arange(T_, device=tokens.device)[None, :] <= ff.to(tokens.device)[:, None]).float()
                        total = (s1_all * s2_all * w_all * mask).sum(1)                           # recomputed head write on u
                        captured_total = (y[idx, ff, h * hd:(h + 1) * hd] @ uO)
                        instr = max(instr, float(((total - captured_total).abs() / captured_total.abs().clamp_min(1e-3)).max()))
                        for i, row in enumerate(chunk):
                            n = int(nn_[i]); fac[(ll, h)][start + i] = {"p_noun": float(s1_all[i, n] * s2_all[i, n]), "p_post": float(s1_all[i, n + 1] * s2_all[i, n + 1]), "dist": int(ff[i]) - n, "total": float(total[i])}
                    v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    import statistics
    def corr(a_, b_):
        ma, mb = statistics.fmean(a_), statistics.fmean(b_); sa, sb = statistics.pstdev(a_), statistics.pstdev(b_)
        return statistics.fmean([(x - ma) * (y - mb) for x, y in zip(a_, b_)]) / (sa * sb) if sa > 0 and sb > 0 else 0.0
    report = {"instrument_max_rel": instr, "readers": {}}
    for r in READERS:
        F_ = fac[r]; idx_ = range(n_text); pn = [F_[i]["p_noun"] for i in idx_]; pp = [F_[i]["p_post"] for i in idx_]; ds = [float(F_[i]["dist"]) for i in idx_]; sm = [x + y for x, y in zip(pn, pp)]
        cv = lambda v: statistics.pstdev(v) / max(abs(statistics.fmean(v)), 1e-9); sign = lambda v: max(sum(1 for x in v if x > 0), sum(1 for x in v if x < 0)) / len(v)
        report["readers"][f"{r[0]}.{r[1]}"] = {"corr_noun_distance": corr(pn, ds), "corr_noun_post": corr(pn, pp), "cv_noun": cv(pn), "cv_post": cv(pp), "cv_sum": cv(sm), "sign_noun": sign(pn), "sign_post": sign(pp), "sign_sum": sign(sm),
                                                "mean_noun": statistics.fmean(pn), "mean_post": statistics.fmean(pp), "distance_range": [min(ds), max(ds)], "n": n_text}
    print(json.dumps(report, indent=1))
    R = report["readers"]; t96 = R["9.6"]
    predictions = {"pred_a_instrument": instr <= INSTR_TOL, "pred_b_9_6_weight_falls_with_distance": t96["corr_noun_distance"] <= DIST_CORR_MAX, "pred_c_noun_and_postnoun_weights_covary": t96["corr_noun_post"] >= COVARY_MIN,
                   "pred_d_two_site_sum_narrower": t96["cv_sum"] <= CV_RATIO_MAX * t96["cv_noun"], "pred_e_two_site_sum_sign_constant": t96["sign_sum"] >= SIGN_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pattern_weight_sites_result_v412", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
