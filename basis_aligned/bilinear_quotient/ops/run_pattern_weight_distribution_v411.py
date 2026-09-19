#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_9_6_pattern_sign_constant pred_c_9_6_pattern_weight_narrow pred_d_9_6_15_1_pattern_uncorrelated_with_number pred_e_12_4_pattern_correlated_with_number
"""Distributional fold of the readers' pattern weight at the noun (v411; Logan's 'distributional folding' family). v401-v410: number rides the value
branch; the pattern factors barely move between plural and singular. But is the pattern weight p = s1 * s2 at (answer, noun) a near-constant gate over the
NATURAL distribution (which would make each reader an effectively linear copier noun -> answer on this distribution), or does it vary with the row while
being number-blind? Here the distribution of p per reader over the 244 natural sequences (122 swapped pairs) and the v76 panel: sign constancy, coefficient
of variation, and its correlation with the noun's number.
PREDICTIONS (scored as written; failures preserved; priors from v403)
    pred_a_instrument                                recomputed head writes match the module's within 2e-2
    pred_b_9_6_pattern_sign_constant                 9.6's p has one sign on >= 0.95 of the natural sequences
    pred_c_9_6_pattern_weight_narrow                 9.6's p has coefficient of variation (std / |mean|) <= 0.50 over the natural sequences. Prior: unsure
    pred_d_9_6_15_1_pattern_uncorrelated_with_number |corr(p, plural indicator)| <= 0.20 over the natural sequences for 9.6 and 15.1
    pred_e_12_4_pattern_correlated_with_number       |corr(p, plural indicator)| >= 0.20 for 12.4 (its second factor halves on plural, v403). Prior: likely
PRICE (registered maximum): 4 text batches + 3 panel batches = 7 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_pronoun_number_dod_battery_v76 as g

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pattern_weight_distribution_v411_result.json"
CANDIDATE_ID = "both_ends.pattern_weight_distribution_v411"
N_HEAD, BATCH = 9, 64
NATURAL = [dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
INSTR_TOL, SIGN_MIN, CV_MAX, CORR_MAX, CORR_MIN = 2e-2, 0.95, 0.50, 0.20, 0.20
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_instrument": "<= 2e-2", "pred_b_9_6_pattern_sign_constant": ">= 0.95", "pred_c_9_6_pattern_weight_narrow": "CV <= 0.50", "pred_d_9_6_15_1_pattern_uncorrelated_with_number": "|corr| <= 0.20 x 2", "pred_e_12_4_pattern_correlated_with_number": "|corr| >= 0.20"}
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
    for r_ in prow: rows.append(Row(list(r_.ids), next(i for i, t in enumerate(r_.ids) if t in nouns), "panel", r_.present, None)); rows[-1].final = r_.final
    noun_of = lambda row: row.n
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "dropped_records": dropped, "natural_records": len(recs), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"instr_tol": INSTR_TOL, "sign_min": SIGN_MIN, "cv_max": CV_MAX, "corr_max": CORR_MAX, "corr_min": CORR_MIN}}
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
        for start in list(range(0, n_text, BATCH)) + list(range(n_text, len(rows), 32)):
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
                            n = int(nn_[i]); fac[(ll, h)][start + i] = {"s1": float(s1_all[i, n]), "s2": float(s2_all[i, n]), "w": float(w_all[i, n]), "total": float(total[i]),
                                                                        "qf": qf[i].cpu(), "q2f": q2f[i].cpu(), "kn": kh[i, n].cpu(), "k2n": k2h[i, n].cpu(), "n": n, "f": int(ff[i])}
                    v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    import statistics
    def stats(F_, idx):
        p = [F_[i]["s1"] * F_[i]["s2"] for i in idx]; pl = [1.0 if rows[i].present else 0.0 for i in idx]
        mean = statistics.fmean(p); sd = statistics.pstdev(p); sign = max(sum(1 for v in p if v > 0), sum(1 for v in p if v < 0)) / len(p)
        mp, ml = statistics.fmean(p), statistics.fmean(pl); cov = statistics.fmean([(a_ - mp) * (b_ - ml) for a_, b_ in zip(p, pl)]); corr = cov / (sd * statistics.pstdev(pl)) if sd > 0 and statistics.pstdev(pl) > 0 else 0.0
        return {"mean": mean, "std": sd, "cv": sd / max(abs(mean), 1e-9), "sign_constancy": sign, "corr_with_plural": corr, "n": len(p), "quantiles": [sorted(p)[int(q * (len(p) - 1))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)]}
    report = {"instrument_max_rel": instr, "readers": {}}
    for r in READERS:
        report["readers"][f"{r[0]}.{r[1]}"] = {"text": stats(fac[r], range(n_text)), "panel": stats(fac[r], range(n_text, len(rows)))}
    print(json.dumps(report, indent=1))
    R = report["readers"]; t96, t151, t124 = R["9.6"]["text"], R["15.1"]["text"], R["12.4"]["text"]
    predictions = {"pred_a_instrument": instr <= INSTR_TOL, "pred_b_9_6_pattern_sign_constant": t96["sign_constancy"] >= SIGN_MIN, "pred_c_9_6_pattern_weight_narrow": t96["cv"] <= CV_MAX,
                   "pred_d_9_6_15_1_pattern_uncorrelated_with_number": abs(t96["corr_with_plural"]) <= CORR_MAX and abs(t151["corr_with_plural"]) <= CORR_MAX, "pred_e_12_4_pattern_correlated_with_number": abs(t124["corr_with_plural"]) >= CORR_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pattern_weight_distribution_result_v411", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
