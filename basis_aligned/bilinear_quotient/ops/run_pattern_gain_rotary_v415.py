#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_cross_closure pred_c_interaction_is_rotary pred_d_key_side_leads_without_rotary pred_e_unrotated_gain_tracks_rotated
"""Is 9.6's query-key interaction the rotary distance term (v415). v414: 9.6's gain splits 0.25 query / 0.37 key / 0.38 interaction; 12.4 and 15.1 are
key-side. The rotary embedding makes q_i . k_j depend on the position difference f_i - n_j, which is a pure interaction in the cross matrix; v412 found
the gain falls with distance. Here the same exact cross split with the rotary REMOVED (q, k after the QK norm, before rotation): if the interaction
was the distance term it drops and the key main effect takes it.
PREDICTIONS (scored as written; failures preserved; priors from v414)
    pred_a_instrument                    recomputed head writes match the module's within 2e-2
    pred_b_cross_closure                 the three components sum to the total within relative 1e-3 (float32) and the rotated diagonal replays p within 1e-3
    pred_c_interaction_is_rotary         for 9.6 the unrotated interaction share is <= 0.50 x the rotated one (0.38). Prior: unsure
    pred_d_key_side_leads_without_rotary for 9.6 the key main effect is the largest component without rotary
    pred_e_unrotated_gain_tracks_rotated corr over rows of the unrotated diagonal p with the rotated (observed) p >= 0.70 for 9.6. Prior: unsure
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
OUT = ROOT / "circuits/followups/pattern_gain_rotary_v415_result.json"
CANDIDATE_ID = "both_ends.pattern_gain_rotary_v415"
N_HEAD, BATCH = 9, 64
NATURAL = [dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
INSTR_TOL, CROSS_TOL, HALF, TRACK_MIN = 2e-2, 1e-3, 0.50, 0.70
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_instrument": "<= 2e-2", "pred_b_cross_closure": "<= 1e-3", "pred_c_interaction_is_rotary": "<= 0.5 x rotated", "pred_d_key_side_leads_without_rotary": "key largest", "pred_e_unrotated_gain_tracks_rotated": "corr >= 0.70"}
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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"instr_tol": INSTR_TOL, "cross_tol": CROSS_TOL, "half": HALF, "track_min": TRACK_MIN}}
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
                    qn_, kn_, q2n_, k2n_ = F.rms_norm(q, (hd,)), F.rms_norm(k, (hd,)), F.rms_norm(q2, (hd,)), F.rms_norm(k2, (hd,))
                    q, k = apply_rotary_emb(qn_, cos, sin), apply_rotary_emb(kn_, cos, sin)
                    q2, k2 = apply_rotary_emb(q2n_, cos, sin), apply_rotary_emb(k2n_, cos, sin)
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
                            n = int(nn_[i]); fac[(ll, h)][start + i] = {"p_noun": float(s1_all[i, n] * s2_all[i, n]), "qf": qf[i].cpu(), "q2f": q2f[i].cpu(), "kn": kh[i, n].cpu(), "k2n": k2h[i, n].cpu(), "uqf": qn_[i, int(ff[i]), h].float().cpu(), "uq2f": q2n_[i, int(ff[i]), h].float().cpu(), "ukn": kn_[i, n, h].float().cpu(), "uk2n": k2n_[i, n, h].float().cpu()}
                    v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    report = {"instrument_max_rel": instr, "readers": {}}; cross_err = 0.0
    def split(P):
        m = P.mean(); rm = P.mean(1, keepdim=True); cm = P.mean(0, keepdim=True)
        total = float(((P - m) ** 2).sum()); q_eff = float(((rm - m) ** 2).sum()) * P.shape[1]; k_eff = float(((cm - m) ** 2).sum()) * P.shape[0]; inter = float(((P - rm - cm + m) ** 2).sum())
        return {"query_main": q_eff / total, "key_main": k_eff / total, "interaction": inter / total}, abs(q_eff + k_eff + inter - total) / max(total, 1e-9)
    for r in READERS:
        F_ = fac[r]; idx_ = range(n_text); st = lambda key: torch.stack([F_[i][key] for i in idx_])
        P = ((st("qf") @ st("kn").T) / hd) * ((st("q2f") @ st("k2n").T) / hd); U = ((st("uqf") @ st("ukn").T) / hd) * ((st("uq2f") @ st("uk2n").T) / hd)
        p_obs = torch.tensor([F_[i]["p_noun"] for i in idx_]); cross_err = max(cross_err, float(((P.diag() - p_obs).abs() / p_obs.abs().clamp_min(1e-9)).max()))
        sP, e1 = split(P); sU, e2 = split(U); cross_err = max(cross_err, e1, e2)
        d1, d2 = P.diag(), U.diag(); track = float(((d1 - d1.mean()) * (d2 - d2.mean())).mean() / (d1.std(unbiased=False) * d2.std(unbiased=False)))
        report["readers"][f"{r[0]}.{r[1]}"] = {"rotated": sP, "unrotated": sU, "diag_corr_unrotated_vs_rotated": track, "diag_mean_rotated": float(d1.mean()), "diag_mean_unrotated": float(d2.mean()), "n": n_text}
    report["cross_closure_max"] = cross_err
    print(json.dumps(report, indent=1))
    R = report["readers"]; r96 = R["9.6"]
    predictions = {"pred_a_instrument": instr <= INSTR_TOL, "pred_b_cross_closure": cross_err <= CROSS_TOL, "pred_c_interaction_is_rotary": r96["unrotated"]["interaction"] <= HALF * r96["rotated"]["interaction"],
                   "pred_d_key_side_leads_without_rotary": max(r96["unrotated"], key=r96["unrotated"].get) == "key_main", "pred_e_unrotated_gain_tracks_rotated": r96["diag_corr_unrotated_vs_rotated"] >= TRACK_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pattern_gain_rotary_result_v415", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
