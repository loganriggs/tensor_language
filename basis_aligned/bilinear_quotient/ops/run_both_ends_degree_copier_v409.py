#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_noun_source_carries_a_third pred_c_copier_is_a_value_copy_too pred_d_third_order_small pred_e_leading_term_replays_across_frames
"""Both-ends degree expansion for the copier head 4.5 (v409). v357-v366: 4.5 copies a determiner x noun-number state from the noun to the token after it;
v383-v385: the post-noun number state is rebuilt by MLPs 4-8 from a small attention seed. Here the exact eight-term expansion of 4.5's noun-source write at
the POST-NOUN token, projected on the direction that number takes downstream (9.6's value-branch reader direction m = (1 - lamb) c_v^T (u O_9.6), v406),
plural - singular noun on the v76 pairs. Is the seed a value copy like the readers' read, or does 4.5's pattern carry number (a product)?
PREDICTIONS (scored as written; failures preserved; priors from v357 / v401)
    pred_a_closure                            eight terms sum to the noun-source contrast within relative 1e-4; recomputed head writes match the module's within 2e-2
    pred_b_noun_source_carries_a_third        the noun-source term carries >= 0.35 of 4.5's total write contrast on m at the post-noun token. Prior: unsure
    pred_c_copier_is_a_value_copy_too         the value-only term is >= 0.60 of the noun-source contrast. Prior: unsure (v357: determiner x noun product)
    pred_d_third_order_small                  |degree-3 term| <= 0.25 of the noun-source contrast
    pred_e_leading_term_replays_across_frames the largest |term| category is the same in all three frames
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
OUT = ROOT / "circuits/followups/both_ends_degree_copier_v409_result.json"
CANDIDATE_ID = "both_ends.degree_expansion_copier_v409"
N_HEAD, BATCH = 9, 32
CLOSURE_TOL, INSTR_TOL, NOUN_MIN, VALUE_MIN, THIRD_MAX = 1e-4, 2e-2, 0.35, 0.60, 0.25
READERS = ((4, 5),)
DOWNSTREAM = (9, 6)
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_closure": "<= 1e-4 / 2e-2", "pred_b_noun_source_carries_a_third": ">= 0.35", "pred_c_copier_is_a_value_copy_too": ">= 0.60", "pred_d_third_order_small": "<= 0.25", "pred_e_leading_term_replays_across_frames": "same x 3"}
TERMS = ("value_only", "s1_only", "s2_only", "s1_s2", "s1_value", "s2_value", "s1_s2_value", "query_change")


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "instr_tol": INSTR_TOL, "noun_min": NOUN_MIN, "value_min": VALUE_MIN, "third_max": THIRD_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u0 = (WU[L._single(" they")] - WU[L._single(" he")]).cuda(); a9 = blocks[DOWNSTREAM[0]].attn; Wp9 = a9.c_proj.weight.detach().float()
    u = float(1 - a9.lamb) * (a9.c_v.weight.detach().float()[DOWNSTREAM[1] * hd:(DOWNSTREAM[1] + 1) * hd].T @ (u0 @ Wp9[:, DOWNSTREAM[1] * hd:(DOWNSTREAM[1] + 1) * hd]))   # m: the direction number takes into 9.6
    layers = {l for l, _ in READERS}; forwards = 0
    # per row and reader: factors at the noun source and the total head write on u
    fac = {r: {} for r in READERS}; instr = 0.0
    from jacclust.tt_model import apply_rotary_emb  # noqa
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); B_ = tokens.shape[0]; idx = torch.arange(B_)
            nn_ = torch.tensor([noun_of(r_) for r_ in chunk]); ff = nn_ + 1   # the write is read at the token AFTER the noun
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
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    pairs = [(i, partner[(r_.construction, r_.group, False)]) for i, r_ in enumerate(rows) if r_.present]
    report = {"instrument_max_rel": instr, "readers": {}}; closure = 0.0
    for r in READERS:
        acc = {t: 0.0 for t in TERMS}; frames = {}; noun_contrast = 0.0; total_contrast = 0.0; skipped = 0
        for (p, s) in pairs:
            P, S = fac[r][p], fac[r][s]
            if P["n"] != S["n"] or P["f"] != S["f"]: skipped += 1; continue
            d1, d2, dw = P["s1"] - S["s1"], P["s2"] - S["s2"], P["w"] - S["w"]
            # query-change term: (q_P with k_S) - (q_S with k_S), value of S
            s1x = float(P["qf"] @ S["kn"]) / hd; s2x = float(P["q2f"] @ S["k2n"]) / hd
            # with the query held at P, the noun-side factors of S must use q_P: redefine S factors under q_P so the 7 terms + query term close exactly
            S1, S2 = s1x, s2x
            terms = {"value_only": S1 * S2 * dw, "s1_only": (P["s1"] - S1) * S2 * S["w"], "s2_only": S1 * (P["s2"] - S2) * S["w"], "s1_s2": (P["s1"] - S1) * (P["s2"] - S2) * S["w"],
                     "s1_value": (P["s1"] - S1) * S2 * dw, "s2_value": S1 * (P["s2"] - S2) * dw, "s1_s2_value": (P["s1"] - S1) * (P["s2"] - S2) * dw, "query_change": (S1 * S2 - S["s1"] * S["s2"]) * S["w"]}
            nc = P["s1"] * P["s2"] * P["w"] - S["s1"] * S["s2"] * S["w"]
            closure = max(closure, abs(sum(terms.values()) - nc) / max(abs(nc), 1e-6))
            noun_contrast += nc; total_contrast += P["total"] - S["total"]
            for t in TERMS: acc[t] += terms[t]
            fr = frames.setdefault(rows[p].construction, {t: 0.0 for t in TERMS})
            for t in TERMS: fr[t] += terms[t]
        shares = {t: acc[t] / noun_contrast for t in TERMS}
        deg = {"degree1": shares["value_only"] + shares["s1_only"] + shares["s2_only"], "degree2": shares["s1_s2"] + shares["s1_value"] + shares["s2_value"], "degree3": shares["s1_s2_value"], "query": shares["query_change"]}
        lead = {c: max(v, key=lambda t: abs(v[t])) for c, v in frames.items()}
        report["readers"][f"{r[0]}.{r[1]}"] = {"noun_source_contrast": noun_contrast, "total_contrast": total_contrast, "noun_source_share": noun_contrast / total_contrast, "term_shares": shares, "degree_shares": deg, "leading_term_by_frame": lead, "pairs": len(pairs) - skipped, "skipped": skipped}
    report["closure_max"] = closure
    print(json.dumps(report, indent=1))
    R = report["readers"]; r96 = R["4.5"]
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL and instr <= INSTR_TOL, "pred_b_noun_source_carries_a_third": r96["noun_source_share"] >= NOUN_MIN, "pred_c_copier_is_a_value_copy_too": r96["term_shares"]["value_only"] >= VALUE_MIN,
                   "pred_d_third_order_small": all(abs(R[k]["degree_shares"]["degree3"]) <= THIRD_MAX for k in R), "pred_e_leading_term_replays_across_frames": len(set(r96["leading_term_by_frame"].values())) == 1}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "both_ends_degree_copier_result_v409", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
