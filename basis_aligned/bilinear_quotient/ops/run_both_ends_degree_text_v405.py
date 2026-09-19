#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_noun_source_carries_a_quarter_on_text pred_c_value_only_term_leads_on_text pred_d_third_order_small pred_e_12_4_cancellation_replays_on_text
"""Both-ends contrast + degree expansion ON NATURAL TEXT (v405). v401-v404 on the panel: the readers copy number (and 'not he / she') from the noun's
current-state value; patterns number-blind; 12.4 cancels. Here the same exact eight-term expansion on the 128 natural verb rows (v272 / v273), made into
aligned pairs by swapping the cue noun's number IN PLACE with the tokenizer (guests <-> guest, bankers <-> banker; 122 of 128 cue nouns have a single-token
partner; the six without are dropped). The pair is (row with the plural noun, row with the singular noun); everything else in the text is identical.
PREDICTIONS (scored as written; failures preserved; priors from v401 / v403)
    pred_a_closure                              eight terms sum to the noun-source contrast within relative 1e-4; recomputed head writes match the module's within 2e-2
    pred_b_noun_source_carries_a_quarter_on_text  for 9.6 the noun-source term carries >= 0.25 of the head's total write contrast on u (text, pooled). Prior: unsure (0.48 on the panel)
    pred_c_value_only_term_leads_on_text        for 9.6 the value-only term is >= 0.60 of the noun-source contrast on text. Prior: likely (0.92 on the panel)
    pred_d_third_order_small                    |degree-3 term| <= 0.25 of the noun-source contrast for each of 9.6 / 12.4 / 15.1
    pred_e_12_4_cancellation_replays_on_text    for 12.4 the s2 x value term has the opposite sign to the value-only term on text. Prior: likely
PRICE (registered maximum): 244 sequences in 4 batches = 4 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/both_ends_degree_text_v405_result.json"
CANDIDATE_ID = "both_ends.degree_expansion_text_v405"
N_HEAD, BATCH = 9, 64
NATURAL = [dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, INSTR_TOL, NOUN_MIN, VALUE_MIN, THIRD_MAX = 1e-4, 2e-2, 0.25, 0.60, 0.25
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "<= 1e-4 / 2e-2", "pred_b_noun_source_carries_a_quarter_on_text": ">= 0.25", "pred_c_value_only_term_leads_on_text": ">= 0.60", "pred_d_third_order_small": "<= 0.25 x 3", "pred_e_12_4_cancellation_replays_on_text": "opposite signs"}
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
    noun_of = lambda row: row.n
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "dropped_records": dropped, "natural_records": len(recs), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "instr_tol": INSTR_TOL, "noun_min": NOUN_MIN, "value_min": VALUE_MIN, "third_max": THIRD_MAX}}
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
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = torch.tensor([r_.ids for r_ in chunk], device="cuda"); B_ = tokens.shape[0]; idx = torch.arange(B_)
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
    R = report["readers"]; r96 = R["9.6"]
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL and instr <= INSTR_TOL, "pred_b_noun_source_carries_a_quarter_on_text": r96["noun_source_share"] >= NOUN_MIN, "pred_c_value_only_term_leads_on_text": r96["term_shares"]["value_only"] >= VALUE_MIN,
                   "pred_d_third_order_small": all(abs(R[k]["degree_shares"]["degree3"]) <= THIRD_MAX for k in R), "pred_e_12_4_cancellation_replays_on_text": R["12.4"]["term_shares"]["s2_value"] * R["12.4"]["term_shares"]["value_only"] < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "both_ends_degree_text_result_v405", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
