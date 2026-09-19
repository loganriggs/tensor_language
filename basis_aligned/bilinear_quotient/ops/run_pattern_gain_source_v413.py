#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_answer_token_explains_half pred_c_noun_token_explains_little pred_d_query_norm_tracks_gain pred_e_sign_flips_are_answer_tokens
"""Where the readers' row-level gain is set (v413). v411 / v412: the pattern weight p = s1 * s2 at (answer, noun) is number-blind, co-varies across the
two source sites and falls with distance — a row-level gain. Two candidates: the ANSWER token (the query side; the natural rows end in ' said', ' if',
' when', ' thought', ...) or the NOUN token (the key side). Here, with no fit: the between-group share of p's variance when the 244 sequences are grouped
by answer token vs by noun token (groups of one sequence count as their own mean, so singletons inflate the share: reported with the group counts and
with the share computed over groups of size >= 3 only), the correlation of p with the query's contribution |q_f| |q2_f| (norms after the QK norm are fixed,
so this is a null check that should be ~0), and whether the sequences with the minority sign share their answer token.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_instrument                    recomputed head writes match the module's within 2e-2
    pred_b_answer_token_explains_half    grouping by answer token (groups of size >= 3) explains >= 0.50 of p's variance for 9.6. Prior: unsure
    pred_c_noun_token_explains_little    grouping by noun token (groups of size >= 3) explains <= 0.30 of p's variance for 9.6. Prior: unsure
    pred_d_query_norm_tracks_gain        |corr(p, |q_f| * |q2_f|)| <= 0.10 for 9.6 (the QK norm fixes the norms; null check)
    pred_e_sign_flips_are_answer_tokens  among 9.6's minority-sign sequences, the most common answer token covers >= 0.40 of them. Prior: unsure
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
OUT = ROOT / "circuits/followups/pattern_gain_source_v413_result.json"
CANDIDATE_ID = "both_ends.pattern_gain_source_v413"
N_HEAD, BATCH = 9, 64
NATURAL = [dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", dod_battery.ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
INSTR_TOL, ANSWER_MIN, NOUN_MAX, QNORM_MAX, FLIP_MIN, MIN_GROUP = 2e-2, 0.50, 0.30, 0.10, 0.40, 3
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_instrument": "<= 2e-2", "pred_b_answer_token_explains_half": ">= 0.50", "pred_c_noun_token_explains_little": "<= 0.30", "pred_d_query_norm_tracks_gain": "|corr| <= 0.10", "pred_e_sign_flips_are_answer_tokens": ">= 0.40"}
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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"instr_tol": INSTR_TOL, "answer_min": ANSWER_MIN, "noun_max": NOUN_MAX, "qnorm_max": QNORM_MAX, "flip_min": FLIP_MIN, "min_group": MIN_GROUP}}
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
                            n = int(nn_[i]); fac[(ll, h)][start + i] = {"p_noun": float(s1_all[i, n] * s2_all[i, n]), "qnorm": float(qf[i].norm() * q2f[i].norm()), "answer": int(tokens[i, int(ff[i])]), "noun": int(tokens[i, n])}
                    v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    import statistics, collections
    E = L.ENCODING
    def corr(a_, b_):
        ma, mb = statistics.fmean(a_), statistics.fmean(b_); sa, sb = statistics.pstdev(a_), statistics.pstdev(b_)
        return statistics.fmean([(x - ma) * (y - mb) for x, y in zip(a_, b_)]) / (sa * sb) if sa > 0 and sb > 0 else 0.0
    def between_share(p, keys, min_group):
        groups = collections.defaultdict(list)
        for v, k in zip(p, keys): groups[k].append(v)
        kept = {k: v for k, v in groups.items() if len(v) >= min_group}; flat = [x for v in kept.values() for x in v]
        if len(flat) < 2: return None, len(groups), 0
        m = statistics.fmean(flat); tot = sum((x - m) ** 2 for x in flat); between = sum(len(v) * (statistics.fmean(v) - m) ** 2 for v in kept.values())
        return between / tot if tot > 0 else None, len(groups), len(flat)
    report = {"instrument_max_rel": instr, "readers": {}}
    for r in READERS:
        F_ = fac[r]; idx_ = range(n_text); p = [F_[i]["p_noun"] for i in idx_]; ans = [F_[i]["answer"] for i in idx_]; nn = [F_[i]["noun"] for i in idx_]; qn = [F_[i]["qnorm"] for i in idx_]
        sa, ga, na = between_share(p, ans, MIN_GROUP); sn, gn, nn_k = between_share(p, nn, MIN_GROUP); sa1, _, _ = between_share(p, ans, 1); sn1, _, _ = between_share(p, nn, 1)
        maj = 1 if sum(1 for x in p if x > 0) >= len(p) / 2 else -1; minority = [E.decode([a_]) for x, a_ in zip(p, ans) if x * maj < 0]
        top = collections.Counter(minority).most_common(3); flip_share = top[0][1] / len(minority) if minority else 0.0
        answer_means = {E.decode([k]): (statistics.fmean(v), len(v)) for k, v in collections.defaultdict(list, {k: [x for x, a_ in zip(p, ans) if a_ == k] for k in set(ans)}).items() if len(v) >= MIN_GROUP}
        report["readers"][f"{r[0]}.{r[1]}"] = {"answer_share_min3": sa, "answer_groups": ga, "answer_n_kept": na, "answer_share_all": sa1, "noun_share_min3": sn, "noun_groups": gn, "noun_n_kept": nn_k, "noun_share_all": sn1,
                                                "corr_qnorm": corr(p, qn), "minority_n": len(minority), "minority_top_answers": top, "flip_share": flip_share, "answer_means_min3": dict(sorted(answer_means.items(), key=lambda kv: kv[1][0]))}
    print(json.dumps(report, indent=1))
    R = report["readers"]; t96 = R["9.6"]
    predictions = {"pred_a_instrument": instr <= INSTR_TOL, "pred_b_answer_token_explains_half": (t96["answer_share_min3"] or 0) >= ANSWER_MIN, "pred_c_noun_token_explains_little": (t96["noun_share_min3"] if t96["noun_share_min3"] is not None else 1.0) <= NOUN_MAX,
                   "pred_d_query_norm_tracks_gain": abs(t96["corr_qnorm"]) <= QNORM_MAX, "pred_e_sign_flips_are_answer_tokens": t96["flip_share"] >= FLIP_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pattern_gain_source_result_v413", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
