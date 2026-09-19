#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_pattern_signs_fix_the_terms pred_c_pattern_weighted_term_ranks_top_two pred_d_negative_pattern_heads pred_e_direct_term_within_range
"""The token term's sign: the readers' signed pattern weights (v560). v559: the weights-only token term t_h = lamb_l (uO_h) W_v0 dn(e) ranks the gender readers'
token-only singles (top two and bottom two) but has the wrong sign for 10.1 and 15.1 -- every single is positive, t_10.1 and t_15.1 are negative. bilin18's
squared attention is SIGNED and unnormalised: pattern = (q.k / d)(q2.k2 / d) can be negative, so a head with a negative pattern weight on the noun transports
the negative of its value. Here the five readers' pattern weights from the answer position to the noun are captured natively on the 61 pairs (2 forwards);
the direct token term per head is then D_h = mean over rows of p_h(answer -> noun) x lamb_l (uO_h) W_v0 (n(e_row) - n(e_partner)) / 2 over the mean he - she gap,
and is compared with v559's token-only singles (read from its receipt; no re-run).
PREDICTIONS (scored as written; failures preserved; priors from v559)
    pred_a_baseline_replays                 the unedited manual forward replays the model's he - she margins within 1e-3
    pred_b_pattern_signs_fix_the_terms      sign(D_h) = sign(single_h) for every reader with |single| >= 0.02 (10.1, 12.4, 15.1)
    pred_c_pattern_weighted_term_ranks_top_two  the two largest |D_h| are 10.1 and 12.4 (v559's top two)
    pred_d_negative_pattern_heads           10.1 and 15.1 have negative mean pattern weight on the noun, 12.4 positive
    pred_e_direct_term_within_range         for 10.1, D_h / single_h is within [0.3, 1.5] (the direct write is a fair part of the effect; the rest is re-expression). Prior: unsure
PRICE (registered maximum): 2 batches x 1 pass (native, with pattern capture) = 2 forwards (61 pairs); 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import pronoun_gender_dod_natural_rows as gm
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_token_term_pattern_sign_v560_result.json"
CANDIDATE_ID = "gender.token_term_pattern_sign_v560"
ROWS = ROOT / "circuits/followups/pronoun_gender_dod_natural_rows_v73.json"
N_HEAD = 9
FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, SMALL, LO, HI = 1e-3, 0.02, 0.3, 1.5
V559 = ROOT / "circuits/followups/gender_token_branch_by_reader_v559_result.json"
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_pattern_signs_fix_the_terms": "signs agree x 3", "pred_c_pattern_weighted_term_ranks_top_two": "{10.1, 12.4}", "pred_d_negative_pattern_heads": "10.1, 15.1 < 0 < 12.4", "pred_e_direct_term_within_range": "[0.3, 1.5]"}


def gender_pairs():
    """Aligned pairs from v73's natural rows: the gendered noun swapped in place for its partner (same surface form: ' king' <-> ' queen', 'King' <-> 'Queen')."""
    E = L.ENCODING; partner_word = {}
    for m, f in gm.PAIRS: partner_word[m], partner_word[f] = f, m
    items = []
    for r_ in json.loads(ROWS.read_text())["rows"]:
        c = r_["cue_offset"]; tok = r_["ids"][c]; t = E.decode([tok]); w = t.strip(); lead = " " if t.startswith(" ") else ""
        base = w.lower()
        if base not in partner_word: continue
        pw = partner_word[base]; pw = pw.capitalize() if w[0].isupper() else pw
        alt = E.encode(lead + pw)
        if len(alt) != 1: continue
        sw = list(r_["ids"]); sw[c] = alt[0]
        male, female = (r_["ids"], sw) if r_["gender"] == "male" else (sw, r_["ids"])
        items.append((male, female, c))
    return items


def main() -> None:
    text_items = gender_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "rows": str(ROWS.name), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "small": SMALL, "lo": LO, "hi": HI}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" he"), L._single(" she")
    LOW_ALL = {l: list(range(N_HEAD)) for l in range(0, 9)}
    HIGH_ALL = {l: list(range(N_HEAD)) for l in range(9, 18)}
    HEADS = ((10, 1), (12, 4), (9, 6), (15, 1), (10, 5))
    MODES = {}; forwards = 0
    # weights-only token term per reader: t_h = lamb_l (u O_h) . W_v0,h (n(e_male) - n(e_female)), pooled over the pairs
    WU = model.lm_head.weight.detach().float(); u = (WU[HE] - WU[SHE]).cuda(); Wv0 = blocks[0].attn.c_v.weight.detach().float(); wte = model.transformer.wte.weight.detach().float()
    def n_(e): return F.rms_norm(e, (D,))
    tok_terms, tok_rows = {}, {}
    for l, h in HEADS:
        attn = blocks[l].attn; uO = u @ attn.c_proj.weight.detach().float()[:, h * hd:(h + 1) * hd]; Wv0h = Wv0[h * hd:(h + 1) * hd]
        per = []
        for male, female, c in text_items:
            d = n_(wte[male[c]]) - n_(wte[female[c]]); t = float(attn.lamb) * float(uO @ (Wv0h @ d)); per += [t, -t]     # male row: + (its change vs partner), female row: -
        tok_rows[f"{l}.{h}"] = torch.tensor(per); tok_terms[f"{l}.{h}"] = float(torch.tensor(per[0::2]).mean())
    print("weights-only token terms", {k: round(v, 3) for k, v in tok_terms.items()})
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, HE] - z[:, SHE]
    def run(seqs, pos, mode):
        """mode: native or a MODES key; values only, at the listed offsets from the noun. Pairs are rows (2i, 2i+1)."""
        nonlocal forwards
        prec = {}
        tokens = torch.tensor(seqs, device="cuda") if len({len(s) for s in seqs}) == 1 else torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if l in {rl for rl, _ in HEADS}:
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    cos, sin = attn.rotary(q)
                    qr, kr = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2r, k2r = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    for (rl, rh) in HEADS:
                        if rl == l:
                            s1 = (qr[idx, fin, rh] * kr[idx, pp, rh]).sum(-1).float() / hd; s2 = (q2r[idx, fin, rh] * k2r[idx, pp, rh]).sum(-1).float() / hd
                            prec[f"{rl}.{rh}"] = (s1 * s2).cpu()                                              # signed pattern weight, answer -> noun
                if mode != "native" and (l in MODES[mode][0] or l in MODES[mode][1] or MODES[mode][2]):
                    hs2, hs0, rest = MODES[mode][0].get(l, []), MODES[mode][1].get(l, []), MODES[mode][2]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    vcur, vtok = (1 - attn.lamb) * v, attn.lamb * v1n.view_as(v); v = vcur + vtok
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in hs2:
                        for o in (0, 1): v[idx, pp + o, h] = v[swap, pp + o, h].clone()               # readers: both sites
                    for h in hs0: v[idx, pp, h] = (vcur[idx, pp, h] + vtok[swap, pp, h]).clone()     # token-only branch swapped, current-state kept
                    if rest:
                        ar = torch.arange(tokens.shape[1], device="cuda")[None, :]; sel = (ar >= pp[:, None] + 2)[:, :, None, None]
                        v = torch.where(sel, v[swap], v)                                              # every head's value from two after the noun onward
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
            return logits_margin(x[idx, fin]).cpu(), prec
    def study(items, batch):
        seqs, pos = [], []
        for p, s_, c in items: seqs += [list(p), list(s_)]; pos += [c, c]
        nats, P = [], {f"{l}.{h}": [] for l, h in HEADS}
        for s0 in range(0, len(seqs), 2 * batch):
            mg, pr = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native"); nats.append(mg)
            for k in P: P[k].append(pr[k])
        nat = torch.cat(nats); gap = nat[0::2] - nat[1::2]; P = {k: torch.cat(v) for k, v in P.items()}
        # direct token term per head: pattern x token value term, both members of each pair, over the mean gap (same normalisation as 'closed')
        direct = {k: float((P[k] * tok_rows[k]).sum() / (2 * gap.sum())) for k in P}
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "pattern_mean": {k: float(v.mean()) for k, v in P.items()}, "pattern_sign_constancy": {k: float(max((v > 0).float().mean(), (v < 0).float().mean())) for k, v in P.items()},
                "direct_term": direct, "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32)
    with torch.no_grad():   # replay check against the module's own forward on the first text batch (rows padded to one length)
        n = min(32, len(text_items)); seqs = [list(p) for p, _, _ in text_items[:n]] + [list(s) for _, s, _ in text_items[:n]]
        T_ = max(len(q) for q in seqs); tokens = torch.tensor([q + [0] * (T_ - len(q)) for q in seqs], device="cuda"); fin_ = torch.tensor([len(q) - 1 for q in seqs], device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[torch.arange(2 * n, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = (z[:, HE] - z[:, SHE]).cpu()
        mine = torch.cat([nat_t[0:2 * n:2], nat_t[1:2 * n:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    singles = json.loads(V559.read_text())["report"]["text"]["closed"]; Dh = text["direct_term"]; pm = text["pattern_mean"]
    text["weights_token_terms"] = tok_terms; text["v559_singles"] = singles; report["text"] = {k: v for k, v in text.items() if k != "native_margins"}
    print("pattern means", {k: round(v, 3) for k, v in pm.items()}, "direct terms", {k: round(v, 3) for k, v in Dh.items()}, "singles", {k: round(v, 3) for k, v in singles.items()})
    top2 = set(sorted(Dh, key=lambda k: -abs(Dh[k]))[:2])
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_pattern_signs_fix_the_terms": all((Dh[k] > 0) == (singles[k] > 0) for k in singles if abs(singles[k]) >= SMALL),
                   "pred_c_pattern_weighted_term_ranks_top_two": top2 == {"10.1", "12.4"}, "pred_d_negative_pattern_heads": pm["10.1"] < 0 and pm["15.1"] < 0 and pm["12.4"] > 0,
                   "pred_e_direct_term_within_range": LO <= Dh["10.1"] / singles["10.1"] <= HI}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_token_term_pattern_sign_result_v560", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
