#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_joint_replays_the_write pred_c_interaction_is_a_fifth pred_d_pattern_replays pred_e_values_replay
"""Pattern x value interaction, closed (v569). v568: for the five gender readers the pattern swap alone closes -0.072, the value swap at every position 0.732,
and the write swap 0.878 (v563) -- 0.22 unaccounted. With signed attention the write is sum_t p_t v_t, so (p + dp)(v + dv) - pv = p dv + dp v + dp dv: the
last term is the interaction and belongs to neither single swap. Here the readers' pattern AND values swapped jointly (JOINT: the partner's q, k, q2, k2 and v
at every position, own everything else), with PATTERN and ALL_POS as replays; interaction = JOINT - PATTERN - ALL_POS.
PREDICTIONS (scored as written; failures preserved; priors from v563 / v567 / v568)
    pred_a_baseline_replays      the unedited manual forward replays the model's he - she margins within 1e-3
    pred_b_joint_replays_the_write  JOINT closes 0.878 +- 0.05 (it is the write swap, up to the token-only v0 mixing, which is per position and also swapped)
    pred_c_interaction_is_a_fifth  JOINT - PATTERN - ALL_POS is within 0.08 of 0.218
    pred_d_pattern_replays       PATTERN closes -0.072 +- 0.03
    pred_e_values_replay         ALL_POS closes 0.732 +- 0.03
PRICE (registered maximum): 2 batches x 4 passes = 8 forwards (61 pairs); 0 backwards; 0 fits. Bar <= 9.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import pronoun_gender_dod_natural_rows as gm
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_pattern_value_interaction_v569_result.json"
CANDIDATE_ID = "gender.pattern_value_interaction_v569"
ROWS = ROOT / "circuits/followups/pronoun_gender_dod_natural_rows_v73.json"
N_HEAD = 9
FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, WRITE_V563, WBAND, INTER_V568, IBAND, PATTERN_V568, ALLPOS_V567, BAND = 1e-3, 0.878, 0.05, 0.218, 0.08, -0.072, 0.732, 0.03
FORWARDS_MAX = 9
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_joint_replays_the_write": "0.878 +- 0.05", "pred_c_interaction_is_a_fifth": "0.218 +- 0.08", "pred_d_pattern_replays": "-0.072 +- 0.03", "pred_e_values_replay": "0.732 +- 0.03"}


def gender_pairs(rows_path=None):
    """Aligned pairs from v73's natural rows (or `rows_path`, e.g. the v75 pile rows): the gendered noun swapped in place for its partner (same surface form: ' king' <-> ' queen', 'King' <-> 'Queen')."""
    E = L.ENCODING; partner_word = {}
    for m, f in gm.PAIRS: partner_word[m], partner_word[f] = f, m
    items = []
    for r_ in json.loads((rows_path or ROWS).read_text())["rows"]:
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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "write_v563": WRITE_V563, "wband": WBAND, "inter_v568": INTER_V568, "iband": IBAND, "pattern_v568": PATTERN_V568, "allpos_v567": ALLPOS_V567, "band": BAND}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" he"), L._single(" she")
    LOW_ALL = {l: list(range(N_HEAD)) for l in range(0, 9)}
    HIGH_ALL = {l: list(range(N_HEAD)) for l in range(9, 18)}
    RULES = {"ALL_POS": lambda rel: rel >= 0, "PATTERN": lambda rel: rel < -10**6, "JOINT": lambda rel: rel < -10**6}
    MODES = {m: ({}, FIVE, False) for m in RULES}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, HE] - z[:, SHE]
    def run(seqs, pos, mode):
        """mode: native or a MODES key; values only, at the listed offsets from the noun. Pairs are rows (2i, 2i+1)."""
        nonlocal forwards
        tokens = torch.tensor(seqs, device="cuda") if len({len(s) for s in seqs}) == 1 else torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if mode != "native" and (l in MODES[mode][0] or l in MODES[mode][1] or MODES[mode][2]):
                    hs2, hs0, rest = MODES[mode][0].get(l, []), MODES[mode][1].get(l, []), MODES[mode][2]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in hs2:
                        for o in (0, 1): v[idx, pp + o, h] = v[swap, pp + o, h].clone()               # readers: both sites
                    rel = torch.arange(tokens.shape[1], device="cuda")[None, :] - pp[:, None]; sel = RULES[mode](rel)[:, :, None]     # (B, T, 1): positions to swap, relative to the noun
                    vs_ = v[swap]
                    for h in hs0: v[:, :, h] = torch.where(sel, vs_[:, :, h], v[:, :, h])
                    if rest:
                        ar = torch.arange(tokens.shape[1], device="cuda")[None, :]; sel = (ar >= pp[:, None] + 2)[:, :, None, None]
                        v = torch.where(sel, v[swap], v)                                              # every head's value from two after the noun onward
                    if mode in ("PATTERN", "JOINT"):
                        y_own = attn.squared_attention(q, k, v, q2, k2)
                        y_sw = attn.squared_attention(q[swap], k[swap], v, q2[swap], k2[swap]) if mode == "PATTERN" else attn.squared_attention(q[swap], k[swap], v[swap], q2[swap], k2[swap])   # partner's pattern (and values for JOINT)
                        y = y_own.clone()
                        for h in hs0: y[:, h] = y_sw[:, h]
                    else:
                        y = attn.squared_attention(q, k, v, q2, k2)
                    y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
            return logits_margin(x[idx, fin]).cpu()
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        res = {m: [] for m in ("native", *MODES)}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res: res[m].append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m))
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        sign = float(max((gap > 0).float().mean(), (gap < 0).float().mean()))
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "sign_constancy": sign, "closed": {m: closed(res[m]) for m in MODES}, "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32)
    with torch.no_grad():   # replay check against the module's own forward on the first text batch (rows padded to one length)
        n = min(32, len(text_items)); seqs = [list(p) for p, _, _ in text_items[:n]] + [list(s) for _, s, _ in text_items[:n]]
        T_ = max(len(q) for q in seqs); tokens = torch.tensor([q + [0] * (T_ - len(q)) for q in seqs], device="cuda"); fin_ = torch.tensor([len(q) - 1 for q in seqs], device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[torch.arange(2 * n, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = (z[:, HE] - z[:, SHE]).cpu()
        mine = torch.cat([nat_t[0:2 * n:2], nat_t[1:2 * n:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    c = text["closed"]
    inter = c["JOINT"] - c["PATTERN"] - c["ALL_POS"]; print("joint", round(c["JOINT"], 3), "pattern", round(c["PATTERN"], 3), "values", round(c["ALL_POS"], 3), "interaction", round(inter, 3))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_joint_replays_the_write": abs(c["JOINT"] - WRITE_V563) <= WBAND, "pred_c_interaction_is_a_fifth": abs(inter - INTER_V568) <= IBAND,
                   "pred_d_pattern_replays": abs(c["PATTERN"] - PATTERN_V568) <= BAND, "pred_e_values_replay": abs(c["ALL_POS"] - ALLPOS_V567) <= BAND}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_pattern_value_interaction_result_v569", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
