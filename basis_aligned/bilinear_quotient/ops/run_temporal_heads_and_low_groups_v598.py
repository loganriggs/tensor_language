#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_11_3_carries pred_c_blocks_7_8_carry_most_of_low pred_d_9_1_or_9_4_carries pred_e_low_groups_add
"""The temporal line by head (v598). v597: on 93 natural temporal pairs every head's value at the cue carries 0.789 of the will - had gap, the four readers
{11.3, 9.1, 15.5, 9.4} 0.254, blocks 0-8 0.664, blocks 9-17 0.302. Here the four readers singly, and blocks 0-8 in three groups (0-4, 5-6, 7-8), values at
the cue swapped.
PREDICTIONS (scored as written; failures preserved; priors from v597 and the 17 Sep temporal battery)
    pred_a_baseline_replays          the unedited manual forward replays the model's will - had margins within 1e-3
    pred_b_11_3_carries              11.3 alone closes >= 0.05
    pred_c_blocks_7_8_carry_most_of_low  blocks 7-8 close >= 0.25
    pred_d_9_1_or_9_4_carries        9.1 or 9.4 alone closes >= 0.03
    pred_e_low_groups_add            B04 + B56 + B78 is within 0.10 of 0.664
PRICE (registered maximum): 3 batches x 8 passes = 24 forwards (93 pairs -> 3 token batches); 0 backwards; 0 fits. Bar <= 25.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import pronoun_gender_dod_natural_rows as gm
import dod_battery
import run_temporal_route_census_cue_v597 as pv

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/temporal_heads_and_low_groups_v598_result.json"
CANDIDATE_ID = "temporal.heads_and_low_groups_v598"
ROWS = ROOT / "circuits/followups/person_reflexive_dod_natural_rows_v110.json"
N_HEAD = 9
FIVE = {8: [1], 13: [1], 10: [5], 15: [1]}   # the four person readers
REPLAY_TOL, H113_MIN, B78_MIN, H9_MIN, LOW_V597, ADD = 1e-3, 0.05, 0.25, 0.03, 0.664, 0.10
FORWARDS_MAX = 25
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_11_3_carries": ">= 0.05", "pred_c_blocks_7_8_carry_most_of_low": ">= 0.25", "pred_d_9_1_or_9_4_carries": ">= 0.03", "pred_e_low_groups_add": "0.664 +- 0.10"}


def person_pairs(rows_path=None):
    """Aligned pairs from v110's natural rows: the person pronoun swapped in place (' I' <-> ' you', 'I' <-> 'You'); (i_row, you_row, cue_offset)."""
    E = L.ENCODING; partner = {" I": " you", " you": " I", "I": "You", "You": "I", " You": " I"}
    items = []
    for r_ in json.loads((rows_path or ROWS).read_text())["rows"]:
        c = r_["cue_offset"]; t = E.decode([r_["ids"][c]])
        if t not in partner: continue
        alt = E.encode(partner[t])
        if len(alt) != 1: continue
        sw = list(r_["ids"]); sw[c] = alt[0]
        i_row, you_row = (r_["ids"], sw) if t.strip() == "I" else (sw, r_["ids"])
        items.append((i_row, you_row, c))
    return items


gender_pairs = person_pairs   # name kept for the lineage's callers


def main() -> None:
    text_items = pv.temporal_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "rows": str(ROWS.name), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "h113_min": H113_MIN, "b78_min": B78_MIN, "h9_min": H9_MIN, "low_v597": LOW_V597, "add": ADD}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" will"), L._single(" had")   # person: the margin is myself - yourself (names kept from the gender lineage)
    LOW_ALL = {l: list(range(N_HEAD)) for l in range(0, 9)}
    HIGH_ALL = {l: list(range(N_HEAD)) for l in range(9, 18)}
    MODES = {"11.3": ({}, {11: [3]}, False), "9.1": ({}, {9: [1]}, False), "15.5": ({}, {15: [5]}, False), "9.4": ({}, {9: [4]}, False), "B04": ({}, {l: list(range(N_HEAD)) for l in range(0, 5)}, False), "B56": ({}, {l: list(range(N_HEAD)) for l in (5, 6)}, False), "B78": ({}, {l: list(range(N_HEAD)) for l in (7, 8)}, False)}; forwards = 0
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
                    for h in hs0: v[idx, pp, h] = v[swap, pp, h].clone()                              # low exits: the noun only
                    if rest:
                        ar = torch.arange(tokens.shape[1], device="cuda")[None, :]; sel = (ar >= pp[:, None] + 2)[:, :, None, None]
                        v = torch.where(sel, v[swap], v)                                              # every head's value from two after the noun onward
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
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
    print("temporal by head / low group", {k: round(v, 3) for k, v in c.items()})
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_11_3_carries": c["11.3"] >= H113_MIN, "pred_c_blocks_7_8_carry_most_of_low": c["B78"] >= B78_MIN,
                   "pred_d_9_1_or_9_4_carries": max(c["9.1"], c["9.4"]) >= H9_MIN, "pred_e_low_groups_add": abs(c["B04"] + c["B56"] + c["B78"] - LOW_V597) <= ADD}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "temporal_heads_and_low_groups_result_v598", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
