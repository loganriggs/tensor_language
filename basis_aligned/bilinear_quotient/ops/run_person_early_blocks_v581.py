#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_block_4_leads_the_early_group pred_c_early_blocks_add pred_d_block_8_rest_and_7_make_the_remainder pred_e_no_single_block_above_0_15
"""The person line's early exits by block (v581). v580: on 43 natural person pairs blocks 0-4's values at the cue carry 0.216 of the myself - yourself gap,
blocks 7-8 0.242 of which 8.1 is 0.148. Here blocks 0, 1, 2, 3, 4 singly, block 7 whole, and block 8 without 8.1 (B8_REST), values at the cue swapped.
PREDICTIONS (scored as written; failures preserved; priors from v580 and the number line's copier block 4)
    pred_a_baseline_replays                   the unedited manual forward replays the model's myself - yourself margins within 1e-3
    pred_b_block_4_leads_the_early_group      block 4 closes the most of blocks 0-4 and >= 0.08 (number's copier 4.5 lives there)
    pred_c_early_blocks_add                   B0 + B1 + B2 + B3 + B4 is within 0.05 of 0.216
    pred_d_block_8_rest_and_7_make_the_remainder  B7 + B8_REST is within 0.05 of 0.242 - 0.148 = 0.094
    pred_e_no_single_block_above_0_15         every mode here closes <= 0.15 (the early exit is spread over blocks, as number's 0.31 of small exits was)
PRICE (registered maximum): 2 batches x 8 passes = 16 forwards (43 pairs); 0 backwards; 0 fits. Bar <= 17.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import pronoun_gender_dod_natural_rows as gm
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/person_early_blocks_v581_result.json"
CANDIDATE_ID = "person.early_blocks_v581"
ROWS = ROOT / "circuits/followups/person_reflexive_dod_natural_rows_v110.json"
N_HEAD = 9
FIVE = {8: [1], 13: [1], 10: [5], 15: [1]}   # the four person readers
REPLAY_TOL, B4_MIN, EARLY_V580, ADD, REMAINDER, SINGLE_MAX = 1e-3, 0.08, 0.216, 0.05, 0.242 - 0.148, 0.15
FORWARDS_MAX = 17
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_block_4_leads_the_early_group": "B4 largest, >= 0.08", "pred_c_early_blocks_add": "0.216 +- 0.05", "pred_d_block_8_rest_and_7_make_the_remainder": "0.094 +- 0.05", "pred_e_no_single_block_above_0_15": "<= 0.15 each"}


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
    text_items = gender_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "rows": str(ROWS.name), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "b4_min": B4_MIN, "early_v580": EARLY_V580, "add": ADD, "remainder": REMAINDER, "single_max": SINGLE_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" myself"), L._single(" yourself")   # person: the margin is myself - yourself (names kept from the gender lineage)
    LOW_ALL = {l: list(range(N_HEAD)) for l in range(0, 9)}
    HIGH_ALL = {l: list(range(N_HEAD)) for l in range(9, 18)}
    MODES = {f"B{l}": ({}, {l: list(range(N_HEAD))}, False) for l in range(0, 5)}; MODES["B7"] = ({}, {7: list(range(N_HEAD))}, False); MODES["B8_REST"] = ({}, {8: [h for h in range(N_HEAD) if h != 1]}, False); forwards = 0
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
    early = {k: c[k] for k in ("B0", "B1", "B2", "B3", "B4")}; print("person early blocks", {k: round(v, 3) for k, v in c.items()}, "early sum", round(sum(early.values()), 3), "B7 + B8_REST", round(c["B7"] + c["B8_REST"], 3))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_block_4_leads_the_early_group": max(early, key=early.get) == "B4" and c["B4"] >= B4_MIN, "pred_c_early_blocks_add": abs(sum(early.values()) - EARLY_V580) <= ADD,
                   "pred_d_block_8_rest_and_7_make_the_remainder": abs(c["B7"] + c["B8_REST"] - REMAINDER) <= ADD, "pred_e_no_single_block_above_0_15": all(abs(v) <= SINGLE_MAX for v in c.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "person_early_blocks_result_v581", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
