#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_readers_and_copier_add pred_c_readers_and_low_blocks_reach_all pred_d_readers_and_low_blocks_add pred_e_holds_on_panel
"""The route census at the noun only (v436). v435: joint swaps of nested routes cancel (a swap is an involution), so the census must take every exit at
ONE position. All exits of the noun are values at the noun (keys 0.0002, v429); all of them together 0.982 (v428); the five readers 0.483 (v433); the copier
4.5 0.187 (v418); every head of blocks 0-8 0.511 (v429). Here, all at the noun: five readers + 4.5; five readers + all of blocks 0-8; and all of blocks
0-8 + all of blocks 9-17 (the v428 replay). Exits at one position are independent sources, so they should add.
PREDICTIONS (scored as written; failures preserved; priors from v418 / v428 / v429 / v433)
    pred_a_baseline_replays               the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_readers_and_copier_add         five readers + 4.5 at the noun is within 0.05 of 0.483 + 0.187 on text
    pred_c_readers_and_low_blocks_reach_all  five readers + blocks 0-8 at the noun closes >= 0.93 on text
    pred_d_readers_and_low_blocks_add     five readers + blocks 0-8 is within 0.08 of 0.483 + 0.511 on text
    pred_e_holds_on_panel                 five readers + blocks 0-8 closes >= 0.90 on the panel (0.340 + 0.660 = 1.00)
PRICE (registered maximum): (4 text + 3 panel batches) x 4 passes = 28 forwards; 0 backwards; 0 fits. Bar <= 29.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/route_census_noun_v436_result.json"
CANDIDATE_ID = "both_ends.route_census_noun_v436"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, FIVE_N, COPIER, LOW, ADD_RC, ADD_RL, ALL_MIN, PANEL_MIN = 1e-3, 0.483, 0.187, 0.511, 0.05, 0.08, 0.93, 0.90
FORWARDS_MAX = 29
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_readers_and_copier_add": "0.670 +- 0.05", "pred_c_readers_and_low_blocks_reach_all": ">= 0.93", "pred_d_readers_and_low_blocks_add": "0.994 +- 0.08", "pred_e_holds_on_panel": ">= 0.90"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "five_n": FIVE_N, "copier": COPIER, "low": LOW, "add_rc": ADD_RC, "add_rl": ADD_RL, "all_min": ALL_MIN, "panel_min": PANEL_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    RD = dict(READERS)
    LOW_ALL = {l: list(range(N_HEAD)) for l in range(0, 9)}
    HIGH_ALL = {l: list(range(N_HEAD)) for l in range(9, 18)}
    MODES = {"R_C": ({}, {**FIVE, 4: [5]}, False), "R_LOW": ({}, {**LOW_ALL, **FIVE}, False), "LOW_HIGH": ({}, {**LOW_ALL, **HIGH_ALL}, False)}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, THEY] - z[:, HE]
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
        res = {m: [] for m in ("native", "R_C", "R_LOW", "LOW_HIGH")}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res: res[m].append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m))
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        out = {m: closed(res[m]) for m in ("R_C", "R_LOW", "LOW_HIGH")}
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "closed": out, "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32); panel, nat_p = study(panel_items, 16)
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    c, cp = text["closed"], panel["closed"]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_readers_and_copier_add": abs(c["R_C"] - (FIVE_N + COPIER)) <= ADD_RC, "pred_c_readers_and_low_blocks_reach_all": c["R_LOW"] >= ALL_MIN,
                   "pred_d_readers_and_low_blocks_add": abs(c["R_LOW"] - (FIVE_N + LOW)) <= ADD_RL, "pred_e_holds_on_panel": cp["R_LOW"] >= PANEL_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "route_census_noun_result_v436", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
