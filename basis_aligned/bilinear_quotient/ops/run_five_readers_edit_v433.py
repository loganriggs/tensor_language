#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_five_heads_are_the_high_route pred_c_five_heads_two_sites_exceed_three pred_d_three_heads_two_sites_replay pred_e_holds_on_panel
"""The five-head reader set edited (v433). v431 / v432: the high route out of the noun is blocks 9 (0.276), 10 (0.128 = 10.1 + 10.5) and 12 (0.089), with
15 small (0.024); the three-head set 9.6 / 12.4 / 15.1 closes 0.368 at the noun and 0.498 at both sites (v416 / v417). Here the five-head set
{9.6, 10.1, 10.5, 12.4, 15.1}: values at the noun (FIVE_N), values at both sites (FIVE_2), and the old three-head two-site swap as the replay (THREE_2).
PREDICTIONS (scored as written; failures preserved; priors from v417 / v429 / v432)
    pred_a_baseline_replays                 the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_five_heads_are_the_high_route    FIVE_N closes >= 0.90 x 0.505 (all values of blocks 9-17 at the noun, v429) on text
    pred_c_five_heads_two_sites_exceed_three  FIVE_2 >= THREE_2 + 0.05 on text
    pred_d_three_heads_two_sites_replay     THREE_2 is within 0.03 of 0.498 (v417) on text
    pred_e_holds_on_panel                   FIVE_N >= 0.90 x 0.330 (v429 panel) on the panel
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
OUT = ROOT / "circuits/followups/five_readers_edit_v433_result.json"
CANDIDATE_ID = "both_ends.five_readers_edit_v433"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, HIGH_TEXT, HIGH_PANEL, ROUTE_MIN, GAIN_MIN, THREE_V417, REPLAY_BAND = 1e-3, 0.505, 0.330, 0.90, 0.05, 0.498, 0.03
FORWARDS_MAX = 29
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_five_heads_are_the_high_route": ">= 0.9 x 0.505", "pred_c_five_heads_two_sites_exceed_three": ">= THREE_2 + 0.05", "pred_d_three_heads_two_sites_replay": "0.498 +- 0.03", "pred_e_holds_on_panel": ">= 0.9 x 0.330"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "route_min": ROUTE_MIN, "gain_min": GAIN_MIN, "three_v417": THREE_V417, "replay_band": REPLAY_BAND}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    RD = dict(READERS)
    MODES = {"FIVE_N": ({l: hs for l, hs in FIVE.items()}, (0,)), "FIVE_2": ({l: hs for l, hs in FIVE.items()}, (0, 1)), "THREE_2": ({l: [RD[l]] for l in RD}, (0, 1))}; forwards = 0
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
                if mode != "native" and l in MODES[mode][0]:
                    hs, offs = MODES[mode][0][l], MODES[mode][1]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in hs:
                        for o in offs: v[idx, pp + o, h] = v[swap, pp + o, h].clone()
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
            return logits_margin(x[idx, fin]).cpu()
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        res = {m: [] for m in ("native", "FIVE_N", "FIVE_2", "THREE_2")}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res: res[m].append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m))
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        out = {m: closed(res[m]) for m in ("FIVE_N", "FIVE_2", "THREE_2")}
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_five_heads_are_the_high_route": c["FIVE_N"] >= ROUTE_MIN * HIGH_TEXT, "pred_c_five_heads_two_sites_exceed_three": c["FIVE_2"] >= c["THREE_2"] + GAIN_MIN,
                   "pred_d_three_heads_two_sites_replay": abs(c["THREE_2"] - THREE_V417) <= REPLAY_BAND, "pred_e_holds_on_panel": cp["FIVE_N"] >= ROUTE_MIN * HIGH_PANEL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "five_readers_edit_result_v433", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
