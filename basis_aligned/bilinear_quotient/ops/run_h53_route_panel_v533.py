#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_5_3_routes_through_the_readers_post_noun_read pred_c_5_3_adds_to_the_readers_noun_read pred_d_5_3_replays pred_e_five_two_sites_replays
"""Does 5.3 feed the panel's pronoun readout through the readers' post-noun read (v533)? v532: on the panel 5.3's value at the post-noun token closes 0.093 of
the pronoun margin. If the readers read that relayed number at the post-noun token, adding 5.3's swap to the readers' TWO-site swap should add little (nested),
while adding it to the readers' NOUN-only swap should add nearly all of it (parallel). 48 panel pairs: 5.3 alone (replay), readers two-site (replay 0.639),
readers two-site + 5.3, readers noun-only + 5.3.
PREDICTIONS (scored as written; failures preserved; priors from v419 / v531 / v532)
    pred_a_baseline_replays                       the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_5_3_routes_through_the_readers_post_noun_read  (readers two-site + 5.3) - (readers two-site) <= 0.5 x 0.093
    pred_c_5_3_adds_to_the_readers_noun_read      (readers noun-only + 5.3) is within 0.03 of 0.340 + 0.093
    pred_d_5_3_replays                            5.3 alone closes 0.093 +- 0.02
    pred_e_five_two_sites_replays                 the readers' two-site swap closes 0.639 +- 0.03
PRICE (registered maximum): 3 panel batches x 5 passes = 15 forwards; 0 backwards; 0 fits. Bar <= 16.
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
OUT = ROOT / "circuits/followups/h53_route_panel_v533_result.json"
CANDIDATE_ID = "both_ends.h53_route_panel_v533"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, H53_V532, FIVEN_PANEL, FIVE2_PANEL, BAND, BAND2 = 1e-3, 0.093, 0.340, 0.639, 0.02, 0.03
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_5_3_routes_through_the_readers_post_noun_read": "<= 0.0465 added", "pred_c_5_3_adds_to_the_readers_noun_read": "0.433 +- 0.03", "pred_d_5_3_replays": "0.093 +- 0.02", "pred_e_five_two_sites_replays": "0.639 +- 0.03"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "h53_v532": H53_V532, "fiven_panel": FIVEN_PANEL, "five2_panel": FIVE2_PANEL, "band": BAND, "band2": BAND2}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    RD = dict(READERS)
    JOINTS = {"FIVE2_53": {**{l: (hs, (0, 1)) for l, hs in FIVE.items()}, 5: ([3], (1,))}, "FIVEN_53": {**{l: (hs, (0,)) for l, hs in FIVE.items()}, 5: ([3], (1,))}}
    MODES = {"H53_N1": ({5: [3]}, (1,)), "FIVE_2": ({l: hs for l, hs in FIVE.items()}, (0, 1)), **{k: ("JOINT", None) for k in JOINTS}}; forwards = 0
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
                JOINT = JOINTS.get(mode, {})
                if mode != "native" and ((MODES[mode][0] == "JOINT" and l in JOINT) or (MODES[mode][0] != "JOINT" and l in MODES[mode][0])):
                    hs, offs = JOINT[l] if MODES[mode][0] == "JOINT" else (MODES[mode][0][l], MODES[mode][1])
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
        res = {m: [] for m in ("native", *MODES)}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res: res[m].append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m))
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        out = {m: closed(res[m]) for m in MODES}
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "closed": out, "native_margins": nat.tolist()}, nat
    panel, nat_p = study(panel_items, 16); text = panel; nat_t = nat_p
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in panel_items[:32]] + [list(s) for _, s, _ in panel_items[:32]]; seqs = [q + [0] * (max(len(t) for t in seqs) - len(q)) for q in seqs]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        fin_ = torch.tensor([len(p) - 1 for p, _, _ in panel_items[:32]] * 2, device="cuda"); z = hook["z"].float()[torch.arange(64, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    cp = panel["closed"]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_5_3_routes_through_the_readers_post_noun_read": cp["FIVE2_53"] - cp["FIVE_2"] <= 0.5 * H53_V532, "pred_c_5_3_adds_to_the_readers_noun_read": abs(cp["FIVEN_53"] - (FIVEN_PANEL + H53_V532)) <= BAND2,
                   "pred_d_5_3_replays": abs(cp["H53_N1"] - H53_V532) <= BAND, "pred_e_five_two_sites_replays": abs(cp["FIVE_2"] - FIVE2_PANEL) <= BAND2}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "h53_route_panel_result_v533", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
