#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_three_units_move_margin pred_c_random_three_units_move_little pred_d_units_beat_random_on_most_pairs pred_e_holds_on_panel
"""The three MLP-8 units edited by replacement on natural pairs (v423). v422: 9.6 reads (0.79) what MLP 8's units 829 / 953 / 1030 write; v408: they are
the top three of MLP 8's number census on text. The replace-edit that matches the chain claim: on each aligned pair, swap ONLY those three units' hidden
activations at the noun position between the plural and singular rows (everything else recomputed natively); control: three random MLP-8 units (seed 0),
the same swap. How much of the they - he margin gap follows three numbers?
PREDICTIONS (scored as written; failures preserved; priors from v386-v388 (zeroing the named units cost 11% of the text margin) and v416)
    pred_a_baseline_replays               the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_three_units_move_margin        the three-unit swap closes >= 0.10 of the gap, pooled over text pairs. Prior: unsure
    pred_c_random_three_units_move_little three random units close <= 0.02 of the gap
    pred_d_units_beat_random_on_most_pairs  |three-unit effect| > |random effect| on >= 0.80 of the text pairs
    pred_e_holds_on_panel                 the three-unit swap closes >= 0.10 of the gap on the panel pairs
PRICE (registered maximum): (4 text + 3 panel batches) x 3 passes = 21 forwards; 0 backwards; 0 fits. Bar <= 22.
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
OUT = ROOT / "circuits/followups/unit_swap_edit_v423_result.json"
CANDIDATE_ID = "chain.unit_swap_edit_v423"
N_HEAD = 9
UNITS = (829, 953, 1030); MLP_LAYER = 8
import random
RANDOM_UNITS = tuple(random.Random(0).sample([j for j in range(4608) if j not in UNITS], 3))
REPLAY_TOL, UNIT_MIN, RANDOM_MAX, MOST_MIN = 1e-3, 0.10, 0.02, 0.80
FORWARDS_MAX = 22
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_three_units_move_margin": ">= 0.10", "pred_c_random_three_units_move_little": "<= 0.02", "pred_d_units_beat_random_on_most_pairs": ">= 0.80", "pred_e_holds_on_panel": ">= 0.10"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "units": list(UNITS), "random_units": list(RANDOM_UNITS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "unit_min": UNIT_MIN, "random_max": RANDOM_MAX, "most_min": MOST_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, THEY] - z[:, HE]
    def run(seqs, pos, mode):
        """mode: native | value (the three units) | key (three random units). Pairs are rows (2i, 2i+1)."""
        nonlocal forwards
        tokens = torch.tensor(seqs, device="cuda") if len({len(s) for s in seqs}) == 1 else torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                attention, v1_ = attn(xin, v1_); x = live + attention; xm = F.rms_norm(x, (D,))
                if l == MLP_LAYER and mode != "native":
                    mlp = block.mlp; Lx, Rx = mlp.Left(xm), mlp.Right(xm); hdn = (F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx); hdn = hdn.clone()
                    units = torch.tensor(UNITS if mode == "value" else RANDOM_UNITS, device="cuda")
                    hdn[idx[:, None], pp[:, None], units[None, :]] = hdn[swap[:, None], pp[:, None], units[None, :]].clone()
                    m_out = mlp.Down(hdn) + mlp.Down_bias
                else:
                    m_out = block.mlp(xm)
                x = x + m_out
            forwards += 1
            return logits_margin(x[idx, fin]).cpu()
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        nat, val, key = [], [], []
        for s0 in range(0, len(seqs), 2 * batch):
            nat.append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native")); val.append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "value")); key.append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "key"))
        nat, val, key = torch.cat(nat), torch.cat(val), torch.cat(key)
        gap = nat[0::2] - nat[1::2]                                   # plural margin - singular margin per pair
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))   # fraction of the gap closed, both directions pooled
        per_pair_v = ((nat[0::2] - val[0::2]) + (val[1::2] - nat[1::2])) / 2; per_pair_k = ((nat[0::2] - key[0::2]) + (key[1::2] - nat[1::2])) / 2
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "value_closed": closed(val), "key_closed": closed(key), "value_beats_key_share": float((per_pair_v.abs() > per_pair_k.abs()).float().mean()),
                "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32); panel, nat_p = study(panel_items, 16)
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_three_units_move_margin": text["value_closed"] >= UNIT_MIN, "pred_c_random_three_units_move_little": abs(text["key_closed"]) <= RANDOM_MAX,
                   "pred_d_units_beat_random_on_most_pairs": text["value_beats_key_share"] >= MOST_MIN, "pred_e_holds_on_panel": panel["value_closed"] >= UNIT_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "unit_swap_edit_result_v423", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
