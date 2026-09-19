#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_top10_closes_a_fifth pred_c_top30_closes_over_a_quarter pred_d_random30_moves_little pred_e_saturation_by_30
"""How many MLP-8 units carry the number the readers copy (v424). v423: three units (829 / 953 / 1030) swapped at the noun close 0.124 of the natural-text
margin gap; v416: the readers' noun-site value swap closes 0.37 and MLP 8 writes half of that value (v406), so the whole of MLP 8 at the noun should be
worth about 0.18-0.20. Here the same replace-edit with the top 3, 10 and 30 units of the text census (v408), and 30 random units (seed 0) as the control.
PREDICTIONS (scored as written; failures preserved; priors from v406 / v408 / v423)
    pred_a_baseline_replays          the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_top10_closes_a_fifth      the top-10 swap closes >= 0.16 of the gap on text (v408: the top 10 carry 0.75 of MLP 8's contrast). Prior: unsure
    pred_c_top30_closes_over_a_quarter  the top-30 swap closes >= 0.18 on text (the top 30 carry ~0.85 of MLP 8's contrast; MLP 8 ~ half of the readers' 0.37)
    pred_d_random30_moves_little     30 random units close <= 0.03 on text
    pred_e_saturation_by_30          top-30 minus top-10 <= 0.5 x (top-10 minus top-3) on text (diminishing returns). Prior: unsure
PRICE (registered maximum): (4 text + 3 panel batches) x 5 passes = 35 forwards; 0 backwards; 0 fits. Bar <= 36.
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
OUT = ROOT / "circuits/followups/unit_swap_count_v424_result.json"
CANDIDATE_ID = "chain.unit_swap_count_v424"
N_HEAD = 9
TOP = (829, 953, 1030, 1484, 3152, 1738, 2092, 3205, 4000, 684, 1959, 2583, 1027, 2733, 4565, 2863, 368, 2271, 1823, 980, 1101, 2766, 4509, 2225, 306, 1038, 3039, 2597, 1173, 2596); MLP_LAYER = 8
import random
RANDOM30 = tuple(random.Random(0).sample([j for j in range(4608) if j not in TOP], 30))
SETS = {"top3": TOP[:3], "top10": TOP[:10], "top30": TOP[:30], "random30": RANDOM30}
REPLAY_TOL, TOP10_MIN, TOP30_MIN, RANDOM_MAX = 1e-3, 0.16, 0.18, 0.03
FORWARDS_MAX = 36
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_top10_closes_a_fifth": ">= 0.16", "pred_c_top30_closes_over_a_quarter": ">= 0.18", "pred_d_random30_moves_little": "<= 0.03", "pred_e_saturation_by_30": "increment halves"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "top10_min": TOP10_MIN, "top30_min": TOP30_MIN, "random_max": RANDOM_MAX}}
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
                    units = torch.tensor(SETS[mode], device="cuda")
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
        res = {m: [] for m in ("native", *SETS)}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res: res[m].append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m))
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "closed": {m: closed(res[m]) for m in SETS}, "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32); panel, nat_p = study(panel_items, 16)
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    c = text["closed"]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_top10_closes_a_fifth": c["top10"] >= TOP10_MIN, "pred_c_top30_closes_over_a_quarter": c["top30"] >= TOP30_MIN, "pred_d_random30_moves_little": abs(c["random30"]) <= RANDOM_MAX,
                   "pred_e_saturation_by_30": (c["top30"] - c["top10"]) <= 0.5 * (c["top10"] - c["top3"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "unit_swap_count_result_v424", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
