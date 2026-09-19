#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_whole_mlp8_matches_writer_share pred_c_mlps_5_to_8_close_most_of_noun_read pred_d_mlps_5_to_8_do_not_exceed_noun_read pred_e_mlp8_share_of_5_to_8_matches_fold
"""Whole-layer replace-edits at the noun: MLP 8 alone and MLPs 5-8 (v425). v424: MLP 8's top-30 units close 0.172 of the natural-text margin gap; v406
(fold): MLP 8 writes 0.48 and MLPs 5-7 ~0.27 of the number the readers copy at the noun; v416 (edit): the readers' noun-site value swap closes 0.37. Here the
whole MLP output at the noun swapped between the pair rows, for {8}, {5, 6, 7, 8} and, as the fold predicts nothing for it, {1, 2, 3} as a low-layer control
(their number is used by the chain below MLP 5, so swapping them may still move the margin: reported, not a null). Does the edit reproduce the fold's shares?
PREDICTIONS (scored as written; failures preserved; priors from v406 / v416 / v424)
    pred_a_baseline_replays                       the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_whole_mlp8_matches_writer_share        the MLP-8 swap closes between 0.15 and 0.25 of the gap on text (fold: 0.48 x 0.37 = 0.18)
    pred_c_mlps_5_to_8_close_most_of_noun_read    the MLPs 5-8 swap closes >= 0.26 on text (fold: 0.75 x 0.37 = 0.28)
    pred_d_mlps_5_to_8_do_not_exceed_noun_read    the MLPs 5-8 swap closes <= 0.45 on text (it cannot carry more than the readers' two-site read, 0.50, minus the post-noun part)
    pred_e_mlp8_share_of_5_to_8_matches_fold      (MLP 8 closed) / (MLPs 5-8 closed) is between 0.50 and 0.80 on text (fold: 0.48 / 0.75 = 0.64). Prior: unsure
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
OUT = ROOT / "circuits/followups/layer_swap_edit_v425_result.json"
CANDIDATE_ID = "chain.layer_swap_edit_v425"
N_HEAD = 9
SETS = {"mlp8": (8,), "mlp5_8": (5, 6, 7, 8), "mlp1_3": (1, 2, 3)}
REPLAY_TOL, M8_LO, M8_HI, M58_MIN, M58_MAX, SHARE_LO, SHARE_HI = 1e-3, 0.15, 0.25, 0.26, 0.45, 0.50, 0.80
FORWARDS_MAX = 29
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_whole_mlp8_matches_writer_share": "0.15-0.25", "pred_c_mlps_5_to_8_close_most_of_noun_read": ">= 0.26", "pred_d_mlps_5_to_8_do_not_exceed_noun_read": "<= 0.45", "pred_e_mlp8_share_of_5_to_8_matches_fold": "0.50-0.80"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "m8": [M8_LO, M8_HI], "m58_min": M58_MIN, "m58_max": M58_MAX, "share": [SHARE_LO, SHARE_HI]}}
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
                m_out = block.mlp(xm)
                if mode != "native" and l in SETS[mode]:
                    m_out = m_out.clone(); m_out[idx, pp] = m_out[swap, pp].clone()      # the whole MLP write at the noun, swapped within the pair
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
    c = text["closed"]; share = c["mlp8"] / c["mlp5_8"] if abs(c["mlp5_8"]) > 1e-9 else float("nan")
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_whole_mlp8_matches_writer_share": M8_LO <= c["mlp8"] <= M8_HI, "pred_c_mlps_5_to_8_close_most_of_noun_read": c["mlp5_8"] >= M58_MIN,
                   "pred_d_mlps_5_to_8_do_not_exceed_noun_read": c["mlp5_8"] <= M58_MAX, "pred_e_mlp8_share_of_5_to_8_matches_fold": SHARE_LO <= share <= SHARE_HI}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "layer_swap_edit_result_v425", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
