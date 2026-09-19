#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_all_mlps_at_noun_reach_the_noun_read pred_c_chain_is_not_additive pred_d_postnoun_rebuild_carries_the_other_site pred_e_both_sites_reach_the_two_site_read
"""The MLP chain edited whole, at both sites (v426). v425: at the noun, MLP 8 0.22, MLPs 5-8 0.235, MLPs 1-3 0.27 — the chain is serial, so the shares do
not add. v416 / v417: the readers' value swap closes 0.37 (noun site) / 0.50 (both sites). Here: all MLPs 1-8 at the noun; MLPs 4-8 at the post-noun token
(the rebuild, v383-v385); and both together. Does swapping the MLP chain's writes at the two sites reproduce what swapping the readers' values does?
PREDICTIONS (scored as written; failures preserved; priors from v416 / v417 / v425)
    pred_a_baseline_replays                      the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_all_mlps_at_noun_reach_the_noun_read  MLPs 1-8 at the noun close >= 0.30 of the gap on text (the readers' noun-site read is 0.37; the embedding's 0.10-0.14 is left out)
    pred_c_chain_is_not_additive                 MLPs 1-8 at the noun close <= 0.80 x (0.266 + 0.235) on text (v425's parts overlap)
    pred_d_postnoun_rebuild_carries_the_other_site  MLPs 4-8 at the post-noun token close >= 0.08 on text (the readers' post-noun read added 0.13 in v417). Prior: unsure
    pred_e_both_sites_reach_the_two_site_read    both sites together close >= 0.40 on text (readers' two-site 0.50 less the embedding parts). Prior: unsure
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
OUT = ROOT / "circuits/followups/layer_swap_sites_v426_result.json"
CANDIDATE_ID = "chain.layer_swap_sites_v426"
N_HEAD = 9
SETS = {"noun_1_8": tuple((l, 0) for l in range(1, 9)), "post_4_8": tuple((l, 1) for l in range(4, 9)), "both": tuple((l, 0) for l in range(1, 9)) + tuple((l, 1) for l in range(4, 9))}
REPLAY_TOL, NOUN_MIN, ADD_MAX, POST_MIN, BOTH_MIN = 1e-3, 0.30, 0.80 * (0.266 + 0.235), 0.08, 0.40
FORWARDS_MAX = 29
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_all_mlps_at_noun_reach_the_noun_read": ">= 0.30", "pred_c_chain_is_not_additive": "<= 0.8 x 0.501", "pred_d_postnoun_rebuild_carries_the_other_site": ">= 0.08", "pred_e_both_sites_reach_the_two_site_read": ">= 0.40"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "noun_min": NOUN_MIN, "add_max": ADD_MAX, "post_min": POST_MIN, "both_min": BOTH_MIN}}
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
                if mode != "native":
                    offs = [o for (ll, o) in SETS[mode] if ll == l]
                    if offs:
                        m_out = m_out.clone()
                        for o in offs: m_out[idx, pp + o] = m_out[swap, pp + o].clone()      # the whole MLP write at that site, swapped within the pair
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_all_mlps_at_noun_reach_the_noun_read": c["noun_1_8"] >= NOUN_MIN, "pred_c_chain_is_not_additive": c["noun_1_8"] <= ADD_MAX,
                   "pred_d_postnoun_rebuild_carries_the_other_site": c["post_4_8"] >= POST_MIN, "pred_e_both_sites_reach_the_two_site_read": c["both"] >= BOTH_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "layer_swap_sites_result_v426", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
