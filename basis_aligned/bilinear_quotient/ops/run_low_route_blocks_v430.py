#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_block4_is_the_largest_low_exit pred_c_blocks_0_1_carry_a_share pred_d_singles_add_to_the_half pred_e_no_low_block_is_negative
"""The low route by block (v430). v429: values at the noun for blocks 0-8 close 0.51 of the margin gap (text); v418: the copier 4.5's value alone 0.19.
Here each block 0..8 on its own (every head's value at the noun, swapped within the pair), natural pairs only, to name the low exits and test additivity.
PREDICTIONS (scored as written; failures preserved; priors from v418 / v429)
    pred_a_baseline_replays          the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_block4_is_the_largest_low_exit  block 4 (the copier's block) has the largest single-block effect among blocks 0-8. Prior: likely
    pred_c_blocks_0_1_carry_a_share  blocks 0 and 1 together close >= 0.08 (the context read that lets MLP 1 convert its lookup, section 4.10). Prior: unsure
    pred_d_singles_add_to_the_half   the nine single-block effects sum to within 0.15 of the joint 0.511 (v429). Prior: unsure
    pred_e_no_low_block_is_negative  no single block closes <= -0.03 (no block carries number against the route)
PRICE (registered maximum): 4 text batches x 10 passes = 40 forwards; 0 backwards; 0 fits. Bar <= 41.
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
OUT = ROOT / "circuits/followups/low_route_blocks_v430_result.json"
CANDIDATE_ID = "both_ends.low_route_blocks_v430"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1))
REPLAY_TOL, B01_MIN, ADD_TOL, LOW_V429, NEG_MIN = 1e-3, 0.08, 0.15, 0.511, -0.03
FORWARDS_MAX = 41
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_block4_is_the_largest_low_exit": "block 4 first", "pred_c_blocks_0_1_carry_a_share": ">= 0.08", "pred_d_singles_add_to_the_half": "within 0.15 of 0.511", "pred_e_no_low_block_is_negative": "> -0.03 x 9"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "b01_min": B01_MIN, "add_tol": ADD_TOL, "low_v429": LOW_V429, "neg_min": NEG_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    MODES = {f"b{b}": b for b in range(9)}; forwards = 0
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
                if mode != "native":
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    T_ = tokens.shape[1]; ar = torch.arange(T_, device="cuda")[None, :]; sel = (ar == pp[:, None])[:, :, None, None]
                    if l == MODES[mode]:
                        v = torch.where(sel, v[swap], v)                                              # every head's values at the noun, this block only
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
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "closed": out, "sum_singles": sum(out.values()), "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32); panel = {"skipped": "text only (price)"}
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": panel}
    print(json.dumps(report, indent=1))
    c = text["closed"]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_block4_is_the_largest_low_exit": max(c, key=c.get) == "b4", "pred_c_blocks_0_1_carry_a_share": c["b0"] + c["b1"] >= B01_MIN,
                   "pred_d_singles_add_to_the_half": abs(text["sum_singles"] - LOW_V429) <= ADD_TOL, "pred_e_no_low_block_is_negative": all(v > NEG_MIN for v in c.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "low_route_blocks_result_v430", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
