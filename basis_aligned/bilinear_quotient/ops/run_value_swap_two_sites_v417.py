#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_two_site_value_swap_closes_over_half pred_c_two_site_key_swap_moves_little pred_d_value_swap_beats_key_swap_on_most_pairs pred_e_effect_holds_on_panel
"""The value-copy account edited at BOTH source sites (v417). v416: swapping the three readers' values at the noun alone closes 0.37 (text) / 0.25 (panel) of the
they - he gap; keys 0.01. v402: the readers read the noun and the token after it about equally. Here the same replace-edit at both sites (noun and post-noun),
values only vs keys only.
PREDICTIONS (scored as written; failures preserved; priors from v380 / v401 / v405)
    pred_a_baseline_replays                   the unedited manual forward replays the model's they - he margins within 1e-3 (max abs over rows)
    pred_b_two_site_value_swap_closes_over_half  the two-site value swap closes >= 0.55 of the gap, pooled over text pairs. Prior: unsure (one site 0.37; the readers write ~0.7 of the signal)
    pred_c_two_site_key_swap_moves_little     the two-site key swap closes <= 0.05 of the gap, pooled over text pairs
    pred_d_value_swap_beats_key_swap_on_most_pairs  |value-swap effect| > |key-swap effect| on >= 0.80 of the text pairs
    pred_e_effect_holds_on_panel              the two-site value swap closes >= 0.40 of the gap on the panel pairs (one site 0.25)
PRICE (registered maximum): (4 text + 3 panel batches) x 3 passes (native, value swap, key swap) = 21 forwards; 0 backwards; 0 fits. Bar <= 22.
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
OUT = ROOT / "circuits/followups/value_swap_two_sites_v417_result.json"
CANDIDATE_ID = "both_ends.value_swap_two_sites_v417"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1))
REPLAY_TOL, VALUE_MIN, KEY_MAX, MOST_MIN, PANEL_MIN = 1e-3, 0.55, 0.05, 0.80, 0.40
FORWARDS_MAX = 22
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_two_site_value_swap_closes_over_half": ">= 0.55", "pred_c_two_site_key_swap_moves_little": "<= 0.05", "pred_d_value_swap_beats_key_swap_on_most_pairs": ">= 0.80", "pred_e_effect_holds_on_panel": ">= 0.40"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "value_min": VALUE_MIN, "key_max": KEY_MAX, "most_min": MOST_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    heads_at = {l: h for l, h in READERS}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, THEY] - z[:, HE]
    def run(seqs, pos, mode):
        """mode: native | value | key. Pairs are rows (2i, 2i+1); the noun position is pos[2i] == pos[2i+1]."""
        nonlocal forwards
        tokens = torch.tensor(seqs, device="cuda") if len({len(s) for s in seqs}) == 1 else torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if l in heads_at and mode != "native":
                    h = heads_at[l]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    if mode == "value":
                        v = v.clone(); v[idx, pp, h] = v[swap, pp, h].clone(); v[idx, pp + 1, h] = v[swap, pp + 1, h].clone()   # both sites: the noun and the token after it
                    else:
                        k = k.clone(); k2 = k2.clone()
                        for site in (pp, pp + 1): k[idx, site, h] = k[swap, site, h].clone(); k2[idx, site, h] = k2[swap, site, h].clone()
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_two_site_value_swap_closes_over_half": text["value_closed"] >= VALUE_MIN, "pred_c_two_site_key_swap_moves_little": abs(text["key_closed"]) <= KEY_MAX,
                   "pred_d_value_swap_beats_key_swap_on_most_pairs": text["value_beats_key_share"] >= MOST_MIN, "pred_e_effect_holds_on_panel": panel["value_closed"] >= PANEL_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "value_swap_two_sites_result_v417", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
