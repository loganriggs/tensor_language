#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_mlp_17_alone_carries_the_detector pred_c_mlp_8_alone_is_small pred_d_singles_sub_additive pred_e_mlps_9_16_below_mlp_17
"""The direct path's MLPs by stage (v542). v501: at the noun's own position on the 43 adjacent rows MLPs 1-8 swapped whole close 0.367, MLPs 9-17 0.426,
MLPs 1-17 0.452 (nested: the late layers recompute from the early ones). v539-v541 closed the attention map at that position (0.313: five verb heads 0.167,
blocks 8 / 10 0.045, late 0.032). Here the MLP side split by the named components: MLP 8 alone (the hub 829 / 953 / 1030), MLP 17 alone (the detector
701 / 2059, verb-specific, at the answer position -- here the noun itself), MLPs 1-7 (the chain below the hub) and MLPs 9-16 (the relays to the detector).
PREDICTIONS (scored as written; failures preserved; priors from v465-v469 (701 moved), v501)
    pred_a_baseline_replays               the unedited manual forward replays the model's verb margins within 1e-3
    pred_b_mlp_17_alone_carries_the_detector  MLP 17 swapped alone at the noun closes >= 0.15 (the detector's write is the largest single MLP item)
    pred_c_mlp_8_alone_is_small           MLP 8 swapped alone at the noun closes <= 0.10 (the hub is rebuilt downstream; at the pronoun three units gave 0.12 through readers)
    pred_d_singles_sub_additive           MLP 17 + MLP 8 + MLPs 1-7 + MLPs 9-16 > 0.452 + 0.05 (the four disjoint stages over-count the nested 1-17)
    pred_e_mlps_9_16_below_mlp_17         MLPs 9-16 close less than MLP 17 alone. Prior: unsure.
PRICE (registered maximum): 2 batches x 5 passes = 10 forwards; 0 backwards; 0 fits. Bar <= 11.
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
OUT = ROOT / "circuits/followups/direct_path_mlp_singles_v542_result.json"
CANDIDATE_ID = "chain.direct_path_mlp_singles_v542"
N_HEAD = 9
SETS = {"mlp_17": ((17, 0),), "mlp_8": ((8, 0),), "mlps_1_7": tuple((l, 0) for l in range(1, 8)), "mlps_9_16": tuple((l, 0) for l in range(9, 17))}
REPLAY_TOL, M17_MIN, M8_MAX, ALL_V501, OVER = 1e-3, 0.15, 0.10, 0.452, 0.05
FORWARDS_MAX = 11
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_mlp_17_alone_carries_the_detector": ">= 0.15", "pred_c_mlp_8_alone_is_small": "<= 0.10", "pred_d_singles_sub_additive": "sum > 0.502", "pred_e_mlps_9_16_below_mlp_17": "9-16 < 17"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]]; E0 = L.ENCODING
    def partner(tok):
        t = E0.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E0.encode(c)) == 1: return E0.encode(c)[0]
        return None
    text_items = []
    for r_ in recs:
        c = r_["cue_offset"]; alt = partner(r_["ids"][c]); vs = r_["second_offset"]
        if alt is None or vs - c != 1: continue
        sw = list(r_["ids"]); sw[c] = alt; plural, singular = (r_["ids"], sw) if r_["cue"] == "plural" else (sw, r_["ids"])
        text_items.append((plural[:vs], singular[:vs], c))                # adjacent: truncated at the verb slot, the answer position is the noun
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "m17_min": M17_MIN, "m8_max": M8_MAX, "all_v501": ALL_V501, "over": OVER}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; ARE, IS, WERE, WAS = L._single(" are"), L._single(" is"), L._single(" were"), L._single(" was")
    forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return (z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS])
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
    text, nat_t = study(text_items, 32); panel = {"skipped": "adjacent natural rows only", "rows": len(text_items)}
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; seqs = [q + [0] * (max(len(t) for t in seqs) - len(q)) for q in seqs]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        fin_ = torch.tensor([len(p) - 1 for p, _, _ in text_items[:32]] * 2, device="cuda"); z = hook["z"].float()[torch.arange(64, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = ((z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS])).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": panel}
    print(json.dumps(report, indent=1))
    c = text["closed"]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_mlp_17_alone_carries_the_detector": c["mlp_17"] >= M17_MIN, "pred_c_mlp_8_alone_is_small": abs(c["mlp_8"]) <= M8_MAX,
                   "pred_d_singles_sub_additive": sum(c.values()) > ALL_V501 + OVER, "pred_e_mlps_9_16_below_mlp_17": c["mlps_9_16"] < c["mlp_17"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "direct_path_mlp_singles_result_v542", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
