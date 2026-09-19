#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_early_mlps_carry_half_of_the_direct_path pred_c_late_mlps_carry_a_third pred_d_all_mlps_carry_most pred_e_late_below_early
"""The direct path's MLPs edited whole (v501). For a verb directly after the noun the answer position is the noun and its own residual carries the agreement
(values at the noun 0.26, v446 / v500; writers across depth by fold, v455). Here the noun position's MLP writes swapped whole within the pair on the 43
adjacent rows, scored on the agreement margin: MLPs 1-8 (the chain to the hub), MLPs 9-17 (the late layers to the detector), and MLPs 1-17 together
(nested, v435).
PREDICTIONS (scored as written; failures preserved; priors from v455 / v460 / v461)
    pred_a_baseline_replays                   the unedited manual forward replays the model's verb margins within 1e-3
    pred_b_early_mlps_carry_half_of_the_direct_path  MLPs 1-8 at the noun close >= 0.40 of the adjacent-row agreement margin gap. Prior: unsure
    pred_c_late_mlps_carry_a_third            MLPs 9-17 at the noun close >= 0.25 (fold: late MLPs 0.34 of the direct change)
    pred_d_all_mlps_carry_most                MLPs 1-17 at the noun close >= 0.60
    pred_e_late_below_early                   MLPs 9-17 close less than MLPs 1-8. Prior: unsure
PRICE (registered maximum): 2 batches x 4 passes = 8 forwards; 0 backwards; 0 fits. Bar <= 9.
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
OUT = ROOT / "circuits/followups/direct_path_mlps_v501_result.json"
CANDIDATE_ID = "chain.direct_path_mlps_v501"
N_HEAD = 9
SETS = {"early_1_8": tuple((l, 0) for l in range(1, 9)), "late_9_17": tuple((l, 0) for l in range(9, 18)), "all_1_17": tuple((l, 0) for l in range(1, 18))}
REPLAY_TOL, EARLY_MIN, LATE_MIN, ALL_MIN = 1e-3, 0.40, 0.25, 0.60
FORWARDS_MAX = 9
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_early_mlps_carry_half_of_the_direct_path": ">= 0.40", "pred_c_late_mlps_carry_a_third": ">= 0.25", "pred_d_all_mlps_carry_most": ">= 0.60", "pred_e_late_below_early": "late < early"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "early_min": EARLY_MIN, "late_min": LATE_MIN, "all_min": ALL_MIN}}
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_early_mlps_carry_half_of_the_direct_path": c["early_1_8"] >= EARLY_MIN, "pred_c_late_mlps_carry_a_third": c["late_9_17"] >= LATE_MIN,
                   "pred_d_all_mlps_carry_most": c["all_1_17"] >= ALL_MIN, "pred_e_late_below_early": c["late_9_17"] < c["early_1_8"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "direct_path_mlps_result_v501", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
