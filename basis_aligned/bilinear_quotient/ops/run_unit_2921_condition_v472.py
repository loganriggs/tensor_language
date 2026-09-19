#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_2921_anticorrelates_with_701 pred_c_2921_tracks_the_margin_gap pred_d_2921_row_conditional pred_e_701_gap_corr_replays
"""Is 2921's row-conditionality the detector's (v472)? v466: 701's plural - singular change at a distant verb tracks the agreement margin gap (corr 0.53) and
flips on 39% of rows. v471: MLP 16's unit 2921 leads the feed into 701's R input at the verb, with a negative pooled term and its own sign constant on 0.61 of
rows. Native pass, 79 distant rows: 2921's change at the verb against 701's and against the margin gap.
PREDICTIONS (scored as written; failures preserved; priors from v466 / v471)
    pred_a_baseline_replays          the unedited manual forward replays the model's verb margins within 1e-3
    pred_b_2921_anticorrelates_with_701  corr over rows between 2921's change and 701's change <= -0.40 (2921 writes negatively into 701's R). Prior: unsure
    pred_c_2921_tracks_the_margin_gap  |corr| between 2921's change and the margin gap >= 0.30
    pred_d_2921_row_conditional      2921's change has one sign on <= 0.75 of rows (it is conditional like 701, not a fixed detector)
    pred_e_701_gap_corr_replays      701's change vs the margin gap replays 0.53 within 0.05
PRICE (registered maximum): 3 batches x 1 pass = 3 forwards; 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/unit_2921_condition_v472_result.json"
CANDIDATE_ID = "chain.unit_2921_condition_v472"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, ANTI_MAX, CORR_MIN, SIGN_MAX, V466_CORR, BAND, UNIT, UNIT16 = 1e-3, -0.40, 0.30, 0.75, 0.527, 0.05, 701, 2921
VERBS = (" is", " are", " was", " were", " has", " have", " does", " do")
UNITS = (829, 953, 1030); MLP_LAYER = 8
SING_V = (" is", " was", " has", " does"); PLUR_V = (" are", " were", " have", " do")
_BASE = ["they", "their", "them", "themselves", "we", "our", "us", "ourselves", "he", "his", "him", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "I", "my", "me", "myself", "you", "your", "yourself"]
def _family():
    out = {}
    for w in _BASE:
        for form in (w, w.capitalize()):
            for sp in ("", " "):
                ids = L.ENCODING.encode(sp + form)
                if len(ids) == 1: out[ids[0]] = w
    return out
PLURAL3 = {"they", "their", "them", "themselves"}; SING3 = {"he", "his", "him", "himself", "she", "her", "herself"}
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_2921_anticorrelates_with_701": "corr <= -0.40", "pred_c_2921_tracks_the_margin_gap": "|corr| >= 0.30", "pred_d_2921_row_conditional": "<= 0.75 one sign", "pred_e_701_gap_corr_replays": "0.53 +- 0.05"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    E0 = L.ENCODING
    def partner(tok):
        t = E0.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E0.encode(c)) == 1: return E0.encode(c)[0]
        return None
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]]; text_items, verb_items = [], []
    for r_ in recs:
        c = r_["cue_offset"]; alt = partner(r_["ids"][c])
        if alt is None: continue
        sw = list(r_["ids"]); sw[c] = alt; plural, singular = (r_["ids"], sw) if r_["cue"] == "plural" else (sw, r_["ids"])
        text_items.append((plural, singular, c))
        vs = r_["second_offset"]
        if vs > c: verb_items.append((plural[:vs], singular[:vs], c, vs - c))   # truncated so the answer position IS the verb slot (the verb is predicted there)
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "anti_max": ANTI_MAX, "corr_min": CORR_MIN, "sign_max": SIGN_MAX, "v466_corr": V466_CORR, "band": BAND, "units": [UNIT, UNIT16]}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; ARE, IS, WERE, WAS = L._single(" are"), L._single(" is"), L._single(" were"), L._single(" was")
    RD = dict(READERS)
    ALL_H = {l: list(range(N_HEAD)) for l in range(18)}
    V4 = {11: [3], 7: [8], 13: [1], 9: [7]}; ALL_H = {l: list(range(N_HEAD)) for l in range(18)}
    MODES = {}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        VI = torch.tensor([L._single(t) for t in VERBS], device=z.device); pv = torch.softmax(z, -1)[:, VI].sum(1)
        return torch.stack([(z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS]), pv, z[:, WERE] - z[:, WAS]], 1), z
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
                x = live + attention; xm = F.rms_norm(x, (D,))
                if l == 16:
                    mlp = block.mlp; Lx, Rx = mlp.Left(xm), mlp.Right(xm); h16 = ((F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx))[idx, fin, UNIT16].float().cpu()
                if l == 17:
                    mlp = block.mlp; Lx, Rx = mlp.Left(xm), mlp.Right(xm); hh = ((F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx)); h17 = torch.stack([hh[idx, fin, UNIT], h16.to(hh.device)], 1).float().cpu()   # 701 (MLP 17) and 2921 (MLP 16) at the answer position
                x = x + block.mlp(xm)
            forwards += 1
            m_, lp = logits_margin(x[idx, fin]); return m_.cpu(), h17
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        nat_m, nat_h = [], []
        for s0 in range(0, len(seqs), 2 * batch):
            m_, h = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native"); nat_m.append(m_); nat_h.append(h)
        nat_m, nat_h = torch.cat(nat_m), torch.cat(nat_h)
        gap = nat_m[0::2, 0] - nat_m[1::2, 0]; d701 = nat_h[0::2, 0] - nat_h[1::2, 0]; d2921 = nat_h[0::2, 1] - nat_h[1::2, 1]
        corr = lambda a_, b_: float(((a_ - a_.mean()) * (b_ - b_.mean())).mean() / (a_.std(unbiased=False) * b_.std(unbiased=False)))
        sign = lambda d: float(max((d > 0).float().mean(), (d < 0).float().mean()))
        return {"pairs": len(items), "corr_d701_gap": corr(d701, gap), "corr_d2921_d701": corr(d2921, d701), "corr_d2921_gap": corr(d2921, gap), "sign_2921": sign(d2921), "sign_701": sign(d701), "mean_d2921": float(d2921.mean()), "native_margins": nat_m[:, 0].tolist()}, nat_m[:, 0]
    dist_items = [(p, s_, c) for p, s_, c, d in verb_items if d >= 2]
    dist, nat_t = study(dist_items, 32); verb = dist; text = dist; panel = {"skipped": "distant rows only", "distant_rows": len(dist_items)}; verb_items = dist_items
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in verb_items[:32]] + [list(s) for _, s, _ in verb_items[:32]]; seqs = [q + [0] * (max(len(t) for t in seqs) - len(q)) for q in seqs]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        fin_ = torch.tensor([len(p) - 1 for p, _, _ in verb_items[:32]] * 2, device="cuda"); z = hook["z"].float()[torch.arange(64, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = ((z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS])).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": panel}
    print(json.dumps(report, indent=1))
    report["verb_slot"] = {k: v for k, v in verb.items() if k != "native_margins"}
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_2921_anticorrelates_with_701": verb["corr_d2921_d701"] <= ANTI_MAX, "pred_c_2921_tracks_the_margin_gap": abs(verb["corr_d2921_gap"]) >= CORR_MIN,
                   "pred_d_2921_row_conditional": verb["sign_2921"] <= SIGN_MAX, "pred_e_701_gap_corr_replays": abs(verb["corr_d701_gap"] - V466_CORR) <= BAND}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "unit_2921_condition_result_v472", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
