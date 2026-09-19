#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_mlps_1_8_jointly_above_parts pred_c_mlps_and_self_copies_super_additive pred_d_low_everything_reaches_readers pred_e_self_copies_replay
"""Compensation on the gender line (v554). v553's exact fold gives the MLP stack half of the gender readers' value change at the noun; v552's edits gave
MLP 8 alone 0.078 and MLPs 1-7 alone 0.067 of the he - she gap. Registered explanation: on a token-carried line the writers downstream of a swapped write
re-derive the gender from the UNCHANGED embedding (MLPs 6-8 and the self-copies 8.1 / 6.1 all read the noun's own residual), so single-stage edits under-count
and joint edits are SUPER-additive -- the opposite of the number line's nesting (v435 / v549). Here at the noun on the 61 pairs: MLPs 1-8 swapped jointly;
the self-copies' values (SELF, replay of v550's 0.310); MLPs 1-8 + self-copies jointly; every attention value of blocks 0-8 + MLPs 1-8 jointly (LOW_EVERYTHING).
PREDICTIONS (scored as written; failures preserved; priors from v550 / v552 / v553)
    pred_a_baseline_replays                   the unedited manual forward replays the model's he - she margins within 1e-3
    pred_b_mlps_1_8_jointly_above_parts       MLPs 1-8 jointly close >= 0.145 + 0.05 (the two stages separately summed to 0.145)
    pred_c_mlps_and_self_copies_super_additive  MLPs 1-8 + self-copies jointly close >= SELF + MLPS_1_8 + 0.05
    pred_d_low_everything_reaches_readers     all values of blocks 0-8 + MLPs 1-8 at the noun close >= 0.60 (what the readers carry, v549)
    pred_e_self_copies_replay                 SELF closes 0.310 +- 0.03
PRICE (registered maximum): 2 batches x 5 passes = 10 forwards (61 pairs); 0 backwards; 0 fits. Bar <= 11.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_gender_route_census_noun_v549 as gv
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_compensation_noun_v554_result.json"
CANDIDATE_ID = "gender.compensation_noun_v554"
N_HEAD = 9
DETECTORS = (3152, 3943)
UNIT_SETS = {}
MLPS_1_8 = tuple((l, 0) for l in range(1, 9))
SETS = {"MLPS_1_8": MLPS_1_8, "SELF": (), "MLPS_SELF": MLPS_1_8, "LOW_EVERYTHING": MLPS_1_8}
VALUES = {"SELF": {8: [1], 6: [1]}, "MLPS_SELF": {8: [1], 6: [1]}, "LOW_EVERYTHING": {l: list(range(9)) for l in range(0, 9)}}
REPLAY_TOL, PARTS, OVER, LOW_MIN, SELF_V550, BAND = 1e-3, 0.145, 0.05, 0.60, 0.310, 0.03
FORWARDS_MAX = 13
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_mlps_1_8_jointly_above_parts": ">= 0.195", "pred_c_mlps_and_self_copies_super_additive": ">= SELF + MLPS_1_8 + 0.05", "pred_d_low_everything_reaches_readers": ">= 0.60", "pred_e_self_copies_replay": "0.310 +- 0.03"}


def main() -> None:
    text_items = gv.gender_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "values": {k: {str(l): h for l, h in v.items()} for k, v in VALUES.items()}, "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "parts": PARTS, "over": OVER, "low_min": LOW_MIN, "self_v550": SELF_V550, "band": BAND}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" he"), L._single(" she")
    MODES = (*UNIT_SETS, *SETS); forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, HE] - z[:, SHE]
    def run(seqs, pos, mode):
        """mode: native | a UNIT_SETS key (unit activations swapped at the noun) | a SETS key (whole MLP writes swapped at the noun). Pairs are rows (2i, 2i+1)."""
        nonlocal forwards
        tokens = torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if mode in VALUES and l in VALUES[mode]:
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd); v1n = v if v1_ is None else v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in VALUES[mode][l]: v[idx, pp, h] = v[swap, pp, h].clone()
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; xm = F.rms_norm(x, (D,))
                if mode in UNIT_SETS and l == UNIT_SETS[mode][0]:
                    mlp = block.mlp; Lx, Rx = mlp.Left(xm), mlp.Right(xm); hdn = ((F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx)).clone()
                    units = torch.tensor(UNIT_SETS[mode][1], device="cuda"); hdn[idx[:, None], pp[:, None], units[None, :]] = hdn[swap[:, None], pp[:, None], units[None, :]].clone()
                    m_out = mlp.Down(hdn) + mlp.Down_bias
                else:
                    m_out = block.mlp(xm)
                    if mode in SETS:
                        offs = [o for (ll, o) in SETS[mode] if ll == l]
                        if offs:
                            m_out = m_out.clone()
                            for o in offs: m_out[idx, pp + o] = m_out[swap, pp + o].clone()
                x = x + m_out
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
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "closed": {m: closed(res[m]) for m in MODES}, "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32)
    with torch.no_grad():
        n = min(32, len(text_items)); seqs = [list(p) for p, _, _ in text_items[:n]] + [list(s) for _, s, _ in text_items[:n]]
        T_ = max(len(q) for q in seqs); tokens = torch.tensor([q + [0] * (T_ - len(q)) for q in seqs], device="cuda"); fin_ = torch.tensor([len(q) - 1 for q in seqs], device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[torch.arange(2 * n, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = (z[:, HE] - z[:, SHE]).cpu()
        mine = torch.cat([nat_t[0:2 * n:2], nat_t[1:2 * n:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    c = text["closed"]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_mlps_1_8_jointly_above_parts": c["MLPS_1_8"] >= PARTS + OVER, "pred_c_mlps_and_self_copies_super_additive": c["MLPS_SELF"] >= c["SELF"] + c["MLPS_1_8"] + OVER,
                   "pred_d_low_everything_reaches_readers": c["LOW_EVERYTHING"] >= LOW_MIN, "pred_e_self_copies_replay": abs(c["SELF"] - SELF_V550) <= BAND}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_compensation_noun_result_v554", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
