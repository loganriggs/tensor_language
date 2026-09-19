#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_9_6_value_moves_by_its_fold_share pred_c_10_1_value_moves_by_its_fold_share pred_d_mlp8_margin_replays pred_e_readout_is_linear_in_the_readers
"""The MLP-8 swap, instrumented (v555). v553's exact fold gives MLP 8 0.27 / 0.15 / 0.34 of the gender readers' (10.1 / 12.4 / 9.6) value change at the noun;
v552's edit of MLP 8's whole write closes 0.078 of the he - she gap, against 0.60 for the readers together (singles 0.255 / 0.170 / 0.107, v551). v554 refuted
downstream compensation. Where is the gap -- at the noun (the readers' values move less than the fold says inside the edited forward) or after the readers
(the margin responds less than linearly to their values)? Here the same MLP-8 swap with the readers' current-state value factors w_h = (uO_h) . (1 - lamb) W_v n(live)
at the noun recorded natively and edited, on the 61 pairs; "moved" = the pair-summed change of w_h under the edit over its native male - female change.
PREDICTIONS (scored as written; failures preserved; priors from v551-v553)
    pred_a_baseline_replays                 the unedited manual forward replays the model's he - she margins within 1e-3
    pred_b_9_6_value_moves_by_its_fold_share  9.6's w moves by 0.341 +- 0.03 (MLP 8 is the last writer before block 9: the fold's direct term is exact)
    pred_c_10_1_value_moves_by_its_fold_share  10.1's w moves by 0.273 +- 0.08 (block 9's attention and MLP 9 lie between). Prior: unsure
    pred_d_mlp8_margin_replays              the MLP-8 swap closes 0.078 +- 0.02 of the gap (v552)
    pred_e_readout_is_linear_in_the_readers the margin closed is >= sum_h moved_h x single_h - 0.03 (singles 0.255 / 0.170 / 0.107): the loss is not after the readers
PRICE (registered maximum): 2 batches x 2 passes = 4 forwards (61 pairs); 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_gender_route_census_noun_v549 as gv
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_mlp8_instrumented_v555_result.json"
CANDIDATE_ID = "gender.mlp8_instrumented_v555"
N_HEAD = 9
DETECTORS = (3152, 3943)
UNIT_SETS = {}
SETS = {"MLP8": ((8, 0),)}
VALUES = {}
READERS = ((10, 1), (12, 4), (9, 6)); FOLD = {"10.1": 0.273, "12.4": 0.151, "9.6": 0.341}; SINGLES = {"10.1": 0.255, "12.4": 0.170, "9.6": 0.107}
REPLAY_TOL, BAND96, BAND101, MLP8_V552, BANDM, LIN = 1e-3, 0.03, 0.08, 0.078, 0.02, 0.03
FORWARDS_MAX = 13
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_9_6_value_moves_by_its_fold_share": "0.341 +- 0.03", "pred_c_10_1_value_moves_by_its_fold_share": "0.273 +- 0.08", "pred_d_mlp8_margin_replays": "0.078 +- 0.02", "pred_e_readout_is_linear_in_the_readers": ">= sum moved x single - 0.03"}


def main() -> None:
    text_items = gv.gender_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "values": {k: {str(l): h for l, h in v.items()} for k, v in VALUES.items()}, "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "band96": BAND96, "band101": BAND101, "mlp8_v552": MLP8_V552, "bandm": BANDM, "lin": LIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" he"), L._single(" she")
    MODES = (*UNIT_SETS, *SETS); forwards = 0
    WU = model.lm_head.weight.detach().float(); u = (WU[HE] - WU[SHE]).cuda(); maps = {}
    for l, h in READERS:
        attn = blocks[l].attn; Wp = attn.c_proj.weight.detach().float(); uO = u @ Wp[:, h * hd:(h + 1) * hd]
        Wv = attn.c_v.weight.detach().float()[h * hd:(h + 1) * hd]; maps[(l, h)] = (float(1 - attn.lamb) * (Wv.T @ uO)).cuda()
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, HE] - z[:, SHE]
    def run(seqs, pos, mode):
        """mode: native | a UNIT_SETS key (unit activations swapped at the noun) | a SETS key (whole MLP writes swapped at the noun). Pairs are rows (2i, 2i+1)."""
        nonlocal forwards
        wrec = {}
        tokens = torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                for (rl, rh) in READERS:
                    if rl == l: wrec[f"{rl}.{rh}"] = (xin[idx, pp].float() @ maps[(rl, rh)]).cpu()
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
            return logits_margin(x[idx, fin]).cpu(), wrec
    def study(items, batch):
        seqs, pos = [], []
        for p, s_, c in items: seqs += [list(p), list(s_)]; pos += [c, c]
        res = {m: [] for m in ("native", *MODES)}; W = {m: {f"{l}.{h}": [] for l, h in READERS} for m in res}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res:
                mg, wr = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m); res[m].append(mg)
                for k in W[m]: W[m][k].append(wr[k])
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        W = {m: {k: torch.cat(v) for k, v in d.items()} for m, d in W.items()}
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        moved = {m: {k: closed_w(W["native"][k], W[m][k]) for k in W[m]} for m in MODES}
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "closed": {m: closed(res[m]) for m in MODES}, "w_moved": moved, "w_native_gap_mean": {k: float((v[0::2] - v[1::2]).mean()) for k, v in W["native"].items()}, "native_margins": nat.tolist()}, nat
    def closed_w(nat, e):
        gap = nat[0::2] - nat[1::2]
        return float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
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
    mv = text["w_moved"]["MLP8"]; expected = sum(mv[k] * SINGLES[k] for k in SINGLES); text["expected_linear"] = expected
    print("w moved under MLP8", {k: round(v, 3) for k, v in mv.items()}, "fold", FOLD, "expected linear", round(expected, 3), "margin", round(c["MLP8"], 3))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_9_6_value_moves_by_its_fold_share": abs(mv["9.6"] - FOLD["9.6"]) <= BAND96, "pred_c_10_1_value_moves_by_its_fold_share": abs(mv["10.1"] - FOLD["10.1"]) <= BAND101,
                   "pred_d_mlp8_margin_replays": abs(c["MLP8"] - MLP8_V552) <= BANDM, "pred_e_readout_is_linear_in_the_readers": c["MLP8"] >= expected - LIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_mlp8_instrumented_result_v555", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
