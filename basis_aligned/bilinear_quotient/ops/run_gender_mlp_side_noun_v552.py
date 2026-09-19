#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_detectors_by_replacement pred_c_mlp8_whole_above_the_detectors pred_d_mlps_1_7_carry pred_e_random_units_inert
"""The gender line's MLP side at the noun (v552). v549-v551: gender leaves the noun through values (0.901 of the he - she gap on 61 natural pairs), read by
10.1 / 12.4 / 9.6 / 15.1 (0.60) and planted back into the noun by the self-copies 8.1 / 6.1 (0.31). The 18 Sep folds named two MLP-8 units at the noun --
3152 (male-noun detector) and 3943 (female) -- and ZEROING them removed 2.6% of the margin. Here the number line's replace-edits at the noun instead: the
two detectors' activations swapped within the pair (U2), two random MLP-8 units (RAND), MLP 8's whole write (MLP8), MLPs 1-7 whole (MLPS_1_7), MLPs 9-17
whole (MLPS_9_17) -- fraction of the male - female he - she gap closed.
PREDICTIONS (scored as written; failures preserved; priors from the 18 Sep folds (carrier split v189) and the number line (v442, v542))
    pred_a_baseline_replays            the unedited manual forward replays the model's he - she margins within 1e-3
    pred_b_detectors_by_replacement    3152 + 3943 swapped at the noun close >= 0.05 of the gap (zeroing: 0.026)
    pred_c_mlp8_whole_above_the_detectors  MLP 8's whole write at the noun closes more than the two units, and >= 0.10
    pred_d_mlps_1_7_carry              MLPs 1-7 at the noun close >= 0.20 (the gender state is built below MLP 8 too: the number line's MLPs 5-7 were half of MLP 8's read)
    pred_e_random_units_inert          two random MLP-8 units close <= 0.01
PRICE (registered maximum): 2 batches x 6 passes = 12 forwards (61 pairs); 0 backwards; 0 fits. Bar <= 13.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_gender_route_census_noun_v549 as gv
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_mlp_side_noun_v552_result.json"
CANDIDATE_ID = "gender.mlp_side_noun_v552"
N_HEAD = 9
DETECTORS = (3152, 3943)
UNIT_SETS = {"U2": (8, DETECTORS), "RAND": (8, tuple(random.Random(0).sample([j for j in range(4608) if j not in DETECTORS], 2)))}
SETS = {"MLP8": ((8, 0),), "MLPS_1_7": tuple((l, 0) for l in range(1, 8)), "MLPS_9_17": tuple((l, 0) for l in range(9, 18))}
REPLAY_TOL, U2_MIN, MLP8_MIN, LOW_MIN, RAND_MAX = 1e-3, 0.05, 0.10, 0.20, 0.01
FORWARDS_MAX = 13
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_detectors_by_replacement": ">= 0.05", "pred_c_mlp8_whole_above_the_detectors": "> U2 and >= 0.10", "pred_d_mlps_1_7_carry": ">= 0.20", "pred_e_random_units_inert": "<= 0.01"}


def main() -> None:
    text_items = gv.gender_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "unit_sets": {k: [v[0], list(v[1])] for k, v in UNIT_SETS.items()}, "sets": {k: list(v) for k, v in SETS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "u2_min": U2_MIN, "mlp8_min": MLP8_MIN, "low_min": LOW_MIN, "rand_max": RAND_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; HE, SHE = L._single(" he"), L._single(" she")
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
                attention, v1_ = attn(xin, v1_); x = live + attention; xm = F.rms_norm(x, (D,))
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_detectors_by_replacement": c["U2"] >= U2_MIN, "pred_c_mlp8_whole_above_the_detectors": c["MLP8"] > c["U2"] and c["MLP8"] >= MLP8_MIN,
                   "pred_d_mlps_1_7_carry": c["MLPS_1_7"] >= LOW_MIN, "pred_e_random_units_inert": abs(c["RAND"]) <= RAND_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_mlp_side_noun_result_v552", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
