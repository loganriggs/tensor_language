#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_remainder_is_high_rank pred_c_no_single_shared_direction pred_d_class_structure_survives pred_e_remainder_is_cross_term
"""MLP 1: what the cross remainder is (v293). v292 falsified the two-term reading: after removing alpha x table(token), MLP 1's write in context
is orthogonal to the context's own table write (cos ~ 0 for 1-64 filler tokens) and to the token's; by v289 the remainder is the token x context
bilinear cross term Down[(L t)(R c) + (L c)(R t)]. This receipt characterises it at context length 1 and 8 over 7 classes x 32 targets:
(i) rank: r90 (dimensions for 90% of the energy) of the 224 remainders vs r90 of the 224 table entries (v287: ~0.7n); (ii) a shared direction:
the energy share of the grand-mean remainder; (iii) class structure: mean cosine of each remainder with its class-mean remainder (leave-one-out)
vs with other classes' means; (iv) the cross vs context-only split of the remainder's energy (exact terms).
PREDICTIONS (scored as written; failures preserved; priors from v287/v289)
    pred_a_pair_closure               Down[cross + context-only] = W - T within relative 1e-3 on every row
    pred_b_remainder_is_high_rank     r90(remainder) >= 0.5 x r90(table) at both lengths (token-specific, not a few context directions)
    pred_c_no_single_shared_direction the grand-mean remainder carries <= 0.30 of the remainders' energy at both lengths
    pred_d_class_structure_survives   leave-one-out class-mean cosine exceeds the other-class mean cosine by >= 0.15 at both lengths
    pred_e_remainder_is_cross_term    the cross term carries >= 0.70 of the remainder's norm (median) at both lengths
PRICE (registered maximum): 2 lengths x 224 rows = 2 forwards + 224 tokens = 1 forward = 3; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_cross_remainder_v293_result.json"
CANDIDATE_ID = "mlp1.token_table.cross_remainder_v293"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8), 32, 256
CLOSURE_TOL, RANK_RATIO, MEAN_MAX, CLASS_GAP, CROSS_MIN = 1e-3, 0.5, 0.30, 0.15, 0.70
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_remainder_is_high_rank": ">= 0.5 r90(table) x 2", "pred_c_no_single_shared_direction": "<= 0.30 x 2", "pred_d_class_structure_survives": "gap >= 0.15 x 2", "pred_e_remainder_is_cross_term": ">= 0.70 x 2"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "rank_ratio": RANK_RATIO, "mean_max": MEAN_MAX, "class_gap": CLASS_GAP, "cross_min": CROSS_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(targets)}
    T = tab["mlp1"]; n_tab = F.rms_norm(tab["x1"], (T.shape[-1],)); Lt, Rt = n_tab @ Lw.T, n_tab @ Rw.T
    def r90(M):
        sv = torch.linalg.svdvals(M - 0) ** 2; cum = sv.cumsum(0) / sv.sum(); return int((cum < 0.90).sum()) + 1
    def class_stats(R):
        cidx = {name: torch.tensor([tindex[t] for t in v]) for name, v in cls.items()}; means = {name: R[i].mean(0) for name, i in cidx.items()}
        own, other = [], []
        for name, i in cidx.items():
            for j in i.tolist():
                loo = (means[name] * len(i) - R[j]) / (len(i) - 1); own.append(float(torch.nn.functional.cosine_similarity(R[j], loo, dim=0)))
                other.append(float(torch.stack([torch.nn.functional.cosine_similarity(R[j], m, dim=0) for n2, m in means.items() if n2 != name]).mean()))
        return sum(own) / len(own), sum(other) / len(other)
    closure, per_k, rank_T = 0.0, {}, r90(T)
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); c = v289.capture(backend, toks, torch.full((len(targets),), k, dtype=torch.long, device="cuda")); forwards += 1
        W = c["mlp1"]; cc = F.rms_norm(c["x1"], (T.shape[-1],)) - n_tab; Lc, Rc = cc @ Lw.T, cc @ Rw.T
        cross = (Lt * Rc + Lc * Rt) @ Dw.T; only = (Lc * Rc) @ Dw.T; change = W - T
        closure = max(closure, float((((cross + only) - change).norm(dim=1) / change.norm(dim=1).clamp_min(1e-6)).max()))
        alpha = (W * T).sum(1) / (T * T).sum(1); R = W - alpha[:, None] * T
        proj = lambda A: A - ((A * T).sum(1) / (T * T).sum(1))[:, None] * T          # the same table-direction removal applied to each exact term
        Rc_, Ro_ = proj(cross), proj(only)
        mean_share = float((R.mean(0).norm() ** 2 * len(R)) / (R * R).sum()); own, other = class_stats(R)
        per_k[k] = {"alpha_median": float(alpha.median()), "r90_remainder": r90(R), "r90_table": rank_T, "r90_write": r90(W), "grand_mean_share": mean_share, "class_own_cos": own, "class_other_cos": other, "class_gap": own - other,
                    "cross_share_median": float((Rc_.norm(dim=1) / (Rc_.norm(dim=1) + Ro_.norm(dim=1))).median()), "remainder_energy_share": float(((R * R).sum(1) / (W * W).sum(1)).median()),
                    "remainder_norm_median": float(R.norm(dim=1).median()), "class_means_cos_matrix": [[float(torch.nn.functional.cosine_similarity(R[torch.tensor([tindex[t] for t in a_])].mean(0), R[torch.tensor([tindex[t] for t in b_])].mean(0), dim=0)) for b_ in cls.values()] for a_ in cls.values()]}
    report = {"closure_max": closure, "classes": list(cls), "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps({str(k): {a_: b_ for a_, b_ in v.items() if a_ != "class_means_cos_matrix"} for k, v in per_k.items()}, indent=1, default=float))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_remainder_is_high_rank": all(v["r90_remainder"] >= RANK_RATIO * v["r90_table"] for v in per_k.values()), "pred_c_no_single_shared_direction": all(v["grand_mean_share"] <= MEAN_MAX for v in per_k.values()),
                   "pred_d_class_structure_survives": all(v["class_gap"] >= CLASS_GAP for v in per_k.values()), "pred_e_remainder_is_cross_term": all(v["cross_share_median"] >= CROSS_MIN for v in per_k.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_cross_remainder_result_v293", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
