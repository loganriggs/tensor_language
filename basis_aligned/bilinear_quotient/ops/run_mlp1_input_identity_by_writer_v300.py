#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_input_identity_tracks_alpha pred_c_attention_identity_tracks_self_share pred_d_floor_writer_is_mlp0 pred_e_mlp0_identity_is_its_own_table
"""MLP 1: the token-identity content of its INPUT, by writer (v300). alpha (MLP 1's gain on the token lookup) tracks the attention own-key share
at short context (v291 / v297) and floors ~0.27-0.29 at 16-64 tokens; the floor is not the lambda1 x0 re-injection (v299). MLP 1 reads x1 =
[lambda-chain x0 terms] + attn0 + mlp0 + attn1, so its identity gain must come from the identity content of x1. Per writer w in {x0 terms, attn0,
mlp0, attn1}, beta_w = (w . u) / ||x1_table|| with u the unit single-token x1 direction of the target (the table's input), and beta = sum_w beta_w
= x1's projection on its own table input. Phrase A, lengths 1 / 8 / 64, 7 classes x 32 targets. Also MLP 0's own table: alpha0 = projection of the
in-context MLP-0 write on the single-token MLP-0 write.
PREDICTIONS (scored as written; failures preserved; priors from v291 / v299)
    pred_a_writer_closure                   sum_w beta_w = beta within relative 1e-3 on every row (the writer split of x1 is exact)
    pred_b_input_identity_tracks_alpha      Pearson r(beta, alpha) over all 672 rows >= 0.80 (MLP 1's gain is its input's identity content)
    pred_c_attention_identity_tracks_self_share  the attention writers' identity share (beta_attn0 + beta_attn1) / beta tracks the own-key share: |ratio - s| <= 0.15 median at every length
    pred_d_floor_writer_is_mlp0             at 64 tokens the largest positive beta_w is MLP 0's. Prior: unsure.
    pred_e_mlp0_identity_is_its_own_table   alpha0 median at 64 tokens >= 0.50 (MLP 0's write keeps half its own lookup even at long context)
PRICE (registered maximum): 1 table batch + 3 x (length batch + self-share pass) = 7 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289
import dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_input_identity_by_writer_v300_result.json"
CANDIDATE_ID = "mlp1.token_table.input_identity_by_writer_v300"
PHRASE = (",", " and", " of", " the", " very")
LENGTHS, N, BATCH = (1, 8, 64), 32, 256
CLOSURE_TOL, R_MIN, ATT_TOL, ALPHA0_MIN = 1e-3, 0.80, 0.15, 0.50
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_writer_closure": "<= 1e-3", "pred_b_input_identity_tracks_alpha": "r >= 0.80", "pred_c_attention_identity_tracks_self_share": "<= 0.15 x 3", "pred_d_floor_writer_is_mlp0": "mlp0 largest at 64", "pred_e_mlp0_identity_is_its_own_table": ">= 0.50"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "lengths": LENGTHS, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "r_min": R_MIN, "att_tol": ATT_TOL, "alpha0_min": ALPHA0_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    l00, l01, l10, l11 = (float(blocks[0].lambdas[0]), float(blocks[0].lambdas[1]), float(blocks[1].lambdas[0]), float(blocks[1].lambdas[1]))
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
    T, u = tab["mlp1"], tab["x1"] / tab["x1"].norm(dim=1, keepdim=True); Tn = tab["x1"].norm(dim=1)
    closure, per, allb, alla = 0.0, {}, [], []
    for k in LENGTHS:
        toks = torch.tensor([filler(k) + [t] for t in targets], device="cuda"); pos = torch.full((len(targets),), k, dtype=torch.long, device="cuda")
        c = v289.capture(backend, toks, pos); forwards += 1
        # x1 = l10 (l00 x0 + l01 x0 + attn0 + mlp0) + l11 x0 + attn1   (exact recurrence; v288 closure)
        writers = {"x0": (l10 * (l00 + l01) + l11) * c["x0"], "attn0": l10 * c["attn0"], "mlp0": l10 * c["mlp0"], "attn1": c["attn1"]}
        recon = sum(writers.values()); closure = max(closure, float(((recon - c["x1"]).norm(dim=1) / c["x1"].norm(dim=1)).max()))
        beta_w = {w: (v * u).sum(1) / Tn for w, v in writers.items()}; beta = (c["x1"] * u).sum(1) / Tn
        alpha = (c["mlp1"] * T).sum(1) / (T * T).sum(1); alpha0 = (c["mlp0"] * tab["mlp0"]).sum(1) / (tab["mlp0"] * tab["mlp0"]).sum(1)
        s_k = dod_units.attention_self_share(backend, toks, pos); forwards += 1
        att_ratio = (beta_w["attn0"] + beta_w["attn1"]) / beta
        per[k] = {"alpha_median": float(alpha.median()), "beta_median": float(beta.median()), "beta_by_writer_median": {w: float(v.median()) for w, v in beta_w.items()}, "alpha0_median": float(alpha0.median()), "self_share_median": float(s_k.median()),
                  "attention_identity_ratio_median": float(att_ratio.median()), "att_gap_median": float((att_ratio - s_k).abs().median()), "x1_norm_over_table_median": float((c["x1"].norm(dim=1) / Tn).median())}
        allb.append(beta); alla.append(alpha)
    b_, a_ = torch.cat(allb), torch.cat(alla); r = float(((b_ - b_.mean()) * (a_ - a_.mean())).sum() / ((b_ - b_.mean()).norm() * (a_ - a_.mean()).norm()))
    largest64 = max(per[64]["beta_by_writer_median"], key=per[64]["beta_by_writer_median"].get)
    report = {"closure_max": closure, "pearson_beta_alpha": r, "largest_writer_at_64": largest64, "per_k": {str(k): v for k, v in per.items()}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_writer_closure": closure <= CLOSURE_TOL, "pred_b_input_identity_tracks_alpha": r >= R_MIN, "pred_c_attention_identity_tracks_self_share": all(per[k]["att_gap_median"] <= ATT_TOL for k in LENGTHS),
                   "pred_d_floor_writer_is_mlp0": largest64 == "mlp0", "pred_e_mlp0_identity_is_its_own_table": per[64]["alpha0_median"] >= ALPHA0_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_input_identity_by_writer_result_v300", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
