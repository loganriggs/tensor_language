#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_mlp2_direction_kept pred_c_mlp2_gain_below_one pred_d_mlp2_cross_negative pred_e_mlp2_gain_tracks_mlp1_gain
"""MLP 2: does the next bilinear layer obey the same law? (v318). MLP 1 on text (v297-v302): write = alpha x table(token) + remainder, alpha ~ 0.26,
alpha tracks the attention self-share, and the loss of gain is the token x context cross term projecting negatively on the lookup (gamma^2 0.85,
cross -0.87, context^2 +0.29). v317: MLP 2 is the principal reader of MLP 1's context-conditioned write. Same instruments on MLP 2 at the 2,944
natural positions: alpha_2 = projection of MLP 2's in-context write on its own single-token table entry (block-2 output for the token alone), cosine,
and the exact three-term expansion of MLP 2's write in its normalised input split into the single-token direction and the rest.
PREDICTIONS (scored as written; failures preserved; priors unsure -- MLP 2 reads a context-conditioned input, so its lookup may already be weak)
    pred_a_closure                  the block-2 recurrence reproduces the captured MLP-2 write within relative 1e-4 (instrument)
    pred_b_mlp2_direction_kept      median cosine(MLP-2 write, its table entry) >= 0.40
    pred_c_mlp2_gain_below_one      median alpha_2 <= 0.60 (context reduces MLP 2's lookup as it does MLP 1's)
    pred_d_mlp2_cross_negative      the token x context cross term projects negatively on MLP 2's table entry at >= 0.80 of positions
    pred_e_mlp2_gain_tracks_mlp1_gain  Pearson r(alpha_2, alpha_1) over positions >= 0.40
PRICE (registered maximum): 2 natural batches + <= 12 unique-token table batches = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp2_lookup_law_natural_v318_result.json"
CANDIDATE_ID = "mlp2.token_table.lookup_law_natural_v318"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, COS_MIN, ALPHA_MAX, NEG_MIN, R_MIN, BATCH = 1e-4, 0.40, 0.60, 0.80, 0.40, 256
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_mlp2_direction_kept": ">= 0.40", "pred_c_mlp2_gain_below_one": "<= 0.60", "pred_d_mlp2_cross_negative": ">= 0.80", "pred_e_mlp2_gain_tracks_mlp1_gain": "r >= 0.40"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "natural": [p.name for p in NATURAL], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cos_min": COS_MIN, "alpha_max": ALPHA_MAX, "neg_min": NEG_MIN, "r_min": R_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); Tlen = nat.shape[1]
    writes, x1s, toks, w2s, x2s = [], [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1, 2):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l == 1: x1, m1 = x, m
                if l == 2: x2, m2 = x, m
                x = x + m
            forwards += 1
            writes.append(m1[:, 1:].reshape(-1, m1.shape[-1]).float().cpu()); x1s.append(x1[:, 1:].reshape(-1, m1.shape[-1]).float().cpu()); toks.append(chunk[:, 1:].reshape(-1).cpu())
            w2s.append(m2[:, 1:].reshape(-1, m2.shape[-1]).float().cpu()); x2s.append(x2[:, 1:].reshape(-1, m2.shape[-1]).float().cpu())
    W, X1, tok, W2, X2 = torch.cat(writes), torch.cat(x1s), torch.cat(toks), torch.cat(w2s), torch.cat(x2s)
    uniq = sorted(set(tok.tolist())); tab = {"mlp1": [], "x1": [], "mlp2": [], "x2": []}
    with torch.no_grad():
        for s0 in range(0, len(uniq), BATCH):
            ids = torch.tensor(uniq[s0:s0 + BATCH], device="cuda").unsqueeze(1); x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1, 2):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l == 1: tab["x1"].append(x[:, 0].float().cpu()); tab["mlp1"].append(m[:, 0].float().cpu())
                if l == 2: tab["x2"].append(x[:, 0].float().cpu()); tab["mlp2"].append(m[:, 0].float().cpu())
                x = x + m
            forwards += 1
    tab = {k: torch.cat(v) for k, v in tab.items()}; tindex = {t: i for i, t in enumerate(uniq)}; ti = torch.tensor([tindex[t] for t in tok.tolist()])
    T1 = tab["mlp1"][ti]; alpha1 = (W * T1).sum(1) / (T1 * T1).sum(1)
    mlp2 = model.transformer.h[2].mlp; Lw, Rw, Dw = mlp2.Left.weight.detach().float().cpu(), mlp2.Right.weight.detach().float().cpu(), mlp2.Down.weight.detach().float().cpu()
    rec = mlp2(F.rms_norm(X2.to("cuda"), (X2.shape[-1],))).float().cpu(); closure = float(((rec - W2).norm(dim=1) / W2.norm(dim=1)).max())
    T2 = tab["mlp2"][ti]; alpha2 = (W2 * T2).sum(1) / (T2 * T2).sum(1); cos2 = (W2 * T2).sum(1) / (W2.norm(dim=1) * T2.norm(dim=1))
    n = F.rms_norm(X2, (X2.shape[-1],)); n_tab = F.rms_norm(tab["x2"][ti], (X2.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
    Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
    quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T; only = (Lc * Rc) @ Dw.T
    pT = lambda A: (A * T2).sum(1) / (T2 * T2).sum(1); g2, pc, po = pT(quad), pT(cross), pT(only)
    def pearson(a, b): a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
    report = {"positions": int(W2.shape[0]), "closure_max": closure, "alpha1_median": float(alpha1.median()), "alpha2_median": float(alpha2.median()), "cos2_median": float(cos2.median()), "gamma2_median": float(g2.median()), "cross_proj_median": float(pc.median()),
              "cross_negative_fraction": float((pc < 0).float().mean()), "only_proj_median": float(po.median()), "pearson_alpha2_alpha1": pearson(alpha2, alpha1), "decomposition_max_rel_err": float((((quad + cross + only) - W2).norm(dim=1) / W2.norm(dim=1)).max()),
              "write2_over_table2_norm_median": float((W2.norm(dim=1) / T2.norm(dim=1)).median()), "by_position_alpha2": [float(alpha2.view(-1, Tlen - 1)[:, p_].median()) for p_ in range(Tlen - 1)]}
    print(json.dumps({k: v for k, v in report.items() if k != "by_position_alpha2"}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_mlp2_direction_kept": report["cos2_median"] >= COS_MIN, "pred_c_mlp2_gain_below_one": report["alpha2_median"] <= ALPHA_MAX, "pred_d_mlp2_cross_negative": report["cross_negative_fraction"] >= NEG_MIN, "pred_e_mlp2_gain_tracks_mlp1_gain": report["pearson_alpha2_alpha1"] >= R_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_lookup_law_natural_result_v318", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
