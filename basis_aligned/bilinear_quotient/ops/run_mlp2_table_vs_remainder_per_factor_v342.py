#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_split_closure pred_c_mlp2_table_part_carries_less_than_mlp1s pred_d_mlp2_remainder_carries_at_least_a_third pred_e_signs_agree
"""MLP 2's share of unit 3465's two factors: its own table part vs conditioned remainder (v342). v340: MLP 2 writes 0.37 of 3465's L contrast and 0.17 of R.
v341: MLP 1's part is 81-85% its context-free entry. v318 / v319: MLP 2's own lookup is weaker (gamma^2 0.4) and its write is mostly context-conditioned. Here
MLP 2's write at the noun, W2 = alpha_2 T2 + R2 (T2 = MLP 2's write for the noun token alone through blocks 0-2), and each part's exact term in 3465's two
factor contrasts (lambda-scaled by one factor, block 3). Registered reading: MLP 2's contribution is mostly the conditioned remainder, unlike MLP 1's.
PREDICTIONS (scored as written; failures preserved; priors from v318 / v341)
    pred_a_closure                              the manual forward reproduces the model's they - he margin (2.048) within 1e-3 (instrument)
    pred_b_split_closure                        table part + remainder part = MLP 2's term within relative 1e-3 on every pair and both factors
    pred_c_mlp2_table_part_carries_less_than_mlp1s  MLP 2's table share is below MLP 1's (0.81 / 0.85) on both factors
    pred_d_mlp2_remainder_carries_at_least_a_third  MLP 2's remainder share >= 0.33 on both factors
    pred_e_signs_agree                          the table part's pooled contribution has the sign of MLP 2's total on both factors
PRICE (registered maximum): 3 row batches x 1 forward + 1 table batch = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp2_table_vs_remainder_per_factor_v342_result.json"
CANDIDATE_ID = "pronoun_number.mlp2_table_vs_remainder_per_factor_v342"
UNIT, BATCH = 3465, 32
CLOSURE_TOL, NATIVE_M, SPLIT_TOL, MLP1_TABLE, REM_MIN = 1e-3, 2.0481, 1e-3, {"L": 0.811, "R": 0.848}, 0.33
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_split_closure": "<= 1e-3", "pred_c_mlp2_table_part_carries_less_than_mlp1s": "< 0.81 / 0.85", "pred_d_mlp2_remainder_carries_at_least_a_third": ">= 0.33 x 2", "pred_e_signs_agree": "both factors"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "split_tol": SPLIT_TOL, "mlp1_table": MLP1_TABLE, "rem_min": REM_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; mlp3 = blocks[3].mlp
    Lr, Rr = mlp3.Left.weight.detach().float()[UNIT].cpu(), mlp3.Right.weight.detach().float()[UNIT].cpu()
    W = {"embedding": [], "attn0": [], "mlp0": [], "attn1": [], "mlp1": [], "attn2": [], "mlp2": [], "attn3": []}; X3 = []; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; parts = {k: torch.zeros_like(x) for k in W}
            parts["embedding"] = x.clone()                       # x0 seeds the stream; its lambda-chain re-injections are added below
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                if l <= 3: parts[f"attn{l}"] = parts[f"attn{l}"] + attention
                if l == 3:
                    X3.append(x[idx, pn].float().cpu())
                    for k in W: W[k].append(parts[k][idx, pn].float().cpu())
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,))); x = x + m
                if l < 3: parts[f"mlp{l}"] = parts[f"mlp{l}"] + m
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    X3 = torch.cat(X3); W = {k: torch.cat(v) for k, v in W.items()}; rms = X3.pow(2).mean(1).sqrt()
    noun_ids = sorted({row.ids[noun_of(row)] for row in rows}); tok = torch.tensor(noun_ids, device="cuda").unsqueeze(1)
    with torch.no_grad():                                          # T2: MLP 2's write for the noun alone (blocks 0-2)
        x = F.rms_norm(model.transformer.wte(tok), (model.config.n_embd,)); x0, v1_ = x, None
        for l in (0, 1, 2):
            block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
            m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            if l == 2: T2_tab = m[:, 0].float().cpu()
            x = x + m
    forwards += 1; tindex = {t: i for i, t in enumerate(noun_ids)}; T2 = torch.stack([T2_tab[tindex[row.ids[noun_of(row)]]] for row in rows])
    scale = lam[3][0]                                              # MLP 2's write reaches block 3's input through one lambda0 factor
    W2 = W["mlp2"]; alpha = (W2 * (scale * T2)).sum(1) / ((scale * T2) ** 2).sum(1); A = alpha[:, None] * scale * T2; Rm = W2 - A
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    per, sclos = {}, 0.0
    for name, vec in (("L", Lr), ("R", Rr)):
        term = lambda M: float(((M @ vec)[plural] / rms[plural] - (M @ vec)[sing] / rms[sing]).sum())
        tot, ta, tr = term(W2), term(A), term(Rm); sclos = max(sclos, abs(ta + tr - tot) / max(abs(tot), 1e-6))
        per[name] = {"mlp2_contrast_term": tot, "table_share": ta / tot, "remainder_share": tr / tot, "alpha2_median": float(alpha.median())}
    report = {"closure_margin_gap": closure, "split_closure": sclos, "per_factor": per}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_split_closure": sclos <= SPLIT_TOL, "pred_c_mlp2_table_part_carries_less_than_mlp1s": all(per[n]["table_share"] < MLP1_TABLE[n] for n in ("L", "R")),
                   "pred_d_mlp2_remainder_carries_at_least_a_third": all(per[n]["remainder_share"] >= REM_MIN for n in ("L", "R")), "pred_e_signs_agree": all(per[n]["table_share"] > 0 for n in ("L", "R"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp2_table_vs_remainder_per_factor_result_v342", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
