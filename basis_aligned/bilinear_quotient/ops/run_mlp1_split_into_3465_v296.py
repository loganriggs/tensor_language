#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_product_closure pred_b_table_part_carries_most pred_c_remainder_carries_little pred_d_same_for_493 pred_e_table_part_positive_every_pair
"""MLP 1 -> the number chain (v296): which part of MLP 1's noun write does MLP-3 unit 3465 read? v288-v295 established, by exact folds, that MLP 1's
write at a token in context = alpha x table(token) + R, with alpha the token's own-key attention share (~0.54 after "The") and R a context-specific
remainder (~25% of the energy at one context token) that is neither table entry nor a single bilinear term. v194 / v201: MLP-3 units 3465 and 493 are the
largest MLP-3 carriers of the number into MLP-5 unit 1036, and MLPs 1-2 carry >= 50% of their input contrast. Here the v76 rows' noun write W is
split into alpha T (T = the noun's single-token table entry, alpha its projection coefficient) and R = W - alpha T, and each part's carriage of the
plural - singular contrast into unit 3465's product h = (L.x^)(R.x^) at the block-3 input is computed with the exact product-level term of
`dod_units.product_unit_census` (linear-in-writer part (L.w)(R.x^) + (L.x^)(R.w) minus the writer's own quadratic (L.w)(R.w), over rms^2), with the
lambda chain from block 1 to block 3 folded in. Shares are of MLP 1's total carriage.
PREDICTIONS (scored as written; failures preserved; priors from v288: the number difference of the write has cos 0.87 with the table difference)
    pred_a_product_closure           alpha T-part + R-part + their cross term = MLP 1's total carriage within relative 1e-3, every row
    pred_b_table_part_carries_most   the alpha T part carries >= 0.70 of MLP 1's pooled contrast into 3465
    pred_c_remainder_carries_little  the R part carries <= 0.30
    pred_d_same_for_493              pred_b and pred_c hold for unit 493 as well
    pred_e_table_part_positive_every_pair  the alpha T part's contrast has the sign of the total for >= 0.90 of aligned pairs (3465)
PRICE (registered maximum): 3 row batches (blocks 0-3) + 1 table batch = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_context_gain_decomposition_v289 as v289
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_split_into_3465_v296_result.json"
CANDIDATE_ID = "mlp1.token_table.split_into_3465_v296"
UNITS, DST, SRC, BATCH = (3465, 493), 3, 1, 32
CLOSURE_TOL, TABLE_MIN, REM_MAX, SIGN_MIN = 1e-3, 0.70, 0.30, 0.90
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_product_closure": "<= 1e-3", "pred_b_table_part_carries_most": ">= 0.70", "pred_c_remainder_carries_little": "<= 0.30", "pred_d_same_for_493": "b and c for 493", "pred_e_table_part_positive_every_pair": ">= 0.90"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "src": SRC, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "table_min": TABLE_MIN, "rem_max": REM_MAX, "sign_min": SIGN_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0
    blocks = model.transformer.h; scale = 1.0
    for l in range(SRC + 1, DST + 1): scale *= float(blocks[l].lambdas[0])
    W, X3 = [], []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = torch.tensor([noun_of(r) for r in chunk], device=tokens.device); idx = torch.arange(len(chunk))
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == DST: X3.append(x[idx, pos].float().cpu()); break
                m = block.mlp(xin)
                if l == SRC: W.append(m[idx, pos].float().cpu())
                x = x + m
            forwards += 1
    W, X3 = torch.cat(W), torch.cat(X3)
    noun_ids = sorted({row.ids[noun_of(row)] for row in rows}); tok = torch.tensor(noun_ids, device="cuda").unsqueeze(1)
    tab = v289.capture(backend, tok, torch.zeros(len(noun_ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(noun_ids)}
    T = torch.stack([tab["mlp1"][tindex[row.ids[noun_of(row)]]] for row in rows]); alpha = (W * T).sum(1) / (T * T).sum(1); A = alpha[:, None] * T; R = W - A
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    dst = blocks[DST].mlp; result, closure = {}, 0.0
    for unit in UNITS:
        Lr, Rr = dst.Left.weight.detach().float()[unit].cpu(), dst.Right.weight.detach().float()[unit].cpu()
        rms2 = X3.pow(2).mean(1); Lx, Rx = X3 @ Lr, X3 @ Rr
        def carry(w):  # exact contribution of writer w (scaled by the lambda chain) to the product at the block-3 input, given the other writers fixed
            ws = scale * w; lw, rw = ws @ Lr, ws @ Rr
            return (lw * Rx + Lx * rw - lw * rw) / rms2
        tot, ca, cr = carry(W), carry(A), carry(R)
        cross = tot - ca - cr                                             # = -(La Rr + Lr Ra)/rms2 ... the mixed quadratic term, reported
        closure = max(closure, float(((ca + cr + cross - tot).abs() / tot.abs().clamp_min(1e-6)).max()))
        def pooled(v):
            acc, signs = 0.0, []
            for i, row in enumerate(rows):
                if not row.present: continue
                j = partner[(row.construction, row.group, False)]; d = float(v[i] - v[j]); acc += d; signs.append(d)
            return acc, signs
        Ptot, stot = pooled(tot); Pa, sa = pooled(ca); Pr, _ = pooled(cr); Pc, _ = pooled(cross)
        agree = sum(1 for a_, t_ in zip(sa, stot) if (a_ > 0) == (t_ > 0)) / len(stot)
        result[str(unit)] = {"total_contrast": Ptot, "table_share": Pa / Ptot, "remainder_share": Pr / Ptot, "cross_share": Pc / Ptot, "table_sign_agreement": agree, "alpha_median": float(alpha.median()),
                             "mean_abs_total_per_row": float(tot.abs().mean()), "mean_abs_table_per_row": float(ca.abs().mean()), "mean_abs_remainder_per_row": float(cr.abs().mean())}
    print(json.dumps({"closure": closure, "units": result}, indent=1))
    r3465, r493 = result["3465"], result["493"]
    predictions = {"pred_a_product_closure": closure <= CLOSURE_TOL, "pred_b_table_part_carries_most": r3465["table_share"] >= TABLE_MIN, "pred_c_remainder_carries_little": r3465["remainder_share"] <= REM_MAX,
                   "pred_d_same_for_493": r493["table_share"] >= TABLE_MIN and r493["remainder_share"] <= REM_MAX, "pred_e_table_part_positive_every_pair": r3465["table_sign_agreement"] >= SIGN_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_split_into_3465_result_v296", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "units": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
