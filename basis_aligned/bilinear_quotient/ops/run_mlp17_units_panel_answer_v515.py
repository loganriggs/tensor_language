#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_top_10_carry_half pred_c_detector_units_outside_top_10 pred_d_pooled_contrast_negative pred_e_top_unit_is_a_number_unit
"""MLP 17 at the PANEL's answer position, by unit (v515). v512: on the panel MLP 17's write at the answer opposes the pronoun margin (-0.115); v514: not the
detector units 701 / 2059. MLP 17's write at the panel answer split exactly by unit on u = W_U[they] - W_U[he] (the final rms shared per row), plural -
singular over the 48 v76 pairs: which units carry the opposing write, and is it a few or a spread?
PREDICTIONS (scored as written; failures preserved; priors from v512 / v514)
    pred_a_unit_closure               per-unit terms sum to MLP 17's term within relative 1e-3 on every row
    pred_b_top_10_carry_half          the top 10 units carry >= 0.50 of MLP 17's pooled contrast on u at the panel answer. Prior: unsure
    pred_c_detector_units_outside_top_10  701 and 2059 are both outside the top 10 by |pooled contrast|
    pred_d_pooled_contrast_negative   MLP 17's pooled plural - singular contrast on u at the panel answer is negative (it opposes, as the edit showed)
    pred_e_top_unit_is_a_number_unit  the top unit's plural - singular activation contrast has one sign on >= 0.80 of the pairs
PRICE (registered maximum): 3 panel batches = 3 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp17_units_panel_answer_v515_result.json"
CANDIDATE_ID = "chain.mlp17_units_panel_answer_v515"
N_HEAD, LAYER = 9, 17
CLOSURE_TOL, TOP10_MIN, SIGN_MIN = 1e-3, 0.50, 0.80
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_top_10_carry_half": ">= 0.50", "pred_c_detector_units_outside_top_10": "rank > 10 x 2", "pred_d_pooled_contrast_negative": "< 0", "pred_e_top_unit_is_a_number_unit": ">= 0.80 one sign"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    items = [(list(rows[i].ids), list(rows[partner_[(rows[i].construction, rows[i].group, False)]].ids), len(rows[i].ids) - 1) for i, r_ in enumerate(rows) if r_.present]   # the panel pairs at their answer position
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(items), "layer": LAYER, "reader": "direct (W_U), panel answer position", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "sign_min": SIGN_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    m = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()   # the pronoun direction on the final residual at the answer position
    mlp = blocks[LAYER].mlp; Dw = mlp.Down.weight.detach().float(); mD = m @ Dw                                     # (4608,)
    seqs = [p for p, _, _ in items] + [s for _, s, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
    per_row_units, per_row_mlp, per_row_h, forwards = [], [], [], 0
    with torch.no_grad():
        for s0 in range(0, len(seqs), 64):
            tokens = torch.tensor([q + [0] * (max(len(t) for t in seqs[s0:s0 + 64]) - len(q)) for q in seqs[s0:s0 + 64]], device="cuda"); pp = torch.tensor(pos[s0:s0 + 64]); idx = torch.arange(tokens.shape[0])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; lam_after = 1.0
            for l, block in enumerate(list(blocks) + [None]):
                if block is None: rms = x[idx, pp].float().pow(2).mean(1).sqrt(); break
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                if l == 18: break
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; xin = F.rms_norm(x, (D,))
                if l == LAYER:
                    Lx, Rx = mlp.Left(xin), mlp.Right(xin); h = (F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx); h8 = h[idx, pp].float(); m8 = mlp(xin)[idx, pp].float()
                x = x + block.mlp(xin)
            # the lambda chain from block 8's output to block 9's live: block 9's lambdas[0] (blocks 9 only, since MLP 8 is added after block 8's live)
            lam = 1.0
            for l_ in range(LAYER + 1, 18): lam *= float(blocks[l_].lambdas[0])          # the lambda chain from block 8 output to the final residual
            per_row_units.append((lam * mD.cpu() * h8.cpu()) / rms.cpu()[:, None]); per_row_h.append(h8.cpu()); per_row_mlp.append((lam * (m8 @ m).cpu() - lam * float(m @ mlp.Down_bias.detach().float())) / rms.cpu())   # module output minus its bias = sum of unit terms
            forwards += 1
    U = torch.cat(per_row_units); M = torch.cat(per_row_mlp)
    closure = float(((U.sum(1) - M).abs() / M.abs().clamp_min(1e-6)).max())
    total = (U[:n] - U[n:]).sum(0); contrast = float(total.sum()); order = torch.argsort(total.abs(), descending=True)
    rank = {j: int((total.abs() > abs(total[j])).sum()) + 1 for j in (829, 953, 1030, 3152, 3943)}
    prev = json.loads((ROOT / "circuits/followups/value_copy_mlp8_units_text_v408_result.json").read_text())["report"]["top_units"][:20]; overlap = len({int(j) for j, _ in prev} & set(order[:20].tolist()))
    share = lambda k: float(total[order[:k]].sum()) / contrast
    report = {"closure_max": closure, "mlp8_contrast": contrast, "shares": {str(k): share(k) for k in (10, 20, 50, 100, 200)}, "top_units": [(int(j), float(total[j])) for j in order[:30]], "ranks": rank, "top20_overlap_with_v408": overlap}
    print(json.dumps(report, indent=1))
    top = int(order[0]); H = torch.cat(per_row_h); dtop = H[:n, top] - H[n:, top]; sign_const = float(max((dtop > 0).float().mean(), (dtop < 0).float().mean()))
    rank_det = {j: int((total.abs() > abs(total[j])).sum()) + 1 for j in (701, 2059)}
    report["top_unit"] = top; report["top_unit_sign_constancy"] = sign_const; report["detector_ranks"] = rank_det
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_top_10_carry_half": share(10) >= TOP10_MIN, "pred_c_detector_units_outside_top_10": min(rank_det.values()) > 10, "pred_d_pooled_contrast_negative": contrast < 0, "pred_e_top_unit_is_a_number_unit": sign_const >= SIGN_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp17_units_panel_answer_result_v515", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
