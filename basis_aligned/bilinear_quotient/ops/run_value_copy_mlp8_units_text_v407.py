#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_829_first_on_text pred_c_953_in_top_3 pred_d_top_50_carry_half pred_e_gender_units_outside_top_20
"""MLP 8 at unit grain on natural text (v407). v406: MLP 8 writes half of the number the readers copy from the noun, on the 122 natural swapped pairs as
on the panel. v168 (panel): that write is carried by units 829 / 953 first. Here the exact per-unit split of MLP 8's share of delta w for 9.6 on the
NATURAL pairs: T_j = (m . Down[:, j]) h_j / rms(live_9) per row, differenced over each pair (m = 9.6's value-branch reader direction, v406).
PREDICTIONS (scored as written; failures preserved; priors from v168 / v399)
    pred_a_unit_closure               per-unit terms sum to MLP 8's writer term within relative 1e-3 on every pair
    pred_b_829_first_on_text          unit 829 has the largest |pooled contrast| of the 4608
    pred_c_953_in_top_3               unit 953 is in the top 3
    pred_d_top_50_carry_half          the top 50 units carry >= 0.50 of MLP 8's pooled contrast
    pred_e_gender_units_outside_top_20  units 3152 and 3943 (gender, v164) are both outside the top 20. Prior: likely (v168: 85th / far)
PRICE (registered maximum): 4 text batches = 4 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/value_copy_mlp8_units_text_v407_result.json"
CANDIDATE_ID = "both_ends.value_copy_mlp8_units_text_v407"
N_HEAD, LAYER, READER = 9, 8, (9, 6)
CLOSURE_TOL, TOP50_MIN = 1e-3, 0.50
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_829_first_on_text": "rank 1", "pred_c_953_in_top_3": "rank <= 3", "pred_d_top_50_carry_half": ">= 0.50", "pred_e_gender_units_outside_top_20": "rank > 20 x 2"}


def main() -> None:
    items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(items), "layer": LAYER, "reader": "9.6", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top50_min": TOP50_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    attn = blocks[READER[0]].attn; Wp = attn.c_proj.weight.detach().float(); uO = u @ Wp[:, READER[1] * hd:(READER[1] + 1) * hd]
    m = float(1 - attn.lamb) * (attn.c_v.weight.detach().float()[READER[1] * hd:(READER[1] + 1) * hd].T @ uO)   # (D,)
    mlp = blocks[LAYER].mlp; Dw = mlp.Down.weight.detach().float(); mD = m @ Dw                                     # (4608,)
    seqs = [p for p, _, _ in items] + [s for _, s, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
    per_row_units, per_row_mlp, forwards = [], [], 0
    with torch.no_grad():
        for s0 in range(0, len(seqs), 64):
            tokens = torch.tensor(seqs[s0:s0 + 64], device="cuda"); pp = torch.tensor(pos[s0:s0 + 64]); idx = torch.arange(tokens.shape[0])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; lam_after = 1.0
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                if l == READER[0]:
                    rms = live[idx, pp].float().pow(2).mean(1).sqrt(); break
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; xin = F.rms_norm(x, (D,))
                if l == LAYER:
                    Lx, Rx = mlp.Left(xin), mlp.Right(xin); h = (F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx); h8 = h[idx, pp].float(); m8 = mlp(xin)[idx, pp].float()
                x = x + block.mlp(xin)
            # the lambda chain from block 8's output to block 9's live: block 9's lambdas[0] (blocks 9 only, since MLP 8 is added after block 8's live)
            lam = float(blocks[READER[0]].lambdas[0])
            per_row_units.append((lam * mD.cpu() * h8.cpu()) / rms.cpu()[:, None]); per_row_mlp.append((lam * (m8 @ m).cpu() + lam * float(m @ mlp.Down_bias.detach().float())) / rms.cpu())
            forwards += 1
    U = torch.cat(per_row_units); M = torch.cat(per_row_mlp)
    closure = float(((U.sum(1) - M).abs() / M.abs().clamp_min(1e-6)).max())
    total = (U[:n] - U[n:]).sum(0); contrast = float(total.sum()); order = torch.argsort(total.abs(), descending=True)
    rank = {j: int((total.abs() > abs(total[j])).sum()) + 1 for j in (829, 953, 1030, 3152, 3943)}
    share = lambda k: float(total[order[:k]].sum()) / contrast
    report = {"closure_max": closure, "mlp8_contrast": contrast, "shares": {str(k): share(k) for k in (10, 20, 50, 100, 200)}, "top_units": [(int(j), float(total[j])) for j in order[:30]], "ranks": rank}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_829_first_on_text": rank[829] == 1, "pred_c_953_in_top_3": rank[953] <= 3, "pred_d_top_50_carry_half": share(50) >= TOP50_MIN, "pred_e_gender_units_outside_top_20": rank[3152] > 20 and rank[3943] > 20}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "value_copy_mlp8_units_text_result_v407", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
