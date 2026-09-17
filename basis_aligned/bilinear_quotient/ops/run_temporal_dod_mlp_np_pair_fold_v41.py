#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_bilinear_closure pred_b_top_pair_involves_head_8_1 pred_c_pairs_with_8_1_carry_half
"""Temporal will/had DoD battery, step 14 (v41): FOLD MLP8/9/10's NP-state writes into writer-pair terms.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v35 (mlp9 0.21, mlp10 0.16, mlp8 0.16 of the NP
contrast on 11.3's reader direction), v37/v39 (attention's part is head 8.1 reading the cue, token-only).

WHY. Last open port on the NP state. Each MLP_L output at an NP position is bilinear in its input x_L = live_L +
attn_L (attention L split by head), and that input is an exact lambda-weighted writer sum, so r . mlp_L(pos) expands
exactly into writer-pair terms (as the aspectual v16). Pooled over the three MLPs and both NP positions, oriented
tomorrow - earlier. If 8.1's write is a factor in most of the mass, the whole NP state is generated from the adverb
token; if diffuse, the kill criterion declares the port.

PREDICTIONS (scored as written; failures preserved)
    pred_a_bilinear_closure            pair sum + bias equals r . mlp_L(pos) within relative 1e-3 everywhere
    pred_b_top_pair_involves_head_8_1  the largest |symmetrized pair share| (pooled) has attnhead:08:1 as a factor
    pred_c_pairs_with_8_1_carry_half   pairs with attnhead:08:1 as a factor carry >= 0.50 of the pooled contrast

PRICE (registered maximum): 2 batches x 1 trace forward = 2 forwards; all pair terms on captured vectors; 0 backwards;
0 fits. Bar <= 6.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28
import run_temporal_dod_np_writer_fold_v35 as v35

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_mlp_np_pair_fold_v41_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_mlp_np_pair_fold_v41"
CLOSURE_TOL, HALF = 1e-3, 0.50
FORWARDS_MAX = 6
COMP = L.Component("attn11_h3_final", 11, "attn", (3,), "final")
MLPS = (8, 9, 10)


def writers_at_mlp_input(tr, pos, layer):
    """Exact writer decomposition of x_layer[pos] = live_layer + attn_layer, with attention `layer` split by head."""
    x0 = tr[("embed", pos)]
    C = {"embed": x0.clone()}
    for l in range(layer):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C:
            C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0
        C[f"attn:{l:02d}"] = tr[(f"attn:{l:02d}", pos)].clone(); C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    l0, l1 = tr[f"lambda0:{layer:02d}"], tr[f"lambda1:{layer:02d}"]
    for k in C:
        C[k] = l0 * C[k]
    C["embed"] = C["embed"] + l1 * x0
    for h in range(9):
        C[f"attnhead:{layer:02d}:{h}"] = tr[(f"attnhead:{layer:02d}:{h}", pos)].clone()
    return C


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMP,), v28.WILL, v28.HAD)
    r = L.reader_directions(model, COMP, fw.directions)[3]
    forwards = 0
    traces = []
    for start in range(0, len(rows), v1.BATCH):
        traces.extend(L.forward_trace_positions(fw, rows[start:start + v1.BATCH], v35.np_positions, upto_layer=11, head_write_layers=MLPS)); forwards += 1
    closure = 0.0
    tables = []   # per row: {(layer,pos): (writers, T)}
    for row, tr in zip(rows, traces):
        entry = {}
        for layer in MLPS:
            mlp = model.transformer.h[layer].mlp
            Lw, Rw, Dw, bias = mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float(), mlp.Down_bias.detach().float()
            u = Dw.T @ r.to(Dw.device)
            for pos in v35.np_positions(row):
                C = writers_at_mlp_input(tr, pos, layer)
                names = list(C); x = sum(C.values()); rho = float(x.pow(2).mean().sqrt())
                M = torch.stack([C[w] for w in names]).to(Dw.device)
                T = (((M @ Lw.T) * u) @ (M @ Rw.T).T) / (rho * rho)
                total = float(T.sum() + r.to(Dw.device) @ bias); true = float(r.to(Dw.device) @ tr[(f"mlp:{layer:02d}", pos)].to(Dw.device))
                closure = max(closure, abs(total - true) / max(abs(true), 1e-6))
                entry[(layer, pos)] = (names, T.cpu())
        tables.append(entry)
    partner = L.partner_of(rows)
    acc = {}
    contrast = 0.0
    for i, row in enumerate(rows):
        if not row.present: continue
        j = rows.index(partner[row.row_id])
        for layer in MLPS:
            for k, pos in enumerate(v35.np_positions(row)):
                names, Ta = tables[i][(layer, pos)]; _, Tb = tables[j][(layer, v35.np_positions(rows[j])[k])]
                D = (Ta - Tb); D = (D + D.T) / 2
                for a_ in range(len(names)):
                    for b_ in range(a_, len(names)):
                        key = f"{names[a_]} x {names[b_]}"; val = float(D[a_, b_] * (1 if a_ == b_ else 2))
                        acc[key] = acc.get(key, 0.0) + val; contrast += val
    shares = {k: v / contrast for k, v in acc.items()}
    ranked = sorted(shares, key=lambda k: -abs(shares[k]))
    with81 = sum(v for k, v in shares.items() if "attnhead:08:1" in k.split(" x "))
    print("contrast", round(contrast, 2), "with 8.1", round(with81, 3), "top", [(k, round(shares[k], 3)) for k in ranked[:8]])
    predictions = {"pred_a_bilinear_closure": closure <= CLOSURE_TOL, "pred_b_top_pair_involves_head_8_1": "attnhead:08:1" in ranked[0].split(" x "), "pred_c_pairs_with_8_1_carry_half": with81 >= HALF}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_mlp_np_pair_fold_result_v41", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "contrast": contrast,
              "pairs_with_8_1_share": with81, "top20": [(k, shares[k]) for k in ranked[:20]], "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
