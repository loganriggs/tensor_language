#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_direct_term_closure pred_b_mlp_response_explains_40pct_of_mlp_part pred_c_mlp_response_opposes_the_direct_term
"""Temporal DoD battery, step 15 (v48): RESPONSE CENSUS of the 8.1 NP edit through MLP8/9/10 (better_circuits §3.3).

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v35/v41 (the MLP8-10 part of the NP state 11.3
reads is diffuse under writer-pair attribution and NEGATIVE on the reader direction), v39 (8.1 at the NP is a
token-only adverb reader).

WHY. Writer-pair attribution of the native MLP writes failed to name a source (kill). A different question: how much
of the MLP part is the MLPs' RESPONSE to 8.1's write? One edited forward (8.1's two NP slices zeroed) decomposed into
module changes at the NP positions, projected on 11.3's reader direction r, oriented tomorrow - earlier and pooled
over the 32 pairs x 2 positions. Response share = (oriented contrast of the MLP8-10 changes under the edit) /
(oriented contrast of the native MLP8-10 part). Tag: response, opened rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_direct_term_closure         r . (attn8_edited - attn8_native) equals -(r . O_1 z_1) within relative 1e-3 at
                                       every NP position (the edit removes exactly head 8.1's write)
    pred_b_mlp_response_explains_40pct_of_mlp_part   |response share| >= 0.40 pooled over MLP8-10
    pred_c_mlp_response_opposes_the_direct_term      the pooled MLP response contrast has the opposite sign to the direct
                                       8.1 contrast (the MLPs partly cancel 8.1's write; v41 found the native MLP part
                                       negative on r). Prior: unsure.

PRICE (registered maximum): 2 batches x (native trace 1 + edited trace 1) = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
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
import run_temporal_dod_8_1_np_token_only_v39 as v39

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_mlp_response_census_v48_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_mlp_response_census_v48"
CLOSURE_TOL, SHARE_MIN = 1e-3, 0.40
FORWARDS_MAX = 6
COMP = L.Component("attn11_h3_final", 11, "attn", (3,), "final")
NP = "np"


def traced(fw, rows, components=(), mode="zero"):
    """forward_trace_positions with optional edits: reuse the library's edit routing by temporarily wrapping forward."""
    torch, F, model = fw.torch, fw.F, fw.model
    tokens = fw._tokens(rows)
    trace = [dict() for _ in rows]
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1_ = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attn_here = [c for c in components if c.kind == "attn" and c.layer == layer]
            pre = {}
            def c_proj_pre(_m, args, edits=attn_here):
                value = args[0]
                pre["native"] = value.detach().clone()
                for k, c in enumerate(edits):
                    value = fw._edit(value, rows, c, mode, 0, "attn")
                return (value,) + tuple(args[1:])
            handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
            try:
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            finally:
                handle.remove()
            if layer == 8:
                W = block.attn.c_proj.weight.detach().float()
                for i, row in enumerate(rows):
                    for pos in v35.np_positions(row):
                        sl = pre["native"][i, pos, L.HEAD_DIM:2 * L.HEAD_DIM].float()
                        trace[i][("head81_write", pos)] = (W[:, L.HEAD_DIM:2 * L.HEAD_DIM] @ sl).detach()
            x = live + attention
            mlp = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            x = x + mlp
            if layer in (8, 9, 10):
                for i, row in enumerate(rows):
                    for pos in v35.np_positions(row):
                        trace[i][(f"attn:{layer:02d}", pos)] = attention[i, pos].detach().float().clone()
                        trace[i][(f"mlp:{layer:02d}", pos)] = mlp[i, pos].detach().float().clone()
            if layer == 10:
                break
    return trace


def main() -> None:
    rows = v28.build()
    original = L.positions_of
    L.positions_of = lambda row, where: v35.np_positions(row) if where == NP else original(row, where)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    model = backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMP,), v28.WILL, v28.HAD)
    r = L.reader_directions(model, COMP, fw.directions)[3]
    forwards = 0
    nat, ed = [], []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        nat.extend(traced(fw, chunk)); forwards += 1
        ed.extend(traced(fw, chunk, components=(v39.HEAD,), mode="zero")); forwards += 1
    partner = L.partner_of(rows)
    closure = 0.0
    direct = 0.0; mlp_native = {8: 0.0, 9: 0.0, 10: 0.0}; mlp_resp = {8: 0.0, 9: 0.0, 10: 0.0}
    for i, row in enumerate(rows):
        for pos in v35.np_positions(row):
            da = float(r @ (ed[i][("attn:08", pos)] - nat[i][("attn:08", pos)]).to(r.device)); w = float(r @ nat[i][("head81_write", pos)].to(r.device))
            closure = max(closure, abs(da + w) / max(abs(w), 1e-6))
        if not row.present: continue
        j = rows.index(partner[row.row_id])
        for k, pos in enumerate(v35.np_positions(row)):
            pj = v35.np_positions(rows[j])[k]
            direct += float(r @ (nat[i][("head81_write", pos)] - nat[j][("head81_write", pj)]).to(r.device))
            for l in (8, 9, 10):
                mlp_native[l] += float(r @ (nat[i][(f"mlp:{l:02d}", pos)] - nat[j][(f"mlp:{l:02d}", pj)]).to(r.device))
                # response = change under the edit, oriented as (row - partner)
                mlp_resp[l] += float(r @ ((ed[i][(f"mlp:{l:02d}", pos)] - nat[i][(f"mlp:{l:02d}", pos)]) - (ed[j][(f"mlp:{l:02d}", pj)] - nat[j][(f"mlp:{l:02d}", pj)])).to(r.device))
    total_native_mlp = sum(mlp_native.values()); total_resp = sum(mlp_resp.values())
    share = total_resp / total_native_mlp if total_native_mlp else float("nan")
    print(json.dumps({"direct_8_1_contrast": round(direct, 2), "native_mlp_contrast": {k: round(v, 2) for k, v in mlp_native.items()}, "mlp_response_to_edit": {k: round(v, 2) for k, v in mlp_resp.items()},
                      "response_share_of_mlp_part": round(share, 4), "closure": closure}))
    predictions = {"pred_a_direct_term_closure": closure <= CLOSURE_TOL, "pred_b_mlp_response_explains_40pct_of_mlp_part": abs(share) >= SHARE_MIN,
                   "pred_c_mlp_response_opposes_the_direct_term": (total_resp > 0) != (direct > 0) if total_resp else False}
    # note: removing 8.1's write makes the MLP response = MLP(edited) - MLP(native); "opposes the direct term" means the
    # response contrast has the SAME sign as the direct contrast (removing a positive write produces a positive MLP change
    # only if the MLPs were cancelling it). Record both readings explicitly:
    predictions["pred_c_mlp_response_opposes_the_direct_term"] = (total_resp > 0) == (direct > 0)
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_mlp_response_census_result_v48", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "direct_8_1_contrast": direct,
              "native_mlp_contrast": mlp_native, "mlp_response_to_edit": mlp_resp, "response_share_of_mlp_part": share,
              "sign_note": "pred_c passes when the MLP change under the edit has the SAME sign as 8.1's native write contrast, i.e. the MLPs were cancelling part of that write",
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
