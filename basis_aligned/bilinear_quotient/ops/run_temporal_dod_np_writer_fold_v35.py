#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_top_four_writers_carry_070 pred_c_no_token_only_share
"""Temporal will/had DoD battery, step 8 (v35): FOLD the subject-NP state head 11.3 reads, by writer.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v30 (11.3's oriented coefficient contrast is
read at the first determiner (0.47) and the agent (0.31), cue 0.15, inherited ~0).

WHY. 11.3's port is the contextual state at `the`/agent on its weight-only reader direction r = V_h^T v_h. That
state at block 11's input is an exact lambda-weighted sum of writers (embedding, attn/mlp 0-10). Same fold as the
aspectual v13, oriented tomorrow - earlier, pooled over the two NP positions. If the token-only (embedding) share is
~0 and the writers are few, the port is a nameable contextual write; if diffuse, the kill criterion applies.

PREDICTIONS (scored as written; failures preserved)
    pred_a_writer_closure              writer sum reproduces live_11 at every NP position within relative 1e-3
    pred_b_top_four_writers_carry_070  the four largest |share| writers carry >= 0.70 (prior: unsure)
    pred_c_no_token_only_share         embedding share < 0.10 (the adverb's identity must travel by context)

PRICE (registered maximum): 2 batches x (trace 1 + pattern fold 1) = 4 forwards; 0 backwards; 0 fits. Bar <= 8.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_np_writer_fold_v35_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_np_writer_fold_v35"
CLOSURE_TOL, TOP4_MIN, EMBED_MAX = 1e-3, 0.70, 0.10
FORWARDS_MAX = 8
COMP = L.Component("attn11_h3_final", 11, "attn", (3,), "final")
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(11) for k in ("attn", "mlp")]


def np_positions(row):
    n = len(row.ids)
    return (n - 5, n - 4)   # first determiner, agent


def writer_contributions(tr, pos, upto):
    x0 = tr[("embed", pos)]
    C = {"embed": x0.clone()}
    for l in range(upto):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C:
            C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0
        C[f"attn:{l:02d}"] = tr[(f"attn:{l:02d}", pos)].clone()
        C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    return C


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    model = backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMP,), v28.WILL, v28.HAD)
    reader = L.reader_directions(model, COMP, fw.directions)[3]
    forwards = 0
    traces, patterns = [], []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, np_positions, upto_layer=11)); forwards += 1
        out, lamb = L.head_source_terms(fw, chunk, COMP, fw.directions); forwards += 1
        patterns.extend(out)
    l0, l1 = float(model.transformer.h[11].lambdas[0]), float(model.transformer.h[11].lambdas[1])
    closure, per_row = 0.0, []
    for row, tr, pat in zip(rows, traces, patterns):
        entry = {}
        for pos in np_positions(row):
            C = writer_contributions(tr, pos, upto=11)
            C = {k: l0 * v for k, v in C.items()}; C["embed"] = C["embed"] + l1 * tr[("embed", pos)]
            recon = sum(C.values()); true = tr[("live", pos)]
            closure = max(closure, float((recon - true).norm() / true.norm()))
            rms = float(true.pow(2).mean().sqrt()); p = pat[3]["pattern"][pos]; scale = p * (1 - lamb) / rms
            r = reader.to(true.device)
            entry[pos] = {w: float(r @ C[w]) * scale for w in WRITERS}
        per_row.append(entry)
    partner = L.partner_of(rows)
    totals = {w: 0.0 for w in WRITERS}; contrast = 0.0
    for i, row in enumerate(rows):
        if not row.present: continue
        j = rows.index(partner[row.row_id])
        for k, pos in enumerate(np_positions(row)):
            a, b = per_row[i][pos], per_row[j][np_positions(rows[j])[k]]
            for w in WRITERS:
                totals[w] += a[w] - b[w]
            contrast += sum(a.values()) - sum(b.values())
    shares = {w: totals[w] / contrast for w in WRITERS}
    ranked = sorted(WRITERS, key=lambda w: -abs(shares[w]))
    top4 = sum(shares[w] for w in ranked[:4])
    print("contrast", round(contrast, 2), "top", [(w, round(shares[w], 3)) for w in ranked[:8]], "top4", round(top4, 3), "embed", round(shares["embed"], 3))
    predictions = {"pred_a_writer_closure": closure <= CLOSURE_TOL, "pred_b_top_four_writers_carry_070": top4 >= TOP4_MIN, "pred_c_no_token_only_share": abs(shares["embed"]) < EMBED_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_np_writer_fold_result_v35", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "contrast_total": contrast, "shares": shares,
              "ranked": ranked, "top4_share": top4, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
