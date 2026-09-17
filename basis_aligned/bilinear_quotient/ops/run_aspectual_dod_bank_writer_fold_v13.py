#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_mlp4_is_the_largest_bank_writer pred_c_embedding_share_at_least_015 pred_d_top_three_writers_carry_060
"""Aspectual has/had definition-of-done battery, step 13: FOLD the block-9 bank read one step back.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v10 (heads 9.1/9.4 read
`last`+period+`the` through the current-block value branch; inherited branch ~0), v12 (8.1 closed).

WHY. The last open ports of the component are the source states heads 9.1/9.4 read. Their
contribution is exact: c_h ∋ sum_s p_h(t,s) (1-lamb) r_h . rms_norm(live_9[s]) with the
weight-only reader direction r_h = V_h^T v_hat_h, and live_9[s] (the state block 9 reads at
source s) is an exact lambda-weighted sum of every earlier writer: embedding, attn:00..08,
mlp:00..08. This run attributes the oriented (since - by) reader-projected bank contrast to
those writers, per head, pooled over the three bank positions. Evidence tag: fold, opened rows.
The released path predicts MLP4 (two-term writer at these positions) as the decisive writer.

PREDICTIONS (scored as written; failures preserved)
    pred_a_writer_closure                    writer sum reproduces live_9 at every bank position within
                                             relative L2 1e-3
    pred_b_mlp4_is_the_largest_bank_writer   for both 9.1 and 9.4 the largest |share| writer is mlp:04
    pred_c_embedding_share_at_least_015      pooled over both heads the embedding (token-only) share of
                                             the contrast is >= 0.15. Prior: unsure.
    pred_d_top_three_writers_carry_060       for each head the three largest writers carry >= 0.60

PRICE (registered maximum): 2 batches x (trace 1 + pattern fold 1) = 4 forwards; 0 backwards; 0 fits.
Bar <= 10.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_bank_writer_fold_v13_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_bank_writer_fold_v13"
TOKENS = {"has": 468, "had": 550}
CLOSURE_TOL, EMBED_MIN, TOP3_MIN = 1e-3, 0.15, 0.60
FORWARDS_MAX = 10
COMP = L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final")
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(9) for k in ("attn", "mlp")]


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": ["9.1", "9.4"],
            "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "embed_min": EMBED_MIN, "top3_min": TOP3_MIN}}


def writer_contributions(tr, pos, upto=9):
    """Exact decomposition of live_9[pos] into writer vectors via the lambda recurrence."""
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
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    model = backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMP,), TOKENS["has"], TOKENS["had"])
    readers = L.reader_directions(model, COMP, fw.directions)
    forwards = 0
    traces, patterns = [], []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda r: r.source_positions, upto_layer=9)); forwards += 1
        out, lamb = L.head_source_terms(fw, chunk, COMP, fw.directions); forwards += 1
        patterns.extend(out)
    # block-9 lambdas are needed for the live_9 mix: read them from the model
    block9 = model.transformer.h[9]
    l0_9, l1_9 = float(block9.lambdas[0]), float(block9.lambdas[1])

    closure_max = 0.0
    per_row = []
    for row, tr, pat in zip(rows, traces, patterns):
        entry = {}
        for pos in row.source_positions:
            C = writer_contributions(tr, pos)
            x9 = sum(C.values())
            live9 = l0_9 * x9 + l1_9 * tr[("embed", pos)]
            C = {k: l0_9 * v for k, v in C.items()}
            C["embed"] = C["embed"] + l1_9 * tr[("embed", pos)]
            recon = sum(C.values())
            true = tr[("live", pos)]
            closure_max = max(closure_max, float((recon - true).norm() / true.norm()))
            rms = float(true.pow(2).mean().sqrt())
            for head in COMP.heads:
                p = pat[head]["pattern"][pos]
                scale = p * (1 - lamb) / rms
                r = readers[head].to(true.device)
                entry[(pos, head)] = {w: float(r @ C[w]) * scale for w in WRITERS}
        per_row.append(entry)

    partner = L.partner_of(rows)
    report = {}
    for head in COMP.heads:
        totals = {w: 0.0 for w in WRITERS}
        contrast = 0.0
        for i, row in enumerate(rows):
            if not row.present:
                continue
            j = rows.index(partner[row.row_id])
            for k, pos in enumerate(row.source_positions):
                pos2 = rows[j].source_positions[k]
                a, b = per_row[i][(pos, head)], per_row[j][(pos2, head)]
                for w in WRITERS:
                    totals[w] += a[w] - b[w]
                contrast += sum(a.values()) - sum(b.values())
        shares = {w: totals[w] / contrast for w in WRITERS}
        ranked = sorted(WRITERS, key=lambda w: -abs(shares[w]))
        report[f"9.{head}"] = {"contrast_total": contrast, "shares": shares, "largest": ranked[0],
                               "top3": ranked[:3], "top3_share": sum(shares[w] for w in ranked[:3])}
        print(f"9.{head}", "contrast", round(contrast, 3), "top", [(w, round(shares[w], 3)) for w in ranked[:6]], "embed", round(shares["embed"], 3))
    pooled_embed = sum(report[k]["shares"]["embed"] * report[k]["contrast_total"] for k in report) / sum(report[k]["contrast_total"] for k in report)
    predictions = {
        "pred_a_writer_closure": closure_max <= CLOSURE_TOL,
        "pred_b_mlp4_is_the_largest_bank_writer": all(r["largest"] == "mlp:04" for r in report.values()),
        "pred_c_embedding_share_at_least_015": pooled_embed >= EMBED_MIN,
        "pred_d_top_three_writers_carry_060": all(r["top3_share"] >= TOP3_MIN for r in report.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_bank_writer_fold_result_v13", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "closure_max_relative_error": closure_max, "lambda_block9_value_mix": lamb, "heads": report,
              "pooled_embedding_share": pooled_embed, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure_max, "pooled_embed": pooled_embed, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
