#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_replays_v10 pred_b_recompute_replays_v11b pred_c_generalized_fold_replays_v14
"""Refactor equivalence check for `aspectual_dod_lib.py` (2026-09-17 review-2 improvement).

WHY. The library's three attention recomputations (`head_source_terms`, `source_restricted_slices`,
`head_source_terms_at`) were merged onto one `attention_factors` helper. Standing lesson: control the new
code path. This run replays three landed receipts through the refactored functions and requires bitwise-
close agreement; it produces no new science.

PREDICTIONS (scored as written)
    pred_a_fold_replays_v10          per-head mean contrasts equal v10's within relative 1e-6
    pred_b_recompute_replays_v11b    the cue_inherited arm damage equals v11b's within 1e-5 logits
    pred_c_generalized_fold_replays_v14   reader-9.1 head-8.1 share equals v14's within 1e-6

PRICE (registered maximum): 2 batches x (fold 1 + capture 1 + recompute 1 + arm 1 + native 1 + zero 1 + 3 bank
folds x 1) = 18 forwards; 0 backwards; 0 fits. Bar <= 22.
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
import run_aspectual_dod_cue_source_v11b as v11b
import run_aspectual_dod_attn8_bank_fold_v14 as v14

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_lib_refactor_check_v24_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_lib_refactor_check_v24"
TOKENS = {"has": 468, "had": 550}
FORWARDS_MAX = 22
COMPONENTS = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"), L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed")
    r10 = json.loads((ROOT / "circuits/followups/aspectual_anchor_dod_source_fold_v10_result.json").read_text())
    r11 = json.loads((ROOT / "circuits/followups/aspectual_anchor_dod_cue_source_v11b_result.json").read_text())
    r14 = json.loads((ROOT / "circuits/followups/aspectual_anchor_dod_attn8_bank_fold_v14_result.json").read_text())
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, COMPONENTS, TOKENS["has"], TOKENS["had"])
    forwards = 0
    # (a) v10 replay: mean contrasts
    terms = [dict() for _ in rows]
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        for comp in COMPONENTS:
            out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
            for j, entry in enumerate(out):
                for head, val in entry.items():
                    terms[start + j][f"{comp.name}:{head}"] = val["coefficient"]
    partner = L.partner_of(rows)
    err_a = 0.0
    for key in ("attn8_h1_final:1", "attn9_h1_h4_final:1", "attn9_h1_h4_final:4"):
        c = [terms[i][key] - terms[rows.index(partner[r.row_id])][key] for i, r in enumerate(rows) if r.present]
        mine = sum(c) / len(c); ref = r10["heads"][key]["mean_contrast"]
        err_a = max(err_a, abs(mine - ref) / abs(ref))
    # (b) v11b replay: cue_inherited arm
    native, n = v1._run_arm(fw, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=(v11b.HEAD,), mode="zero"); forwards += n
    table = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        table.update(L.source_restricted_slices(fw, chunk, v11b.HEAD, lambda r, s: v11b.cat(r, s) == "cue", ("inherited",))); forwards += 1
    fw.subtract = table
    arm, n = v1._run_arm(fw, rows, components=(v11b.HEAD,), mode="replace"); forwards += n
    fw.use_subtract = False
    dmg = L.summarize(rows, native, arm)["target_damage_mean"]
    err_b = abs(dmg - r11["arms"]["cue_inherited"]["arm_damage"])
    # (c) v14 replay: reader 9.1, head 8.1 share (bank position 0 only would differ; replay the full three)
    readers = L.reader_directions(backend.model, COMPONENTS[1], fw.directions)
    O8 = backend.model.transformer.h[8].attn.c_proj.weight.detach().float()
    dirs = {hp: O8[:, hp * L.HEAD_DIM:(hp + 1) * L.HEAD_DIM].T @ readers[1].to(O8.device) for hp in range(9)}
    by_head = {hp: 0.0 for hp in range(9)}; contrast = 0.0
    folds = {k: [] for k in range(3)}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        for k in range(3):
            out, _ = L.head_source_terms_at(fw, chunk, 8, lambda r, k=k: r.source_positions[k], dirs); forwards += 1
            folds[k].extend(out)
    for i, row in enumerate(rows):
        if not row.present:
            continue
        j = rows.index(partner[row.row_id])
        for k in range(3):
            for hp in range(9):
                d = folds[k][i][hp]["total"] - folds[k][j][hp]["total"]
                by_head[hp] += d; contrast += d
    err_c = abs(by_head[1] / contrast - r14["readers"]["9.1"]["head_shares"]["8.1"])
    predictions = {"pred_a_fold_replays_v10": err_a <= 1e-6, "pred_b_recompute_replays_v11b": err_b <= 1e-5, "pred_c_generalized_fold_replays_v14": err_c <= 1e-6}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_lib_refactor_check_result_v24", "candidate_id": CANDIDATE_ID, "errors": {"v10_rel": err_a, "v11b_abs": err_b, "v14_abs": err_c},
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
