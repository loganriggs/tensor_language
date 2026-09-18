#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_pronoun_state_is_mostly_mlp_written pred_c_token_readers_do_not_write_10_5_source pred_d_embedding_share_at_least_010 pred_e_largest_mlp_writer_is_block_8_or_9
"""Reflexive person DoD (v118): WRITER FOLD of the state head 10.5 -- the person set's contextual member -- reads at the I / you position.

Lane: Claude circuit lane. Parents: v112 (10.5: 52% of its contrast from the pronoun position, 29% token-only; 24% final, 24% other), v113
(the other three heads close to tokens). Exact recurrence at the pronoun position into embedding, heads of blocks 0-9, biases, MLPs 0-9, projected
on r = V_{10.5}^T v_hat (`dod_folds.writer_fold`, the v82 / v103 body). Evidence tag: fold, fresh rows.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_writer_closure                          recon of live_10 within relative 1e-3
    pred_b_pronoun_state_is_mostly_mlp_written     MLP writers >= 0.50 of the reader-projected contrast at the pronoun position (v82's shape)
    pred_c_token_readers_do_not_write_10_5_source  attnhead 8.1 carries <= 0.10 (the token reader writes at the final query, not at the pronoun)
    pred_d_embedding_share_at_least_010            embedding >= 0.10 (the pronoun's own token identity survives to block 10)
    pred_e_largest_mlp_writer_is_block_8_or_9      the largest MLP writer is mlp:08 or mlp:09
PRICE (registered maximum): 3 batches x (pattern fold + positional trace) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_person_reflexive_dod_battery_v104 as line
import dod_battery, dod_folds

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/person_dod_head_10_5_writer_fold_v118_result.json"
CANDIDATE_ID = "reflexive_person.i_vs_you.dod_head_10_5_writer_fold_v118"
LAYER, HEAD = 10, 5
CLOSURE_TOL, MLP_MIN, H81_MAX, EMBED_MIN = 1e-3, 0.50, 0.10, 0.10
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_writer_closure": "<= 1e-3", "pred_b_pronoun_state_is_mostly_mlp_written": ">= 0.50", "pred_c_token_readers_do_not_write_10_5_source": "8.1 <= 0.10",
               "pred_d_embedding_share_at_least_010": ">= 0.10", "pred_e_largest_mlp_writer_is_block_8_or_9": "mlp:08 | mlp:09"}


def main() -> None:
    rows, pos, neg, agents, objects = line.build()
    pronouns = {L._single(" I"), L._single("I"), L._single(" you"), L._single("you")}
    positions_of = lambda row: {"pronoun": next(i for i, t in enumerate(row.ids) if t in pronouns)}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "head": "10.5", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mlp_min": MLP_MIN, "h81_max": H81_MAX, "embed_min": EMBED_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    comp = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, pos, neg, ((LAYER, HEAD),)).set_components())
    fw.directions = L.readout_directions(model, (comp,), pos, neg)
    reader = L.reader_directions(model, comp, fw.directions)[HEAD]
    forwards, pattern = 0, {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
        for row, entry in zip(chunk, out):
            pattern[row.row_id] = entry[HEAD]["pattern"]
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    result, n = dod_folds.writer_fold(fw, model, rows, LAYER, HEAD, reader, positions_of, pattern, partner); forwards += n
    P = result["positions"]["pronoun"]; mlps = P["mlp_by_block"]; largest_mlp = max(mlps, key=lambda k: abs(mlps[k]))
    print("10.5 at pronoun: heads", round(P["head_part"], 3), "mlp", round(P["mlp_part"], 3), "embed", round(P["embed_share"], 3), "top heads", [(w, round(s, 3)) for w, s in P["top_heads"][:5]], "mlp", {k: round(v, 3) for k, v in mlps.items()})
    predictions = {"pred_a_writer_closure": result["closure_max"] <= CLOSURE_TOL, "pred_b_pronoun_state_is_mostly_mlp_written": P["mlp_part"] >= MLP_MIN,
                   "pred_c_token_readers_do_not_write_10_5_source": abs(P["shares"]["attnhead:08:1"]) <= H81_MAX, "pred_d_embedding_share_at_least_010": P["embed_share"] >= EMBED_MIN,
                   "pred_e_largest_mlp_writer_is_block_8_or_9": largest_mlp in ("mlp:08", "mlp:09")}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "person_dod_head_10_5_writer_fold_result_v118", "candidate_id": CANDIDATE_ID, "plan": plan, "result": result, "largest_mlp": largest_mlp, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
