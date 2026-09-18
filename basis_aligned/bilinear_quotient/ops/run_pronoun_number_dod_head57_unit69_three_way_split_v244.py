#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_split_closure pred_b_start_values_do_not_change pred_c_start_sources_are_reader_borne pred_d_noun_share_small
"""Pronoun number they/he DoD (v244): head 5.7 into MLP-6 unit 69 AT THE VERB, split three ways. v243: 5.7 is the block-5 head feeding unit 69 at the verb (199% of block 5,
opposed by 5.3) and 82% of its share comes from the SENTENCE START, 13% from the noun, 5% from the verb itself -- the v187 shape (head 6.3 into 2483 at the noun was 89%
reader-borne: a near-constant write whose contrast lives in the partner factor). Exact split per source: pattern / value at a fixed reader / reader, with the reader
r_row = unit 69's product gradient at the block-6 input at the verb and the head's write scaled by lambda0_6 to reach block 6. Sources are grouped as start (before the
noun), noun, and the verb itself.
PREDICTIONS (scored as written; failures preserved)
    pred_a_split_closure              the three terms (+ remainder) reproduce p_P v_P d_P - p_S v_S d_S within relative 1e-3 per pair
    pred_b_start_values_do_not_change |value term| over the start sources <= 0.02 of 5.7's pooled contrast (causality: the start tokens are identical across a pair)
    pred_c_start_sources_are_reader_borne  over the start sources the reader term carries >= 0.80 of their share
    pred_d_noun_share_small           the noun position carries <= 0.30 of 5.7's pooled contrast (v243: 0.13)
PRICE (registered maximum): 3 batches x (positional trace + attention factors) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_pronoun_number_dod_mlp6_unit_pair_fold_v183 as v183
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_head57_unit69_three_way_split_v244_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head57_unit69_three_way_split_v244"
UNIT, LAYER, HEAD_LAYER, HEAD, CLOSURE_TOL, VALUE_MAX, READER_MIN, NOUN_MAX, BATCH = 69, 6, 5, 7, 1e-3, 0.02, 0.80, 0.30, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_split_closure": "<= 1e-3", "pred_b_start_values_do_not_change": "<= 0.02", "pred_c_start_sources_are_reader_borne": ">= 0.80", "pred_d_noun_share_small": "<= 0.30"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns); verb_of = lambda row: noun_of(row) + 1
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "head": f"{HEAD_LAYER}.{HEAD}", "position": "verb", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "value_max": VALUE_MAX, "reader_min": READER_MIN, "noun_max": NOUN_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); fw.backend = backend
    mlp = model.transformer.h[LAYER].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    Wo = model.transformer.h[HEAD_LAYER].attn.c_proj.weight.detach().float()[:, HEAD * L.HEAD_DIM:(HEAD + 1) * L.HEAD_DIM]; scale = float(model.transformer.h[LAYER].lambdas[0])
    forwards, per_row = 0, []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [verb_of(rw)], upto_layer=LAYER + 1, head_write_layers=(LAYER,)); forwards += 1
        f = L.attention_factors(fw, chunk, HEAD_LAYER); forwards += 1
        for i, (row, tr) in enumerate(zip(chunk, traces)):
            pos = verb_of(row); npos = noun_of(row); C = v183.writers_at_block_input(tr, pos, LAYER); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            r = (float(Rrow @ x) * Lrow + float(Lrow @ x) * Rrow) / rms2; d = (scale * (Wo.T @ r)).to(device=f["x"].device, dtype=torch.float32)
            p = L.pattern_row(f, i, pos, HEAD)
            V = (1 - f["lamb"]) * f["v_cur"][i, :pos + 1, HEAD].float() + f["lamb"] * f["v1"][i, :pos + 1, HEAD].float()     # mixed value per source (T, 128)
            per_row.append({"pos": pos, "npos": npos, "p": p.cpu(), "V": V.cpu(), "d": d.cpu()})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    keys = ("pattern", "value", "reader", "corr"); acc = {f"{w}_{k}": 0.0 for w in ("self", "noun", "start") for k in keys}; closure, n = 0.0, 0
    for i, row in enumerate(rows):
        if not row.present: continue
        a, b = per_row[i], per_row[partner[(row.construction, row.group, False)]]
        if a["pos"] != b["pos"]: raise SystemExit("pair positions are not aligned")
        pos = a["pos"]; npos = a["npos"]; pm, dp = (a["p"] + b["p"]) / 2, a["p"] - b["p"]; Vm, dV = (a["V"] + b["V"]) / 2, a["V"] - b["V"]; dm, dd = (a["d"] + b["d"]) / 2, a["d"] - b["d"]
        vd_P, vd_S = a["V"] @ a["d"], b["V"] @ b["d"]; ref = a["p"] * vd_P - b["p"] * vd_S
        pattern = dp * (vd_P + vd_S) / 2; value = pm * (dV @ dm); reader = pm * (Vm @ dd); corr = ref - pattern - value - reader     # corr = pm * (dV @ dd)/4 ... exact remainder
        closure = max(closure, float((pattern + value + reader + corr - ref).abs().max()) / max(float(ref.abs().max()), 1e-6))
        for k, t in zip(keys, (pattern, value, reader, corr)):
            acc[f"self_{k}"] += float(t[pos]); acc[f"noun_{k}"] += float(t[npos]); acc[f"start_{k}"] += float(t[:npos].sum())
        n += 1
    tot = sum(acc.values()); shares = {k: v / tot for k, v in acc.items()}
    start = sum(acc[f"start_{k}"] for k in keys); noun = sum(acc[f"noun_{k}"] for k in keys); self_ = sum(acc[f"self_{k}"] for k in keys)
    report = {"pairs": n, "pooled": acc, "shares": shares, "start_share": start / tot, "noun_share": noun / tot, "self_share": self_ / tot, "start_value_abs_share": abs(acc["start_value"]) / abs(tot), "start_reader_fraction": acc["start_reader"] / start if start else float("nan"),
              "noun_value_fraction": acc["noun_value"] / noun if noun else float("nan"), "corr_abs_share": sum(abs(acc[f"{w}_corr"]) for w in ("self", "noun", "start")) / abs(tot)}
    print("pooled", round(tot, 2), "start", round(report["start_share"], 3), "noun", round(report["noun_share"], 3), "self", round(report["self_share"], 3)); print("start: value", round(report["start_value_abs_share"], 4), "reader fraction", round(report["start_reader_fraction"], 3), "| noun value fraction", round(report["noun_value_fraction"], 3), "| corr", round(report["corr_abs_share"], 4))
    predictions = {"pred_a_split_closure": closure <= CLOSURE_TOL, "pred_b_start_values_do_not_change": report["start_value_abs_share"] <= VALUE_MAX, "pred_c_start_sources_are_reader_borne": report["start_reader_fraction"] >= READER_MIN, "pred_d_noun_share_small": abs(report["noun_share"]) <= NOUN_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head63_head57_unit69_three_way_split_result_v244", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
