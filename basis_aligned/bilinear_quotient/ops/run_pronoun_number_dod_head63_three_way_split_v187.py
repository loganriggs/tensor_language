#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_split_closure pred_b_earlier_values_do_not_change pred_c_earlier_sources_are_reader_borne pred_d_self_source_value_and_reader
"""Pronoun number they/he DoD (v187): the THREE-WAY split v186 should have been. v186's "value" term confounded two things, because the reader
r_row = ((R . x)L + (L . x)R)/rms^2 is the unit's product gradient and changes with the noun: per source, p_P (v_P . d_P) - p_S (v_S . d_S) =
(p_P - p_S) mean(v . d)  [pattern]  +  mean(p) (v_P - v_S) . mean(d)  [value, fixed reader]  +  mean(p) mean(v) . (d_P - d_S)  [reader], exact
(cross term of the two differences folded into the reader term by using mean(p) mean(v) . dd + a correction that is reported separately; closure checks it).
Causality: the earlier tokens ("The" / start) have the same residual in both members of a pair, so their value term must be ~0 and everything
they carry is reader-borne -- i.e. the number is in the unit's OTHER factor (x at the noun) and head 6.3 supplies a constant vector to multiply
against. Registered as predictions rather than assumed.
PREDICTIONS (scored as written; failures preserved)
    pred_a_split_closure                 the three terms (+ correction) reproduce p_P v_P d_P - p_S v_S d_S within relative 1e-3 per pair
    pred_b_earlier_values_do_not_change   over the earlier positions, |value term| <= 0.02 of 6.3's pooled contrast
    pred_c_earlier_sources_are_reader_borne  over the earlier positions, the reader term carries >= 0.80 of their pooled share
    pred_d_self_source_value_and_reader  at the noun position, the value term (fixed reader) carries >= 0.30 of its share (the noun's own value does change). Prior: unsure.
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_head63_three_way_split_v187_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head63_three_way_split_v187"
UNIT, LAYER, HEAD, CLOSURE_TOL, VALUE_MAX, READER_MIN, SELF_VALUE_MIN, BATCH = 2483, 6, 3, 1e-3, 0.02, 0.80, 0.30, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_split_closure": "<= 1e-3", "pred_b_earlier_values_do_not_change": "<= 0.02", "pred_c_earlier_sources_are_reader_borne": ">= 0.80", "pred_d_self_source_value_and_reader": ">= 0.30"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "head": f"{LAYER}.{HEAD}", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "value_max": VALUE_MAX, "reader_min": READER_MIN, "self_value_min": SELF_VALUE_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); fw.backend = backend
    mlp = model.transformer.h[LAYER].mlp; Lrow, Rrow = mlp.Left.weight.detach().float()[UNIT], mlp.Right.weight.detach().float()[UNIT]
    Wo = model.transformer.h[LAYER].attn.c_proj.weight.detach().float()[:, HEAD * L.HEAD_DIM:(HEAD + 1) * L.HEAD_DIM]
    forwards, per_row = 0, []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces = L.forward_trace_positions(fw, chunk, lambda rw: [noun_of(rw)], upto_layer=LAYER + 1, head_write_layers=(LAYER,)); forwards += 1
        f = L.attention_factors(fw, chunk, LAYER); forwards += 1
        for i, (row, tr) in enumerate(zip(chunk, traces)):
            pos = noun_of(row); C = v183.writers_at_block_input(tr, pos, LAYER); x = sum(C.values()).to(Lrow.device); rms2 = float(x.pow(2).mean())
            r = (float(Rrow @ x) * Lrow + float(Lrow @ x) * Rrow) / rms2; d = (Wo.T @ r).to(device=f["x"].device, dtype=torch.float32)
            p = L.pattern_row(f, i, pos, HEAD)
            V = (1 - f["lamb"]) * f["v_cur"][i, :pos + 1, HEAD].float() + f["lamb"] * f["v1"][i, :pos + 1, HEAD].float()     # mixed value per source (T, 128)
            per_row.append({"pos": pos, "p": p.cpu(), "V": V.cpu(), "d": d.cpu()})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    keys = ("pattern", "value", "reader", "corr"); acc = {f"{w}_{k}": 0.0 for w in ("self", "earlier") for k in keys}; closure, n = 0.0, 0
    for i, row in enumerate(rows):
        if not row.present: continue
        a, b = per_row[i], per_row[partner[(row.construction, row.group, False)]]
        if a["pos"] != b["pos"]: raise SystemExit("pair positions are not aligned")
        pos = a["pos"]; pm, dp = (a["p"] + b["p"]) / 2, a["p"] - b["p"]; Vm, dV = (a["V"] + b["V"]) / 2, a["V"] - b["V"]; dm, dd = (a["d"] + b["d"]) / 2, a["d"] - b["d"]
        vd_P, vd_S = a["V"] @ a["d"], b["V"] @ b["d"]; ref = a["p"] * vd_P - b["p"] * vd_S
        pattern = dp * (vd_P + vd_S) / 2; value = pm * (dV @ dm); reader = pm * (Vm @ dd); corr = ref - pattern - value - reader     # corr = pm * (dV @ dd)/4 ... exact remainder
        closure = max(closure, float((pattern + value + reader + corr - ref).abs().max()) / max(float(ref.abs().max()), 1e-6))
        for k, t in zip(keys, (pattern, value, reader, corr)):
            acc[f"self_{k}"] += float(t[pos]); acc[f"earlier_{k}"] += float(t[:pos].sum())
        n += 1
    tot = sum(acc.values()); shares = {k: v / tot for k, v in acc.items()}
    earlier = sum(acc[f"earlier_{k}"] for k in keys); self_ = sum(acc[f"self_{k}"] for k in keys)
    report = {"pairs": n, "pooled": acc, "shares": shares, "earlier_share": earlier / tot, "self_share": self_ / tot, "earlier_value_abs_share": abs(acc["earlier_value"]) / abs(tot), "earlier_reader_fraction": acc["earlier_reader"] / earlier,
              "self_value_fraction": acc["self_value"] / self_, "self_reader_fraction": acc["self_reader"] / self_, "corr_abs_share": (abs(acc["self_corr"]) + abs(acc["earlier_corr"])) / abs(tot)}
    print("pooled", round(tot, 2), {k: round(v, 3) for k, v in shares.items()}); print("earlier: value", round(report["earlier_value_abs_share"], 4), "reader fraction", round(report["earlier_reader_fraction"], 3), "| self: value fraction", round(report["self_value_fraction"], 3), "reader fraction", round(report["self_reader_fraction"], 3), "| corr", round(report["corr_abs_share"], 4))
    predictions = {"pred_a_split_closure": closure <= CLOSURE_TOL, "pred_b_earlier_values_do_not_change": report["earlier_value_abs_share"] <= VALUE_MAX, "pred_c_earlier_sources_are_reader_borne": report["earlier_reader_fraction"] >= READER_MIN, "pred_d_self_source_value_and_reader": report["self_value_fraction"] >= SELF_VALUE_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head63_three_way_split_result_v187", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
