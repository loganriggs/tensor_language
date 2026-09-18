#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_split_closure pred_b_earlier_sources_are_pattern_borne pred_c_self_source_is_value_borne pred_d_pattern_terms_carry_half_overall
"""Pronoun number they/he DoD (v186): is head 6.3's number signal carried by its PATTERN or its VALUES? v185: 6.3's linear contribution to MLP-6 unit 2483
at the noun is 33% self-position and 67% earlier tokens ("The" / start), whose values are the same in the plural and singular member of every pair --
so their contrast can only come from the pattern (the noun's query gating a fixed source). Exact split, per source s of every aligned pair
(plural P, singular S; same positions): p_P v_P - p_S v_S = (p_P - p_S) (v_P + v_S)/2 + (p_P + p_S)/2 (v_P - v_S), with v the reader-projected
(current + token-only) value; the first term is pattern-borne, the second value-borne. Reader and closure as in v185.
PREDICTIONS (scored as written; failures preserved)
    pred_a_split_closure                   pattern + value terms = the v185 source terms within relative 1e-3 per pair (identity; a code check)
    pred_b_earlier_sources_are_pattern_borne  over the earlier positions, the pattern term carries >= 0.80 of their pooled contrast
    pred_c_self_source_is_value_borne      at the noun position, the value term carries >= 0.50 of its pooled contrast. Prior: unsure.
    pred_d_pattern_terms_carry_half_overall pattern terms carry >= 0.50 of 6.3's pooled contrast overall
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_head63_pattern_value_split_v186_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head63_pattern_value_split_v186"
UNIT, LAYER, HEAD, CLOSURE_TOL, EARLIER_MIN, SELF_MIN, OVERALL_MIN, BATCH = 2483, 6, 3, 1e-3, 0.80, 0.50, 0.50, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_split_closure": "<= 1e-3", "pred_b_earlier_sources_are_pattern_borne": ">= 0.80", "pred_c_self_source_is_value_borne": ">= 0.50", "pred_d_pattern_terms_carry_half_overall": ">= 0.50"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "head": f"{LAYER}.{HEAD}", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "earlier_min": EARLIER_MIN, "self_min": SELF_MIN, "overall_min": OVERALL_MIN}}
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
            v = (1 - f["lamb"]) * (f["v_cur"][i, :pos + 1, HEAD].float() @ d) + f["lamb"] * (f["v1"][i, :pos + 1, HEAD].float() @ d)     # reader-projected value per source
            per_row.append({"pos": pos, "p": p.cpu(), "v": v.cpu()})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    acc = {"self_pattern": 0.0, "self_value": 0.0, "earlier_pattern": 0.0, "earlier_value": 0.0}; closure, n = 0.0, 0
    for i, row in enumerate(rows):
        if not row.present: continue
        a, b = per_row[i], per_row[partner[(row.construction, row.group, False)]]
        if a["pos"] != b["pos"]: raise SystemExit("pair positions are not aligned")
        pos = a["pos"]; pat = (a["p"] - b["p"]) * (a["v"] + b["v"]) / 2; val = (a["p"] + b["p"]) / 2 * (a["v"] - b["v"]); ref = a["p"] * a["v"] - b["p"] * b["v"]
        closure = max(closure, float((pat + val - ref).abs().max()) / max(float(ref.abs().max()), 1e-6))
        acc["self_pattern"] += float(pat[pos]); acc["self_value"] += float(val[pos]); acc["earlier_pattern"] += float(pat[:pos].sum()); acc["earlier_value"] += float(val[:pos].sum()); n += 1
    tot = sum(acc.values()); shares = {k: v / tot for k, v in acc.items()}
    earlier = acc["earlier_pattern"] + acc["earlier_value"]; self_ = acc["self_pattern"] + acc["self_value"]
    earlier_pat = acc["earlier_pattern"] / earlier if earlier else float("nan"); self_val = acc["self_value"] / self_ if self_ else float("nan"); pattern_overall = shares["self_pattern"] + shares["earlier_pattern"]
    report = {"pairs": n, "pooled": acc, "shares": shares, "earlier_share": earlier / tot, "self_share": self_ / tot, "earlier_pattern_fraction": earlier_pat, "self_value_fraction": self_val, "pattern_share_overall": pattern_overall}
    print("pooled", round(tot, 2), {k: round(v, 3) for k, v in shares.items()}, "earlier pattern-borne", round(earlier_pat, 3), "self value-borne", round(self_val, 3), "pattern overall", round(pattern_overall, 3))
    predictions = {"pred_a_split_closure": closure <= CLOSURE_TOL, "pred_b_earlier_sources_are_pattern_borne": earlier_pat >= EARLIER_MIN, "pred_c_self_source_is_value_borne": self_val >= SELF_MIN, "pred_d_pattern_terms_carry_half_overall": pattern_overall >= OVERALL_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head63_pattern_value_split_result_v186", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
