#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_attention_carries_050 pred_c_embedding_carries_nothing pred_d_one_attention_writer_leads
"""Aspectual has/had DoD (v282): what carries the temporal detector, MLP-7 unit 1250, at the bank? v278 / v280 / v281: 1250 is a since-context detector at the bank
(last / period / the) and the MLP-7 trio it leads is 2.2% of the has - had margin under edit. The bank tokens are identical in the two members of a pair (only the cue,
Since / By, differs), so by causality the embedding at the bank carries nothing and the since - by contrast can only arrive by attention -- scorecard rows 26-27 name
head 8.1's cue copy as the largest bank writer for the block-9 readers; at MLP 7's input the candidates are attention blocks 0-6 and the heads of block 7. Carrier split
(exact, `dod_units.carrier_split`) of u_1250's since - by contrast at the PERIOD-noun bank position (row.source_positions[1]) over the 32 aligned pairs of the v1 rows.
PREDICTIONS (scored as written; failures preserved)
    pred_a_carrier_closure           carrier shares sum to 1 within 1e-3 and the per-pair identity holds within relative 1e-3
    pred_b_attention_carries_050     attention writers (attn:00..06 + the block-7 heads) together carry >= 0.50
    pred_c_embedding_carries_nothing |embedding carrier share| <= 0.02 (causality: the period token is the same across a pair)
    pred_d_one_attention_writer_leads the largest single attention writer carries >= 0.20
PRICE (registered maximum): 2 batches x 1 positional trace = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/aspectual_dod_unit1250_bank_carrier_split_v282_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_unit1250_bank_carrier_split_v282"
UNITS, LAYER, CLOSURE_TOL, ATTN_MIN, EMBED_MAX, LEAD_MIN, BATCH = (1250,), 7, 1e-3, 0.50, 0.02, 0.20, 32
FORWARDS_MAX = 5
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(LAYER) for k in ("attn", "mlp")] + [f"attnhead:{LAYER:02d}:{h}" for h in range(9)]
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-3", "pred_b_attention_carries_050": ">= 0.50", "pred_c_embedding_carries_nothing": "<= 0.02", "pred_d_one_attention_writer_leads": ">= 0.20"}


def writers_at_block_input(tr, pos, layer):
    """Exact writer decomposition of x_layer[pos] = live_layer + attn_layer (attn_layer split by head); v16's function for any block."""
    x0 = tr[("embed", pos)]; C = {"embed": x0.clone()}
    for l in range(layer):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C: C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0
        C[f"attn:{l:02d}"] = tr[(f"attn:{l:02d}", pos)].clone(); C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    l0, l1 = tr[f"lambda0:{layer:02d}"], tr[f"lambda1:{layer:02d}"]
    for k in C: C[k] = l0 * C[k]
    C["embed"] = C["embed"] + l1 * x0
    for h in range(9): C[f"attnhead:{layer:02d}:{h}"] = tr[(f"attnhead:{layer:02d}:{h}", pos)].clone()
    return C


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    he, she = L._single(" has"), L._single(" had")
    noun_of = lambda row: row.source_positions[1]          # the period-noun bank position
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": UNITS, "layer": LAYER, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "attn_min": ATTN_MIN, "embed_max": EMBED_MAX, "lead_min": LEAD_MIN}, "position": "bank period noun"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); mlp = model.transformer.h[LAYER].mlp
    forwards, traces = 0, []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: [noun_of(rw)], upto_layer=LAYER + 1, head_write_layers=(LAYER,))); forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    n = len(WRITERS); closure, report = 0.0, {}
    predictions = {"pred_a_carrier_closure": True, "pred_b_attention_carries_050": True, "pred_c_embedding_carries_nothing": True, "pred_d_one_attention_writer_leads": True}
    Cs = [writers_at_block_input(tr, noun_of(row), LAYER) for row, tr in zip(rows, traces)]
    for unit in UNITS:
        Lrow, Rrow = mlp.Left.weight.detach().float()[unit], mlp.Right.weight.detach().float()[unit]; factors = []
        for C in Cs:
            x = sum(C.values()); rms = float(x.pow(2).mean().sqrt()); M = torch.stack([C[w] for w in WRITERS]).to(Lrow.device)
            factors.append(((M @ Lrow) / rms, (M @ Rrow) / rms))
        carrier, mass, contrast = torch.zeros(n), torch.zeros(n), 0.0
        for i, row in enumerate(rows):
            if not row.present: continue
            aP, bP = factors[i]; aS, bS = factors[partner[(row.construction, row.group, False)]]
            ref = float(aP.sum() * bP.sum() - aS.sum() * bS.sum()); da, db, ma, mb = aP - aS, bP - bS, (aP + aS) / 2, (bP + bS) / 2
            left, right = da * float(mb.sum()), db * float(ma.sum())
            closure = max(closure, abs(float(left.sum() + right.sum()) - ref) / max(abs(ref), 1e-6)); carrier += (left + right).cpu(); contrast += ref
            Tm = torch.outer(aP, bP) - torch.outer(aS, bS); mass += ((Tm + Tm.T) / 2).sum(1).cpu()
        cs = {w: float(carrier[k]) / contrast for k, w in enumerate(WRITERS)}; ms = {w: float(mass[k]) / contrast for k, w in enumerate(WRITERS)}
        stack = sum(cs[f"mlp:{l:02d}"] for l in range(1, LAYER)); ranked = sorted(cs, key=lambda w: -abs(cs[w]))
        report[str(unit)] = {"contrast": contrast, "carrier_share": cs, "mass_share": ms, "carrier_sum": sum(cs.values()), "mlp_stack": stack, "ranked": ranked}
        print(unit, "contrast", round(contrast, 2), "sum", round(sum(cs.values()), 4), "mlp stack", round(stack, 3), [(w, round(cs[w], 3), "mass", round(ms[w], 3)) for w in ranked[:9]])
        attn = {w: v for w, v in cs.items() if w.startswith("attn")}; lead = max(attn, key=lambda w: abs(attn[w]))
        print("   attention total", round(sum(attn.values()), 3), "lead", lead, round(attn[lead], 3), "embed", round(cs["embed"], 4))
        predictions["pred_a_carrier_closure"] &= abs(sum(cs.values()) - 1) <= CLOSURE_TOL; predictions["pred_b_attention_carries_050"] &= sum(attn.values()) >= ATTN_MIN
        predictions["pred_c_embedding_carries_nothing"] &= abs(cs["embed"]) <= EMBED_MAX; predictions["pred_d_one_attention_writer_leads"] &= abs(attn[lead]) >= LEAD_MIN
    predictions["pred_a_carrier_closure"] &= closure <= CLOSURE_TOL
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp6_unit_carrier_split_result_v282", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "units": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
