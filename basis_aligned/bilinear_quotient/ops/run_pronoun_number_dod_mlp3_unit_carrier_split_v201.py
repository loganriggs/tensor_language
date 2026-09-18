#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_carrier_closure pred_b_mlp_stack_carries_050 pred_c_block5_heads_carry_little pred_d_embedding_carries_020
"""Pronoun number they/he DoD (v201): CARRIER split of MLP-3 units 3465 and 493 (v194: the two largest MLP-3 carriers of the number into MLP-5 unit 1036,
16% and 13%) at the block-3 input, via `dod_units.carrier_split` (review-23 library). Writers: embedding, attn / mlp totals of blocks 0-2, the nine heads
of block 3. Registered question: does the MLP-carried number chain bottom out in the noun embedding this far down (embedding the largest carrier), or
is it still built from MLPs 1-2?
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_carrier_closure           carrier shares sum to 1 within 1e-3 and the per-pair identity holds within relative 1e-3, both units
    pred_b_mlp_stack_carries_050     mlp:01 + mlp:02 carry >= 0.50, both units
    pred_c_block5_heads_carry_little every block-3 head carries <= 0.10, both units
    pred_d_embedding_carries_020     the embedding carries >= 0.20, both units
PRICE (registered maximum): 3 batches x 1 positional trace = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp3_unit_carrier_split_v201_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp3_unit_carrier_split_v201"
UNITS, LAYER, CLOSURE_TOL, STACK_MIN, HEAD_MAX, EMBED_MIN, BATCH = (3465, 493), 3, 1e-3, 0.50, 0.10, 0.20, 32
FORWARDS_MAX = 5
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(LAYER) for k in ("attn", "mlp")] + [f"attnhead:{LAYER:02d}:{h}" for h in range(9)]
PREDICTIONS = {"pred_a_carrier_closure": "<= 1e-3", "pred_b_mlp_stack_carries_050": ">= 0.50 x 2", "pred_c_block5_heads_carry_little": "<= 0.10 each x 2", "pred_d_embedding_carries_020": ">= 0.20 x 2"}


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
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": UNITS, "layer": LAYER, "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "stack_min": STACK_MIN, "head_max": HEAD_MAX, "embed_min": EMBED_MIN}}
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
    predictions = {"pred_a_carrier_closure": True, "pred_b_mlp_stack_carries_050": True, "pred_c_block5_heads_carry_little": True, "pred_d_embedding_carries_020": True}
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
            ref, c_, m_ = dod_units.carrier_split(aP, bP, aS, bS)
            closure = max(closure, abs(float(c_.sum()) - ref) / max(abs(ref), 1e-6)); carrier += c_.cpu(); mass += m_.cpu(); contrast += ref
        cs = {w: float(carrier[k]) / contrast for k, w in enumerate(WRITERS)}; ms = {w: float(mass[k]) / contrast for k, w in enumerate(WRITERS)}
        stack = sum(cs[f"mlp:{l:02d}"] for l in range(1, LAYER)); ranked = sorted(cs, key=lambda w: -abs(cs[w]))
        report[str(unit)] = {"contrast": contrast, "carrier_share": cs, "mass_share": ms, "carrier_sum": sum(cs.values()), "mlp_stack": stack, "ranked": ranked}
        print(unit, "contrast", round(contrast, 2), "sum", round(sum(cs.values()), 4), "mlp stack", round(stack, 3), [(w, round(cs[w], 3), "mass", round(ms[w], 3)) for w in ranked[:9]])
        predictions["pred_a_carrier_closure"] &= abs(sum(cs.values()) - 1) <= CLOSURE_TOL; predictions["pred_b_mlp_stack_carries_050"] &= stack >= STACK_MIN
        predictions["pred_c_block5_heads_carry_little"] &= all(abs(cs[f"attnhead:{LAYER:02d}:{h}"]) <= HEAD_MAX for h in range(9)); predictions["pred_d_embedding_carries_020"] &= cs["embed"] >= EMBED_MIN
    predictions["pred_a_carrier_closure"] &= closure <= CLOSURE_TOL
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp6_unit_carrier_split_result_v191", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "units": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
