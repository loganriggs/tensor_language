#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_top_pair_involves_the_embedding pred_c_embed_pairs_carry_half pred_d_block6_head_pairs_carry_030
"""Pronoun number they/he DoD (v183): what the two largest MLP-6 inputs of the plural detector (v182: units 2483 and 2826, 22% and 21%) are
built from -- v181's product-level pair fold, one block earlier. u_j(noun) = (L_j . x)(R_j . x), x = x_6[noun] / rms, x_6 = live_6 + attn6 a sum of
20 writers (embedding, attn / mlp totals of blocks 0-5, the nine heads of block 6): u = sum_{a,b} (L . C_a)(R . C_b) / rms^2. Pooled plural -
singular contrast of every pair term, per unit. Question registered: are these units token detectors (embedding-built, like the gender detectors)
or contextual (built on earlier blocks / block-6 heads)?
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_pair_closure                     sum of pair terms = captured u_j(noun) within relative 1e-3, every row, both units
    pred_b_top_pair_involves_the_embedding  the largest |pooled pair term| has the embedding as a factor, both units
    pred_c_embed_pairs_carry_half           pairs with the embedding carry >= 0.50 of the contrast, both units
    pred_d_block6_head_pairs_carry_030      pairs with any block-6 head carry >= 0.30 of the contrast, both units
PRICE (registered maximum): 3 batches x 1 positional trace = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp6_unit_pair_fold_v183_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp6_unit_pair_fold_v183"
UNITS, LAYER, CLOSURE_TOL, EMBED_MIN, HEAD_MIN, BATCH = (2483, 2826), 6, 1e-3, 0.50, 0.30, 32
FORWARDS_MAX = 5
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(LAYER) for k in ("attn", "mlp")] + [f"attnhead:{LAYER:02d}:{h}" for h in range(9)]
PREDICTIONS = {"pred_a_pair_closure": "<= 1e-3", "pred_b_top_pair_involves_the_embedding": "embed factor x 2", "pred_c_embed_pairs_carry_half": ">= 0.50 x 2", "pred_d_block6_head_pairs_carry_030": ">= 0.30 x 2"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "embed_min": EMBED_MIN, "head_min": HEAD_MIN}}
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
    n = len(WRITERS); closure, report, predictions = 0.0, {}, {"pred_a_pair_closure": True, "pred_b_top_pair_involves_the_embedding": True, "pred_c_embed_pairs_carry_half": True, "pred_d_block6_head_pairs_carry_030": True}
    Cs = [writers_at_block_input(tr, noun_of(row), LAYER) for row, tr in zip(rows, traces)]
    for unit in UNITS:
        Lrow, Rrow = mlp.Left.weight.detach().float()[unit], mlp.Right.weight.detach().float()[unit]; tables = []
        for row, C in zip(rows, Cs):
            x = sum(C.values()); rms = float(x.pow(2).mean().sqrt())
            M = torch.stack([C[w] for w in WRITERS]).to(Lrow.device); T = torch.outer(M @ Lrow, M @ Rrow) / (rms * rms)
            xin = F.rms_norm(x.to(Lrow.device), (x.shape[-1],)); true = float((Lrow @ xin) * (Rrow @ xin))
            closure = max(closure, abs(float(T.sum()) - true) / max(abs(true), 1e-6)); tables.append(T.cpu())
        acc = torch.zeros(n, n)
        for i, row in enumerate(rows):
            if row.present: acc += tables[i] - tables[partner[(row.construction, row.group, False)]]
        contrast = float(acc.sum()); sym = (acc + acc.T) / 2
        shares = {f"{WRITERS[a]} x {WRITERS[b]}": float(sym[a, b] * (1 if a == b else 2)) / contrast for a in range(n) for b in range(a, n)}
        ranked = sorted(shares, key=lambda k: -abs(shares[k]))
        factor_share = lambda pred: sum(v for k, v in shares.items() if any(pred(f) for f in k.split(" x ")))
        rep = {"contrast": contrast, "top12": [(k, shares[k]) for k in ranked[:12]], "embed_pairs": factor_share(lambda f: f == "embed"), "block6_head_pairs": factor_share(lambda f: f.startswith("attnhead")),
               "per_writer": {w: factor_share(lambda f, w=w: f == w) for w in WRITERS}, "shares": shares}
        report[str(unit)] = rep
        print(unit, "contrast", round(contrast, 2), "top", [(k, round(v, 3)) for k, v in rep["top12"][:8]]); print("   embed pairs", round(rep["embed_pairs"], 3), "block-6 head pairs", round(rep["block6_head_pairs"], 3), "writers", {w: round(v, 2) for w, v in sorted(rep["per_writer"].items(), key=lambda kv: -abs(kv[1]))[:8]})
        predictions["pred_b_top_pair_involves_the_embedding"] &= "embed" in ranked[0].split(" x "); predictions["pred_c_embed_pairs_carry_half"] &= rep["embed_pairs"] >= EMBED_MIN; predictions["pred_d_block6_head_pairs_carry_030"] &= rep["block6_head_pairs"] >= HEAD_MIN
    predictions["pred_a_pair_closure"] = closure <= CLOSURE_TOL
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp6_unit_pair_fold_result_v183", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "units": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
