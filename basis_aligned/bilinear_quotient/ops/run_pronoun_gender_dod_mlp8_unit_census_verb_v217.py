#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_detectors_lead_at_verb pred_c_top_10_units_carry_half pred_d_verb_top50_overlaps_noun_top50
"""Pronoun gender he/she DoD (v217): MLP 8 at unit grain at the VERB on the gender line. v203 (number): the same MLP-8 detectors re-fire at the verb (829 33%,
953 25%), and v216: 9.6 reads the verb as much as the noun on gender too. v164's census (reader = 9.6's he - she direction, exact per unit) at the verb of
the v71 rows ("The hero LOST the compass and so"): do the gender detectors 3152 (male) and 3943 (female) re-fire there?
PREDICTIONS (scored as written; failures preserved; priors from v203)
    pred_a_unit_closure                   sum_j T_j + bias = r . mlp(x) within 1e-3, every row
    pred_b_detectors_lead_at_verb         3152 and 3943 are both among the top-3 |pooled| units at the verb
    pred_c_top_10_units_carry_half        top-10 units carry >= 0.50 of the pooled contrast
    pred_d_verb_top50_overlaps_noun_top50 Jaccard(verb top-50, v164 noun top-50) >= 1/3
PRICE (registered maximum): 3 batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_mlp8_unit_census_verb_v217_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_mlp8_unit_census_verb_v217"
LAYER = 8
CLOSURE_TOL, TOP50_MIN, TOP200_MIN, JACCARD_MIN = 1e-3, 0.50, 0.80, 1 / 3
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_detectors_lead_at_verb": "3152, 3943 in top-3", "pred_c_top_10_units_carry_half": ">= 0.50", "pred_d_verb_top50_overlaps_noun_top50": "Jaccard >= 1/3"}
V168 = ROOT / "circuits/followups/pronoun_gender_dod_mlp8_unit_census_v164_result.json"


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "layer": LAYER, "reader_head": "9.6", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top50_min": TOP50_MIN, "top200_min": TOP200_MIN, "jaccard_min": JACCARD_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she)
    r = L.reader_directions(model, comp96, fw.directions)[6].float()
    mlp = model.transformer.h[LAYER].mlp
    Lw, Rw, Dw, bias = mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float(), mlp.Down_bias.detach().float()
    rD = (r.to(Dw.device) @ Dw)                                   # (4608,): r . Down[:, j]
    r_bias = float(r.to(bias.device) @ bias)
    forwards, per_row, closure = 0, [], 0.0
    with torch.no_grad():
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]; tokens = fw._tokens(chunk)
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                x = live + attention
                xin = F.rms_norm(x, (model.config.n_embd,))
                if layer == LAYER:
                    Lx, Rx = mlp.Left(xin), mlp.Right(xin)
                    h = (F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx)         # (B, T, 4608)
                    out = mlp(xin)
                    for i, row in enumerate(chunk):
                        s = noun_of(row) + 1; hj = h[i, s].float()
                        T = (rD * hj)                                                      # per-unit terms
                        true = float(r.to(out.device) @ out[i, s].float()); recon = float(T.sum()) + r_bias
                        closure = max(closure, abs(recon - true) / max(abs(true), 1e-6))
                        per_row.append(T.cpu())
                    break
                x = x + block.mlp(xin)
            forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    def pooled(idx):
        acc = torch.zeros(Dw.shape[1])
        for i in idx:
            row = rows[i]
            if not row.present: continue
            j = partner[(row.construction, row.group, False)]; acc += per_row[i] - per_row[j]
        return acc
    total = pooled(range(len(rows))); contrast = float(total.sum())
    order = torch.argsort(total.abs(), descending=True)
    share = lambda k: float(total[order[:k]].sum()) / contrast
    top50 = set(order[:50].tolist())
    frame_tops = {c: set(torch.argsort(pooled([i for i, r_ in enumerate(rows) if r_.construction == c]).abs(), descending=True)[:50].tolist()) for c in sorted({r_.construction for r_ in rows})}
    jac = [len(a & b) / len(a | b) for a in frame_tops.values() for b in frame_tops.values() if a is not b]
    print("contrast", round(contrast, 3), "closure", closure, "top-50 share", round(share(50), 3), "top-200", round(share(200), 3), "top-500", round(share(500), 3), "min pairwise Jaccard of frame top-50s", round(min(jac), 3))
    print("top units", [(int(j), round(float(total[j]), 3)) for j in order[:12]]); rank = {j: int((total.abs() > abs(total[j])).sum()) + 1 for j in (3152, 3943)}; print("gender units rank here", rank)
    noun_top50 = {j for j, _ in json.loads(V168.read_text())["top_units"][:50]}; jac_noun = len(top50 & noun_top50) / len(top50 | noun_top50); top3 = set(order[:3].tolist())
    print("top-10 share", round(share(10), 3), "Jaccard with noun top-50", round(jac_noun, 3), "top-3", sorted(top3))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_detectors_lead_at_verb": {3152, 3943} <= top3, "pred_c_top_10_units_carry_half": share(10) >= TOP50_MIN, "pred_d_verb_top50_overlaps_noun_top50": jac_noun >= JACCARD_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp8_unit_census_verb_result_v217", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "contrast_total": contrast,
                               "shares": {str(k): share(k) for k in (10, 20, 50, 100, 200, 500, 1000)}, "top_units": [(int(j), float(total[j])) for j in order[:200]], "frame_top50": {c: sorted(v) for c, v in frame_tops.items()},
                               "min_jaccard": min(jac), "gender_units_rank": rank, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
