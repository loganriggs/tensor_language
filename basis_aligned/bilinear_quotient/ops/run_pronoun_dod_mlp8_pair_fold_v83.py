#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_bilinear_closure pred_b_gender_verb_top_pair_involves_head_8_1 pred_c_gender_verb_pairs_with_8_1_carry_half pred_d_number_verb_pairs_with_mlp6_or_mlp7_carry_half pred_e_noun_embedding_self_term_at_least_020
"""Pronoun gender + number DoD (v83): FOLD MLP 8's write at the noun and the verb into writer-pair terms (better_circuits §3.5/§3.9).

Lane: Claude circuit lane. Parent: v82 (the states head 9.6 reads at the noun and the verb are MLP-written, MLP 8 first: gender
verb 0.51, number verb 0.54, noun 0.27 / 0.50). MLP 8 is bilinear, so r . mlp8(s) expands EXACTLY into writer-pair terms
T[a,b] = sum_j (Down^T r)_j (L C_a)_j (R C_b)_j / rho^2 over the 26 writers of x_8[s] (embedding, attn:00-07, mlp:00-07, the nine
heads of block 8), plus the bias (cancels in the oriented contrast); r = V_{9.6}^T v_hat (weight-only reader direction, v16's
construction). Reported per line and position: the symmetrized pair shares of the oriented contrast. Evidence tag: fold, fresh rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_bilinear_closure                              all pairs + bias = r . mlp8(s) from the trace within relative 1e-3, every row / position / line
    pred_b_gender_verb_top_pair_involves_head_8_1        gender, verb: the largest |pair share| has attnhead:08:1 as a factor. Prior: unsure
                                                         (8.1 writes 20% of the verb feature directly; MLP 8 may amplify the same read).
    pred_c_gender_verb_pairs_with_8_1_carry_half         gender, verb: pairs with 8.1 as a factor carry >= 0.50. Prior: unsure.
    pred_d_number_verb_pairs_with_mlp6_or_mlp7_carry_half  number, verb: pairs with mlp:06 or mlp:07 as a factor carry >= 0.50 (the number
                                                         resolved by the earlier MLPs feeds MLP 8). Prior: unsure.
    pred_e_noun_embedding_self_term_at_least_020         noun, both lines: the embed x embed term carries >= 0.20 (the token's own feature
                                                         squared). Prior: unsure.

PRICE (registered maximum): gender 2 + number 3 batches x 1 positional trace = 5 forwards; all pair terms on captured vectors;
0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_mlp8_pair_fold_v16 as v16
import run_pronoun_dod_source_fold_v80 as v80
import run_pronoun_dod_verb_writer_fold_v82 as v82
import dod_battery

OUT = v80.ROOT / "circuits/followups/pronoun_dod_mlp8_pair_fold_v83_result.json"
CANDIDATE_ID = "pronoun.gender_and_number.dod_mlp8_pair_fold_v83"
CLOSURE_TOL, HALF, SELF_MIN = 1e-3, 0.50, 0.20
FORWARDS_MAX = 8
HEAD = 6
WRITERS = v16.WRITERS
PREDICTIONS = {"pred_a_bilinear_closure": "<= 1e-3", "pred_b_gender_verb_top_pair_involves_head_8_1": "8.1 factor", "pred_c_gender_verb_pairs_with_8_1_carry_half": ">= 0.50",
               "pred_d_number_verb_pairs_with_mlp6_or_mlp7_carry_half": ">= 0.50", "pred_e_noun_embedding_self_term_at_least_020": ">= 0.20 x 2"}


def fold_line(fw, backend, rows, pos_id, neg_id, nouns, name):
    torch, model = backend.torch, backend.model
    comp = next(c for c in dod_battery.LineSpec(name, "x", rows, pos_id, neg_id, ((9, HEAD),)).set_components())
    fw.directions = L.readout_directions(model, (comp,), pos_id, neg_id)
    r = L.reader_directions(model, comp, fw.directions)[HEAD]
    mlp8 = model.transformer.h[8].mlp
    Lw, Rw, Dw = mlp8.Left.weight.detach().float(), mlp8.Right.weight.detach().float(), mlp8.Down.weight.detach().float()
    bias = mlp8.Down_bias.detach().float(); r = r.to(Dw.device); u = Dw.T @ r
    forwards, traces = 0, []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: list(v82.positions_of(rw, nouns).values()), upto_layer=9, head_write_layers=(8,))); forwards += 1
    closure_max, tables, n = 0.0, [], len(WRITERS)
    for row, tr in zip(rows, traces):
        per = {}
        for label, pos in v82.positions_of(row, nouns).items():
            C = v16.writers_at_block8_input(tr, pos)
            x8 = sum(C.values()); rho = float(x8.pow(2).mean().sqrt())
            M = torch.stack([C[w] for w in WRITERS]).to(Dw.device)
            LA, RB = M @ Lw.T, M @ Rw.T
            T = ((LA * u) @ RB.T) / (rho * rho)
            total = float(T.sum() + r @ bias); true = float(r @ tr[("mlp:08", pos)].to(Dw.device))
            closure_max = max(closure_max, abs(total - true) / max(abs(true), 1e-6))
            per[label] = T.cpu()
        tables.append(per)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for label in ("noun", "verb"):
        acc = torch.zeros(n, n)
        for i, row in enumerate(rows):
            if not row.present:
                continue
            j = partner[(row.construction, row.group, False)]
            acc += tables[i][label] - tables[j][label]
        contrast = float(acc.sum()); sym = (acc + acc.T) / 2
        shares = {}
        for a in range(n):
            for b in range(a, n):
                shares[f"{WRITERS[a]} x {WRITERS[b]}"] = float(sym[a, b] * (1 if a == b else 2)) / contrast
        ranked = sorted(shares, key=lambda k: -abs(shares[k]))
        factor_share = lambda names: sum(v for k, v in shares.items() if any(f in k.split(" x ") for f in names))
        report[label] = {"contrast": contrast, "top10": [(k, shares[k]) for k in ranked[:10]], "top_pair": ranked[0],
                         "pairs_with_8_1_share": factor_share(["attnhead:08:1"]), "pairs_with_mlp6_or_mlp7_share": factor_share(["mlp:06", "mlp:07"]),
                         "embed_self_share": shares["embed x embed"], "linear_in_embed_share": factor_share(["embed"])}
        print(name, label, "contrast", round(contrast, 2), "top", [(k, round(v, 3)) for k, v in report[label]["top10"][:6]], "8.1", round(report[label]["pairs_with_8_1_share"], 3), "mlp6/7", round(report[label]["pairs_with_mlp6_or_mlp7_share"], 3), "embed^2", round(report[label]["embed_self_share"], 3))
    return {"closure_max": closure_max, "positions": report}, forwards


def main() -> None:
    LN = v80.lines()
    plan = {"candidate_id": CANDIDATE_ID, "rows": {k: len(v[0]) for k, v in LN.items()}, "rows_sha256": {k: L.rows_sha256(v[0]) for k, v in LN.items()}, "head": "9.6", "writers": WRITERS,
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "half": HALF, "self_min": SELF_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards, result = 0, {}
    for name, (rows, pos, neg, heads, nouns, objs) in LN.items():
        result[name], n = fold_line(fw, backend, rows, pos, neg, nouns, name); forwards += n
    GV, NV = result["gender"]["positions"]["verb"], result["number"]["positions"]["verb"]
    predictions = {"pred_a_bilinear_closure": max(v["closure_max"] for v in result.values()) <= CLOSURE_TOL,
                   "pred_b_gender_verb_top_pair_involves_head_8_1": "attnhead:08:1" in GV["top_pair"].split(" x "),
                   "pred_c_gender_verb_pairs_with_8_1_carry_half": GV["pairs_with_8_1_share"] >= HALF,
                   "pred_d_number_verb_pairs_with_mlp6_or_mlp7_carry_half": NV["pairs_with_mlp6_or_mlp7_share"] >= HALF,
                   "pred_e_noun_embedding_self_term_at_least_020": all(v["positions"]["noun"]["embed_self_share"] >= SELF_MIN for v in result.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_dod_mlp8_pair_fold_result_v83", "candidate_id": CANDIDATE_ID, "plan": plan, "lines": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
