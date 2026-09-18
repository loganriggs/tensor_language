#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_verb_state_is_mostly_attention_written pred_c_top_three_heads_carry_070_of_verb_attention pred_d_largest_verb_writer_shared_across_lines pred_e_noun_state_is_mostly_embedding_and_early_mlp
"""Pronoun gender + number DoD (v82): WRITER FOLD of the states head 9.6 reads at the noun and the verb (better_circuits §3.8).

Lane: Claude circuit lane. Parents: v80/v81 (9.6 reads the subject noun and the verb equally, through its contextual value
branch, on both lines). Its contribution is exact: c ∋ sum_s p(t,s) (1 - lamb) r . rms_norm(live_9[s]) with the weight-only reader
direction r = V_{9.6}^T v_hat (v13's construction), and live_9[s] is an exact lambda-weighted sum of every earlier writer: the
embedding, each head of blocks 0-8 (c_proj column blocks; the c_proj bias kept as its own writer per block), and mlp:00-08.
This run attributes the oriented reader-projected contrast at the NOUN and at the VERB to those writers, on the fresh rows of
both lines. The verb-position share is a relay: something copied the subject's gender / number from the noun to the verb.
Evidence tag: fold (no intervention), fresh rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_writer_closure                            reconstruction of live_9 at both positions within relative 1e-3 on every row
    pred_b_verb_state_is_mostly_attention_written    verb: the head writers (all blocks 0-8) carry >= 0.50 of the contrast, both lines
    pred_c_top_three_heads_carry_070_of_verb_attention  verb: three heads carry >= 0.70 of the head part, both lines. Prior: unsure.
    pred_d_largest_verb_writer_shared_across_lines   the largest head writer at the verb is the same head on both lines. Prior: unsure.
    pred_e_noun_state_is_mostly_embedding_and_early_mlp  noun: embedding + mlp:00-02 carry >= 0.50 of the contrast, both lines. Prior: unsure.

PRICE (registered maximum): gender 2 batches + number 3 batches, each (positional trace + block-9 fold) = 10 forwards; 0
backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_dod_source_fold_v80 as v80
import run_pronoun_dod_source_fold_positions_v81 as v81
import dod_battery

OUT = v80.ROOT / "circuits/followups/pronoun_dod_verb_writer_fold_v82_result.json"
CANDIDATE_ID = "pronoun.gender_and_number.dod_verb_writer_fold_v82"
CLOSURE_TOL, ATTN_MIN, TOP3_MIN, NOUN_MIN = 1e-3, 0.50, 0.70, 0.50
FORWARDS_MAX = 12
HEAD = 6
PREDICTIONS = {"pred_a_writer_closure": "<= 1e-3", "pred_b_verb_state_is_mostly_attention_written": ">= 0.50 x 2", "pred_c_top_three_heads_carry_070_of_verb_attention": ">= 0.70 x 2",
               "pred_d_largest_verb_writer_shared_across_lines": "same head", "pred_e_noun_state_is_mostly_embedding_and_early_mlp": ">= 0.50 x 2"}
WRITERS = ["embed"] + [f"attnhead:{l:02d}:{h}" for l in range(9) for h in range(9)] + [f"attnbias:{l:02d}" for l in range(9)] + [f"mlp:{l:02d}" for l in range(9)]


def contributions(tr, pos):
    x0 = tr[("embed", pos)]
    C = {"embed": x0.clone()}
    for l in range(9):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C:
            C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0
        heads = {f"attnhead:{l:02d}:{h}": tr[(f"attnhead:{l:02d}:{h}", pos)].clone() for h in range(9)}
        C.update(heads)
        C[f"attnbias:{l:02d}"] = tr[(f"attn:{l:02d}", pos)] - sum(heads.values())
        C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    return C


def positions_of(row, nouns):
    noun = next(i for i, t in enumerate(row.ids) if t in nouns)
    verb = next(i for i, t in enumerate(row.ids) if t in v81.VERB)
    return {"noun": noun, "verb": verb}


def fold_line(fw, model, rows, pos_id, neg_id, nouns, name):
    comp = next(c for c in dod_battery.LineSpec(name, "x", rows, pos_id, neg_id, ((9, HEAD),)).set_components())
    fw.directions = L.readout_directions(model, (comp,), pos_id, neg_id)
    r = L.reader_directions(model, comp, fw.directions)[HEAD]
    block9 = model.transformer.h[9]; l0_9, l1_9 = float(block9.lambdas[0]), float(block9.lambdas[1])
    forwards, traces, patterns = 0, [], []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: list(positions_of(rw, nouns).values()), upto_layer=9, head_write_layers=range(9))); forwards += 1
        out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
        patterns.extend(out)
    closure_max, per_row = 0.0, []
    for row, tr, pat in zip(rows, traces, patterns):
        entry = {}
        for label, pos in positions_of(row, nouns).items():
            C = contributions(tr, pos)
            C = {k: l0_9 * v for k, v in C.items()}; C["embed"] = C["embed"] + l1_9 * tr[("embed", pos)]
            true = tr[("live", pos)]; recon = sum(C.values())
            closure_max = max(closure_max, float((recon - true).norm() / true.norm()))
            rms = float(true.pow(2).mean().sqrt()); scale = pat[HEAD]["pattern"][pos] * (1 - lamb) / rms
            rr = r.to(true.device)
            entry[label] = {w: float(rr @ C[w]) * scale for w in WRITERS}
        per_row.append(entry)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for label in ("noun", "verb"):
        totals = {w: 0.0 for w in WRITERS}; contrast = 0.0
        for i, row in enumerate(rows):
            if not row.present:
                continue
            j = partner[(row.construction, row.group, False)]
            a, b = per_row[i][label], per_row[j][label]
            for w in WRITERS:
                totals[w] += a[w] - b[w]
            contrast += sum(a.values()) - sum(b.values())
        shares = {w: totals[w] / contrast for w in WRITERS}
        heads = {w: s for w, s in shares.items() if w.startswith("attnhead")}
        head_part = sum(heads.values()); ranked = sorted(heads, key=lambda w: -abs(heads[w]))
        report[label] = {"contrast_total": contrast, "shares": shares, "head_part": head_part, "mlp_part": sum(s for w, s in shares.items() if w.startswith("mlp")),
                         "embed_share": shares["embed"], "embed_plus_early_mlp": shares["embed"] + sum(shares[f"mlp:{l:02d}"] for l in range(3)),
                         "top_heads": [(w, heads[w]) for w in ranked[:6]], "top3_head_share_of_head_part": sum(heads[w] for w in ranked[:3]) / head_part if head_part else None}
        print(name, label, "contrast", round(contrast, 3), "heads", round(head_part, 3), "mlp", round(report[label]["mlp_part"], 3), "embed", round(shares["embed"], 3), "top", [(w, round(s, 3)) for w, s in report[label]["top_heads"][:4]])
    return {"closure_max": closure_max, "positions": report}, forwards


def main() -> None:
    LN = v80.lines()
    plan = {"candidate_id": CANDIDATE_ID, "rows": {k: len(v[0]) for k, v in LN.items()}, "rows_sha256": {k: L.rows_sha256(v[0]) for k, v in LN.items()}, "head": "9.6", "writers": len(WRITERS),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "attn_min": ATTN_MIN, "top3_min": TOP3_MIN, "noun_min": NOUN_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    forwards, result = 0, {}
    for name, (rows, pos, neg, heads, nouns, objs) in LN.items():
        result[name], n = fold_line(fw, model, rows, pos, neg, nouns, name); forwards += n
    V = {k: v["positions"]["verb"] for k, v in result.items()}; Nn = {k: v["positions"]["noun"] for k, v in result.items()}
    predictions = {"pred_a_writer_closure": max(v["closure_max"] for v in result.values()) <= CLOSURE_TOL,
                   "pred_b_verb_state_is_mostly_attention_written": all(v["head_part"] >= ATTN_MIN for v in V.values()),
                   "pred_c_top_three_heads_carry_070_of_verb_attention": all(v["top3_head_share_of_head_part"] is not None and v["top3_head_share_of_head_part"] >= TOP3_MIN for v in V.values()),
                   "pred_d_largest_verb_writer_shared_across_lines": len({v["top_heads"][0][0] for v in V.values()}) == 1,
                   "pred_e_noun_state_is_mostly_embedding_and_early_mlp": all(v["embed_plus_early_mlp"] >= NOUN_MIN for v in Nn.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_dod_verb_writer_fold_result_v82", "candidate_id": CANDIDATE_ID, "plan": plan, "lines": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
