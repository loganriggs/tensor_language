#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_gender_contextual_heads_largest_non_noun_source_is_verb pred_c_gender_contextual_heads_read_noun_through_current_branch pred_d_head_10_1_noun_term_is_token_only pred_e_number_noun_plus_verb_carry_060
"""Pronoun gender + number DoD (v81): the v80 source fold at POSITION grain (splits v80's pooled "other" category).

Lane: Claude circuit lane. Parent: v80 (gender: 10.1 token-only noun reader, 15.1 half, 9.6 / 12.4 contextual with ~50% from
"other" positions; number: contextual throughout, noun share 0.34). Same rows, same exact fold (`run_pronoun_dod_source_fold_v80
.fold_line` with a finer category function): prefix ("Because"), det_noun (the determiner before the noun), noun, verb (lost /
wanted / bought), det_object, object, conj (" and"), final. Evidence tag: fold, fresh rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                                            <= 1e-3 on both lines
    pred_b_gender_contextual_heads_largest_non_noun_source_is_verb  for 9.6 and 12.4 on the gender line the largest |share| among
                                                                    non-noun categories is the verb. Prior: unsure.
    pred_c_gender_contextual_heads_read_noun_through_current_branch for 9.6 and 12.4, the noun-position term is >= 0.85 current
                                                                    branch (they read a contextual state AT the noun, not its token)
    pred_d_head_10_1_noun_term_is_token_only                        10.1's noun term is >= 0.70 inherited (v80 replay at position grain)
    pred_e_number_noun_plus_verb_carry_060                          number line: noun + verb >= 0.60 of the contrast for every head.
                                                                    Prior: unsure (subject number may be resolved at the verb).

PRICE (registered maximum): as v80, 25 forwards; 0 backwards; 0 fits. Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_dod_source_fold_v80 as v80

OUT = v80.ROOT / "circuits/followups/pronoun_dod_source_fold_positions_v81_result.json"
CANDIDATE_ID = "pronoun.gender_and_number.dod_source_fold_positions_v81"
CATEGORIES = ("prefix", "det_noun", "noun", "verb", "det_object", "object", "conj", "final")
CURRENT_MIN, INHERITED_MIN, NOUN_VERB_MIN = 0.85, 0.70, 0.60
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_gender_contextual_heads_largest_non_noun_source_is_verb": "verb x 2",
               "pred_c_gender_contextual_heads_read_noun_through_current_branch": ">= 0.85 current", "pred_d_head_10_1_noun_term_is_token_only": ">= 0.70 inherited",
               "pred_e_number_noun_plus_verb_carry_060": ">= 0.60 x 4"}
DET = {L._single(" the"), L._single("The"), L._single(" The")}
VERB = {L._single(" lost"), L._single(" wanted"), L._single(" bought")}
CONJ = {L._single(" and")}
PREFIX = {L._single("Because")}


def make_category(nouns, objs):
    def category(row, s, _n=None, _o=None):
        if s == row.final:
            return "final"
        tid = row.ids[s]
        if tid in nouns:
            return "noun"
        if tid in objs:
            return "object"
        if tid in VERB:
            return "verb"
        if tid in CONJ:
            return "conj"
        if tid in PREFIX:
            return "prefix"
        if tid in DET:
            noun_pos = next(i for i, t in enumerate(row.ids) if t in nouns)
            return "det_noun" if s < noun_pos else "det_object"
        raise SystemExit(f"uncategorized token {tid} at {s} in {row.text!r}")
    return category


def main() -> None:
    LN = v80.lines()
    plan = {"candidate_id": CANDIDATE_ID, "rows": {k: len(v[0]) for k, v in LN.items()}, "rows_sha256": {k: L.rows_sha256(v[0]) for k, v in LN.items()}, "heads": {k: list(v[3]) for k, v in LN.items()},
            "categories": CATEGORIES, "forwards_max": v80.FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": {"closure_tol": v80.CLOSURE_TOL, "current_min": CURRENT_MIN, "inherited_min": INHERITED_MIN, "noun_verb_min": NOUN_VERB_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    v80.CATEGORIES = CATEGORIES
    forwards, result = 0, {}
    for name, (rows, pos, neg, heads, nouns, objs) in LN.items():
        v80.category = make_category(nouns, objs)
        result[name], n = v80.fold_line(fw, rows, pos, neg, heads, nouns, objs, name); forwards += n
    G, N = result["gender"]["heads"], result["number"]["heads"]

    def branch_share(h, cat):
        b = h["by_category_branch"][cat]; tot = b["current"] + b["inherited"]
        return (b["current"] / tot if tot else None, b["inherited"] / tot if tot else None)

    ctx = ("9.6", "12.4")
    largest_non_noun = {k: max((c for c in CATEGORIES if c != "noun"), key=lambda c: abs(G[k]["shares"][c])) for k in ctx}
    noun_current = {k: branch_share(G[k], "noun")[0] for k in ctx}
    inh_10_1 = branch_share(G["10.1"], "noun")[1]
    noun_verb = {k: N[k]["shares"]["noun"] + N[k]["shares"]["verb"] for k in N}
    predictions = {"pred_a_fold_closure": max(result["gender"]["closure_max"], result["number"]["closure_max"]) <= v80.CLOSURE_TOL,
                   "pred_b_gender_contextual_heads_largest_non_noun_source_is_verb": all(v == "verb" for v in largest_non_noun.values()),
                   "pred_c_gender_contextual_heads_read_noun_through_current_branch": all(v is not None and v >= CURRENT_MIN for v in noun_current.values()),
                   "pred_d_head_10_1_noun_term_is_token_only": inh_10_1 is not None and inh_10_1 >= INHERITED_MIN,
                   "pred_e_number_noun_plus_verb_carry_060": all(v >= NOUN_VERB_MIN for v in noun_verb.values())}
    if forwards > v80.FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {v80.FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_dod_source_fold_positions_result_v81", "candidate_id": CANDIDATE_ID, "plan": plan, "lines": result,
                               "derived": {"largest_non_noun": largest_non_noun, "noun_current_share": noun_current, "inherited_10_1_noun": inh_10_1, "number_noun_plus_verb": noun_verb},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "largest_non_noun": largest_non_noun, "noun_current": noun_current, "inh_10_1": inh_10_1, "noun_verb": noun_verb, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
