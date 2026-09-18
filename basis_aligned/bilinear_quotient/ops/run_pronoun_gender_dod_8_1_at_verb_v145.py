#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_noun_position_carries_8_1_write_at_verb pred_c_8_1_write_at_verb_is_token_only pred_d_contrast_positive
"""Pronoun gender he/she DoD (v145): is head 8.1's write at the VERB position -- 20% of the gender state 9.6 reads there (v82) -- the same token-only copy
of the gendered noun that 8.1 makes at the final query for the temporal, person and correlative families (v133)?

Fold (exact, `head_source_terms_at`) at query = the verb position of the v71 fresh rows, block 8, head 1, direction d = O_{8.1}^T r where r = V_{9.6}^T v_hat
is 9.6's reader direction (v82's construction): per-source terms of r . O_{8.1} z_{8.1}(verb), split by source category (noun / other / verb itself) and
value branch (current vs token-only block-0 value), oriented male − female over aligned pairs. Evidence tag: fold, fresh rows (opened by v71).
PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                          sum of the per-source terms = the captured r . O z within relative 1e-3 per row
    pred_b_noun_position_carries_8_1_write_at_verb  the gendered-noun position carries >= 0.80 of the oriented contrast
    pred_c_8_1_write_at_verb_is_token_only       the token-only branch carries >= 0.50 of it (as 8.1's writes do at the final query)
    pred_d_contrast_positive                     mean oriented contrast > 0 (8.1 writes the male side of 9.6's reader direction)
PRICE (registered maximum): 2 batches x (capture at the verb + fold) = 4 forwards; 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_gender_dod_battery_v71 as g
import run_pronoun_dod_source_fold_positions_v81 as v81
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_8_1_at_verb_v145_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_8_1_at_verb_v145"
CLOSURE_TOL, NOUN_MIN, INH_MIN = 1e-3, 0.80, 0.50
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_noun_position_carries_8_1_write_at_verb": ">= 0.80", "pred_c_8_1_write_at_verb_is_token_only": ">= 0.50", "pred_d_contrast_positive": "> 0"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    verb_of = lambda row: next(i for i, t in enumerate(row.ids) if t in v81.VERB)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "noun_min": NOUN_MIN, "inh_min": INH_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she)
    r = L.reader_directions(model, comp96, fw.directions)[6]                          # 1152-d
    O = model.transformer.h[8].attn.c_proj.weight.detach().float()[:, 128:256]      # head 8.1 column block
    d = (O.T @ r.to(O.device)).float()                                                 # 128-d direction on 8.1's slice
    comp81 = L.Component("attn8_h1_verb", 8, "attn", (1,), "final")
    forwards, terms, closure = 0, [], 0.0
    L.positions_of = (lambda _orig: (lambda row, where: [verb_of(row)] if where == "final" and getattr(row, "_verb_query", False) else _orig(row, where)))(L.positions_of)
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        out, lamb = L.head_source_terms_at(fw, chunk, 8, verb_of, {1: d}); forwards += 1
        # closure against a direct capture of the slice at the verb
        f = L.attention_factors(fw, chunk, 8); forwards += 1
        for i, (row, entry) in enumerate(zip(chunk, out)):
            t = verb_of(row); p = fw.torch.tensor(entry[1]["pattern"])
            z = ((1 - f["lamb"]) * (p.to(f["v_cur"].device)[:, None] * f["v_cur"][i, :t + 1, 1].float()).sum(0) + f["lamb"] * (p.to(f["v1"].device)[:, None] * f["v1"][i, :t + 1, 1].float()).sum(0))
            direct = float(d.to(z.device) @ z); closure = max(closure, abs(entry[1]["total"] - direct) / max(abs(direct), 1e-6))
            terms.append(entry[1])
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    pooled = {c: {"current": 0.0, "inherited": 0.0} for c in ("noun", "verb", "other")}; contrast, n = 0.0, 0
    for i, row in enumerate(rows):
        if not row.present:
            continue
        j = partner[(row.construction, row.group, False)]; a, b = terms[i], terms[j]
        for s in range(len(a["pattern"])):
            c = "noun" if row.ids[s] in nouns else ("verb" if s == verb_of(row) else "other")
            pooled[c]["current"] += a["term_current"][s] - b["term_current"][s]; pooled[c]["inherited"] += a["term_inherited"][s] - b["term_inherited"][s]
        contrast += a["total"] - b["total"]; n += 1
    shares = {c: (v["current"] + v["inherited"]) / contrast for c, v in pooled.items()}; inh = sum(v["inherited"] for v in pooled.values()) / contrast
    print("8.1 at verb on 9.6's reader: contrast", round(contrast / n, 3), "shares", {c: round(s, 3) for c, s in shares.items()}, "inherited", round(inh, 3), "closure", closure)
    predictions = {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_noun_position_carries_8_1_write_at_verb": shares["noun"] >= NOUN_MIN, "pred_c_8_1_write_at_verb_is_token_only": inh >= INH_MIN, "pred_d_contrast_positive": contrast > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_8_1_at_verb_result_v145", "candidate_id": CANDIDATE_ID, "plan": plan, "mean_contrast": contrast / n, "shares": shares, "inherited_share": inh, "by_category_branch": pooled,
                               "closure_max": closure, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
