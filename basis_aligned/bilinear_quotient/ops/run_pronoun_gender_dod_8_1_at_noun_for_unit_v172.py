#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_self_position_carries_8_1_write pred_c_8_1_write_is_token_only
"""Pronoun gender he/she DoD (v172): head 8.1's write AT THE NOUN POSITION, as read by the two factors of MLP-8 unit 3152 (v171: 21-23% of each factor).

Fold (exact, `head_source_terms_at`) at query = noun, block 8, head 1, along d_L = O_{8.1}^T L_3152 and d_R = O_{8.1}^T R_3152: per-source terms split by
source (the noun itself = self, others) and value branch (token-only block-0 value vs contextual). If 8.1 at the noun attends to the noun and copies
its block-0 value, this input of the detector is the token again (v145's result at the verb, now at the noun).
PREDICTIONS (scored as written; failures preserved): pred_a closure <= 1e-3; pred_b the noun (self) position carries >= 0.80 of the oriented contrast for
both factors; pred_c the token-only branch carries >= 0.50 for both.
PRICE (registered maximum): 2 batches x (capture + fold) = 4 forwards; bar <= 6.
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
OUT = ROOT / "circuits/followups/pronoun_gender_dod_8_1_at_noun_for_unit_v172_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_8_1_at_noun_for_unit_v172"
UNIT, CLOSURE_TOL, SELF_MIN, INH_MIN = 3152, 1e-3, 0.80, 0.50
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_self_position_carries_8_1_write": ">= 0.80 x 2", "pred_c_8_1_write_is_token_only": ">= 0.50 x 2"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "self_min": SELF_MIN, "inh_min": INH_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    mlp = model.transformer.h[8].mlp; O = model.transformer.h[8].attn.c_proj.weight.detach().float()[:, 128:256]
    dirs = {"L": O.T @ mlp.Left.weight.detach().float()[UNIT], "R": O.T @ mlp.Right.weight.detach().float()[UNIT]}
    forwards, terms, closure = 0, {"L": [], "R": []}, 0.0
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        f = L.attention_factors(fw, chunk, 8); forwards += 1
        for name, d in dirs.items():
            out, lamb = L.head_source_terms_at(fw, chunk, 8, noun_of, {1: d}); forwards += 1 if name == "L" else 0
            for i, (row, entry) in enumerate(zip(chunk, out)):
                t = noun_of(row); p = fw.torch.tensor(entry[1]["pattern"])
                z = ((1 - f["lamb"]) * (p.to(f["v_cur"].device)[:, None] * f["v_cur"][i, :t + 1, 1].float()).sum(0) + f["lamb"] * (p.to(f["v1"].device)[:, None] * f["v1"][i, :t + 1, 1].float()).sum(0))
                direct = float(d.to(z.device) @ z); closure = max(closure, abs(entry[1]["total"] - direct) / max(abs(direct), 1e-6)); terms[name].append(entry[1])
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    report = {}
    for name in ("L", "R"):
        pooled = {c: {"current": 0.0, "inherited": 0.0} for c in ("self", "other")}; contrast = 0.0
        for i, row in enumerate(rows):
            if not row.present: continue
            j = partner[(row.construction, row.group, False)]; a, b = terms[name][i], terms[name][j]
            for s in range(len(a["pattern"])):
                c = "self" if s == noun_of(row) else "other"
                pooled[c]["current"] += a["term_current"][s] - b["term_current"][s]; pooled[c]["inherited"] += a["term_inherited"][s] - b["term_inherited"][s]
            contrast += a["total"] - b["total"]
        shares = {c: (v["current"] + v["inherited"]) / contrast for c, v in pooled.items()}; inh = sum(v["inherited"] for v in pooled.values()) / contrast
        report[name] = {"contrast": contrast, "shares": shares, "inherited_share": inh, "by_category_branch": pooled}
        print(name, "contrast", round(contrast, 3), "self", round(shares["self"], 3), "other", round(shares["other"], 3), "token-only", round(inh, 3))
    predictions = {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_self_position_carries_8_1_write": all(r["shares"]["self"] >= SELF_MIN for r in report.values()), "pred_c_8_1_write_is_token_only": all(r["inherited_share"] >= INH_MIN for r in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_8_1_at_noun_for_unit_result_v172", "candidate_id": CANDIDATE_ID, "plan": plan, "closure_max": closure, "factors": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
