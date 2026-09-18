#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_margin_positive pred_b_8_1_inert_for_829 pred_c_8_1_inert_for_the_margin pred_d_no_block8_head_moves_margin_002 pred_e_8_8_largest_on_829
"""Pronoun number they/he DoD (v223): the last cell of the table -- head 8.1 (and every block-8 head) zeroed at the VERB on the NUMBER line. v221 (gender): zeroing
8.1 at the verb costs 29% of 3152's verb contrast and 3.5% of he - she; v206 (number): zeroing 4.5 at the verb costs 7.7% of they - he; v219 (gender): 4.5 at the
verb is inert. If number travels as a computed feature (v188 / v204: 8.1 carries -1% / 0.6% of 829), 8.1 at the verb should be inert for they - he. Same
runner as v221 on the v76 rows, reading u_829 at the verb and the they - he margin; every block-8 head zeroed at the verb, one at a time.
PREDICTIONS (scored as written; failures preserved; priors from v188 / v204)
    pred_a_native_margin_positive          the native pooled they - he margin is > 0 (v206: 196.62)
    pred_b_8_1_inert_for_829               |fractional change of u_829's verb contrast| under the 8.1 edit <= 0.05
    pred_c_8_1_inert_for_the_margin        |fractional change of the they - he margin| under the 8.1 edit <= 0.01
    pred_d_no_block8_head_moves_margin_002 no single block-8 head zeroed at the verb moves the margin by more than 0.02 (the verb-site route on number is 4.5, not block 8)
    pred_e_8_8_largest_on_829              head 8.8 has the largest |fractional change| of u_829's verb contrast among block-8 heads (v188: 8.8 the largest block-8 carrier at the noun, 3.9%). Prior: unsure.
PRICE (registered maximum): 3 batches x (baseline + 9 single-head edits) = 30 forwards; 0 backwards; 0 fits. Bar <= 33.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_block8_heads_verb_edit_v223_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_block8_heads_verb_edit_v223"
BLOCK, HEAD, U829, INERT_829, INERT_MARGIN, ANY_MAX, BATCH = 8, 1, 829, 0.05, 0.01, 0.02, 32
FORWARDS_MAX = 33
PREDICTIONS = {"pred_a_native_margin_positive": "> 0", "pred_b_8_1_inert_for_829": "<= 0.05", "pred_c_8_1_inert_for_the_margin": "<= 0.01", "pred_d_no_block8_head_moves_margin_002": "<= 0.02 all", "pred_e_8_8_largest_on_829": "8.8 largest"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns); verb_of = lambda row: noun_of(row) + 1
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "block": BLOCK, "head": HEAD, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"inert_829": INERT_829, "inert_margin": INERT_MARGIN, "any_max": ANY_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); blocks = model.transformer.h
    forwards = 0

    def run(chunk, head):
        tokens = fw._tokens(chunk); pos = [verb_of(r) for r in chunk]; idx = torch.arange(len(chunk)); out = {}
        handle = None
        if head is not None:
            def pre(_m, args):
                z = args[0].clone()
                for i, p in enumerate(pos): z[i, p, head * L.HEAD_DIM:(head + 1) * L.HEAD_DIM] = 0
                return (z,)
            handle = blocks[BLOCK].attn.c_proj.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0
                    attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                    x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l == 8: out["u829"] = dod_units.hidden(model, block.mlp, xin)[idx, pos, U829].float().cpu()
                    x = x + block.mlp(xin)
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
                fin = torch.tensor([r.final for r in chunk]); lg = logits[idx, fin].float(); out["margin"] = (lg[:, he] - lg[:, she]).cpu()
        finally:
            if handle is not None: handle.remove()
        return out

    conditions = [("baseline", None)] + [(f"head{h}", h) for h in range(9)]
    per_row = {name: [] for name, _ in conditions}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        for name, h in conditions:
            o = run(chunk, h); forwards += 1
            per_row[name].extend({"u829": float(o["u829"][i]), "margin": float(o["margin"][i])} for i in range(len(chunk)))
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    pooled = lambda name, key: sum(per_row[name][i][key] - per_row[name][partner[(row.construction, row.group, False)]][key] for i, row in enumerate(rows) if row.present)
    base = {k: pooled("baseline", k) for k in ("u829", "margin")}
    frac = {h: {k: (abs(pooled(f"head{h}", k)) - abs(base[k])) / abs(base[k]) for k in base} for h in range(9)}
    others = [h for h in range(9) if h != HEAD]; me = frac[HEAD]
    report = {"baseline": base, "fractional_change_by_head": frac}
    print("baseline", {k: round(v, 2) for k, v in base.items()}); print("8.1", {k: round(v, 4) for k, v in me.items()}); print("others", {h: {k: round(v, 4) for k, v in frac[h].items()} for h in others})
    largest_829 = max(range(9), key=lambda h: abs(frac[h]["u829"]))
    predictions = {"pred_a_native_margin_positive": base["margin"] > 0, "pred_b_8_1_inert_for_829": abs(me["u829"]) <= INERT_829, "pred_c_8_1_inert_for_the_margin": abs(me["margin"]) <= INERT_MARGIN,
                   "pred_d_no_block8_head_moves_margin_002": all(abs(frac[h]["margin"]) <= ANY_MAX for h in range(9)), "pred_e_8_8_largest_on_829": largest_829 == 8}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head45_verb_edit_result_v223", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
