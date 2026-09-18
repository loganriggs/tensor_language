#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_margin_positive pred_b_3152_verb_contrast_drops_015 pred_c_beats_every_other_block8_head pred_d_margin_drops_and_beats_others pred_e_shrinks_not_grows
"""Pronoun gender he/she DoD (v221): EDIT deciding v220 -- head 8.1's copy of the gendered token to the VERB feeds the male detector's re-firing there (34% of
3152's carriage at the verb) and, through it, the he - she readout. Zero head 8.1's slice at the verb position only, in a plain forward, and read the pooled
male - female contrast of u_3152 at the verb (block 8, after the edit) and the he - she margin at the final token. Matched null: the same edit for each of the
other eight heads of block 8. (v206 on number: zeroing 4.5 at the verb cost 7.7% of the margin; v219 on gender: zeroing 4.5 there cost nothing.)
PREDICTIONS (scored as written; failures preserved; priors from v220)
    pred_a_native_margin_positive          the native pooled he - she margin is > 0 (instrument sanity; the gender panel's pooled margin was 103.5 in v219)
    pred_b_3152_verb_contrast_drops_015    the 8.1 edit removes >= 0.15 of |u_3152 verb contrast| (carriage 0.34; edits run 2-4x below shares)
    pred_c_beats_every_other_block8_head   its |fractional change of the 3152 verb contrast| exceeds every other block-8 head's
    pred_d_margin_drops_and_beats_others   the he - she margin drops (> 0) and by more than any other block-8 head's edit
    pred_e_shrinks_not_grows               the 3152 verb contrast shrinks (not grows)
PRICE (registered maximum): 2 batches x (baseline + 9 single-head edits) = 20 forwards; 0 backwards; 0 fits. Bar <= 22.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_head81_verb_edit_v221_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_head81_verb_edit_v221"
BLOCK, HEAD, U829, DROP_MIN, BATCH = 8, 1, 3152, 0.15, 32
FORWARDS_MAX = 22
PREDICTIONS = {"pred_a_native_margin_positive": "> 0", "pred_b_3152_verb_contrast_drops_015": ">= 0.15", "pred_c_beats_every_other_block8_head": "> all 8", "pred_d_margin_drops_and_beats_others": "> 0 and > all 8", "pred_e_shrinks_not_grows": "shrink"}


def main() -> None:
    rows, he, she = g.build()
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns); verb_of = lambda row: noun_of(row) + 1
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "block": BLOCK, "head": HEAD, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"drop_min": DROP_MIN}}
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
    predictions = {"pred_a_native_margin_positive": base["margin"] > 0, "pred_b_3152_verb_contrast_drops_015": -me["u829"] >= DROP_MIN,
                   "pred_c_beats_every_other_block8_head": all(abs(me["u829"]) > abs(frac[h]["u829"]) for h in others), "pred_d_margin_drops_and_beats_others": me["margin"] < 0 and all(-me["margin"] > -frac[h]["margin"] for h in others),
                   "pred_e_shrinks_not_grows": me["u829"] < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head45_verb_edit_result_v221", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
