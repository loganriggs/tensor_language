#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rows_load pred_b_all_chain_units_separate_at_the_cue pred_c_signs_match_panel pred_d_separation_grows_or_holds_up_the_chain pred_e_none_at_the_final
"""The unit chain on natural text (v369). v368 named the noun-state chain at unit grain: 3465 / 493 (MLP 3) -> 1036 (MLP 5) -> 2483 (MLP 6) -> 1779 (MLP 7)
-> 829 (MLP 8). On the 128 natural sentences: each unit's activation at the cue noun and at the final token, plural-cue minus singular-cue separation in
pooled std and its sign against the panel (v334 / v368 signs: 3465 negative, 493 / 1036 / 829 positive for plural; 2483 and 1779 taken from their v368
contribution signs, both negative into 829's factors, so their own plural sign is reported and compared with the panel rows' sign measured here too).
PREDICTIONS (scored as written; failures preserved; priors from v344 / v368)
    pred_a_rows_load                          128 rows with valid cue offsets (instrument)
    pred_b_all_chain_units_separate_at_the_cue  each of 3465, 493, 1036, 2483, 1779, 829 separates plural-cue from singular-cue rows at the cue noun by >= 0.8 pooled std
    pred_c_signs_match_panel                  each unit's text sign at the cue equals its panel-row sign at the noun (measured in the same run on the v76 rows)
    pred_d_separation_grows_or_holds_up_the_chain  |separation| at 829 >= 0.7 x |separation| at 3465 on text (the state is not lost up the chain). Prior: unsure.
    pred_e_none_at_the_final                  no unit separates at the final token by >= 1 pooled std
PRICE (registered maximum): 2 natural batches + 3 panel batches (blocks 0-8) = 5 forwards; 0 backwards; 0 fits. Bar <= 7.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/unit_chain_natural_v369_result.json"
CANDIDATE_ID = "pronoun_number.unit_chain_natural_v369"
UNITS = {3: (3465, 493), 5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}; BATCH = 64
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
SEP_MIN, HOLD, FINAL_MAX = 0.8, 0.7, 1.0
FORWARDS_MAX = 7
PREDICTIONS = {"pred_a_rows_load": "128 rows", "pred_b_all_chain_units_separate_at_the_cue": ">= 0.8 std x 6", "pred_c_signs_match_panel": "6 signs", "pred_d_separation_grows_or_holds_up_the_chain": "829 >= 0.7x 3465", "pred_e_none_at_the_final": "< 1 std x 6"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    cue = [int(r["cue_offset"]) for r in recs]; verb = [int(r["second_offset"]) for r in recs]; fin = [len(r["ids"]) - 1 for r in recs]; plural = [i for i, r in enumerate(recs) if r["cue"] == "plural"]; sing = [i for i, r in enumerate(recs) if r["cue"] != "plural"]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "plural_rows": len(plural), "singular_rows": len(sing), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"sep_min": SEP_MIN, "hold": HOLD, "final_max": FINAL_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); POS = {"cue": cue, "final": fin}; units = [u for l in UNITS for u in UNITS[l]]
    def capture(ids, positions):
        out = {(p_, u): [] for p_ in positions for u in units}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None; idx = torch.arange(ids.shape[0])
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    hh = dod_units.hidden(model, block.mlp, xin).float().cpu()
                    for p_, pos in positions.items():
                        for u in UNITS[l]: out[(p_, u)].append(hh[idx, torch.tensor(pos), u])
                x = x + block.mlp(xin)
                if l == 8: break
        return {k: torch.cat(v) for k, v in out.items()}
    H = {k: [] for p_ in POS for u in units for k in [(p_, u)]}
    for s0 in range(0, len(recs), BATCH):
        o = capture(nat[s0:s0 + BATCH], {p_: offs[s0:s0 + BATCH] for p_, offs in POS.items()}); forwards += 1
        for k, v in o.items(): H[k].append(v)
    H = {k: torch.cat(v) for k, v in H.items()}
    # panel signs at the noun, same run
    rows, he, she, agents, objects = g.build(); nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    fw = L.ManualForward(backend); HP = {u: [] for u in units}
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]; o = capture(fw._tokens(chunk), {"noun": [noun_of(r_) for r_ in chunk]}); forwards += 1
        for u in units: HP[u].append(o[("noun", u)])
    HP = {u: torch.cat(v) for u, v in HP.items()}; partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pp = [i for i, r_ in enumerate(rows) if r_.present]; ps = [partner[(rows[i].construction, rows[i].group, False)] for i in pp]
    def sep(a_, b_): std = float(torch.cat([a_ - a_.mean(), b_ - b_.mean()]).std()); return float((a_.median() - b_.median()) / std)
    text = {p_: {str(u): sep(H[(p_, u)][plural], H[(p_, u)][sing]) for u in units} for p_ in POS}; panel = {str(u): sep(HP[u][pp], HP[u][ps]) for u in units}
    report = {"text_separation_over_std": text, "panel_noun_separation_over_std": panel}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_rows_load": len(recs) == 128, "pred_b_all_chain_units_separate_at_the_cue": all(abs(text["cue"][str(u)]) >= SEP_MIN for u in units), "pred_c_signs_match_panel": all(text["cue"][str(u)] * panel[str(u)] > 0 for u in units),
                   "pred_d_separation_grows_or_holds_up_the_chain": abs(text["cue"]["829"]) >= HOLD * abs(text["cue"]["3465"]), "pred_e_none_at_the_final": all(abs(text["final"][str(u)]) < FINAL_MAX for u in units)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "unit_chain_natural_result_v369", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
