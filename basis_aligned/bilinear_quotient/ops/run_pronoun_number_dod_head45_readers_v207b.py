#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_one_reader_loses_half pred_c_loser_is_a_contextual_reader pred_d_edit_selective pred_e_all_positions_removes_more
"""Pronoun number they/he DoD (v207b): WHERE is head 4.5's copy read? v207's margin arm pooled answer - foil, which flips sign between the two members of a pair (native 48.6 vs
v206's 196.6) -- void. Here the margin is they - he for every row, as in v206; the coefficient arm is unchanged (repeated for the receipt to be whole). v206: zeroing 4.5's slice at the verb drops the they - he margin 7.7% (10x any
other block-4 head) while the verb-position plural detector 829 grows 5% -- the copy does not act through 829. Here the four reader heads' weight-only
coefficients c_h = v_h . z_h(final) (h in {9.6, 12.4, 15.1, 10.5}; `fw.capture` + `readout_directions`) under three conditions -- native, 4.5 zeroed at the
verb, 4.5 zeroed at ALL positions -- pooled plural - singular over the 48 pairs; plus the margin and the three number-free readers (will-would,
who-which, night-day) for the verb edit, with the other eight block-4 heads at the verb as the matched null.
PREDICTIONS (scored as written; failures preserved; priors unsure except c, from v80: 12.4 / 15.1 / 10.5 read 40-57% from non-noun positions)
    pred_a_baseline_replays_v206     the native pooled margin matches v206 (196.62) within relative 1e-3
    pred_b_one_reader_loses_half     one reader head's coefficient drop is >= 0.50 of the summed coefficient drop over the four
    pred_c_loser_is_a_contextual_reader  that head is 12.4, 15.1 or 10.5 (not 9.6)
    pred_d_edit_selective            each unrelated reader |move| <= mean |move| over the eight other heads + 0.25 x the margin damage
    pred_e_all_positions_removes_more  zeroing 4.5 at all positions removes >= 1.5 x the verb-only margin damage
PRICE (registered maximum): 3 batches x (3 captures + 2 margin forwards + 8 null forwards) = 39 forwards; 0 backwards; 0 fits. Bar <= 42.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_head45_readers_v207b_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_head45_readers_v207b"
BLOCK, HEAD, V206_MARGIN, HALF, GATE_RATIO, MORE, BATCH = 4, 5, 196.62, 0.50, 0.25, 1.5, 32
HEADS = ((9, 6), (12, 4), (15, 1), (10, 5))
FORWARDS_MAX = 42
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_one_reader_loses_half": ">= 0.50", "pred_c_loser_is_a_contextual_reader": "not 9.6", "pred_d_edit_selective": "three gates", "pred_e_all_positions_removes_more": ">= 1.5x"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    verb_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns) + 1
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}; readers["target_they_he"] = (he, she)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "block": BLOCK, "head": HEAD, "reader_heads": list(HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"half": HALF, "gate_ratio": GATE_RATIO, "more": MORE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    fw = L.ManualForward(backend); fw.backend = backend
    comps = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, HEADS).set_components()
    directions = L.readout_directions(model, comps, he, she)
    block = model.transformer.h[BLOCK]

    class Edit:
        def __init__(self, head, where): self.head, self.where = head, where
        def __enter__(self):
            head, where = self.head, self.where
            def pre(_m, args):
                z = args[0].clone(); s, e = head * L.HEAD_DIM, (head + 1) * L.HEAD_DIM
                if where == "all": z[:, :, s:e] = 0
                else:
                    for i, p in enumerate(where): z[i, p, s:e] = 0
                return (z,)
            self.h = block.attn.c_proj.register_forward_pre_hook(pre); return self
        def __exit__(self, *a): self.h.remove()

    forwards = 0
    coeffs = {"native": {}, "verb": {}, "all": {}}; margins = {"native": [], "verb": [], "all": []}; null_margins = {h: [] for h in range(9) if h != HEAD}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]; verbs = [verb_of(r) for r in chunk]
        def coefs(store):
            out = {}
            for row in chunk:
                for c in comps:
                    for h in c.heads:
                        v = directions[(c.name, h)].float(); v = v / v.norm(); w = store[(row.row_id, c.name, row.final, h)].float()
                        out[(row.row_id, f"{c.layer}.{h}")] = float(w @ v.to(w.device))
            return out
        coeffs["native"].update(coefs(fw.capture(chunk, comps))); forwards += 1
        with Edit(HEAD, verbs): coeffs["verb"].update(coefs(fw.capture(chunk, comps))); forwards += 1
        with Edit(HEAD, "all"): coeffs["all"].update(coefs(fw.capture(chunk, comps))); forwards += 1
        margins["native"].extend(dod_units.forward_margins(backend, fw, chunk, 8, None, readers)); forwards += 1
        with Edit(HEAD, verbs): margins["verb"].extend(dod_units.forward_margins(backend, fw, chunk, 8, None, readers)); forwards += 1
        with Edit(HEAD, "all"): margins["all"].extend(dod_units.forward_margins(backend, fw, chunk, 8, None, readers)); forwards += 1
        for h in null_margins:
            with Edit(h, verbs): null_margins[h].extend(dod_units.forward_margins(backend, fw, chunk, 8, None, readers)); forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    def pooled_coef(cond, key): return sum(coeffs[cond][(rows[i].row_id, key)] - coeffs[cond][(rows[j].row_id, key)] for i, j in pairs)
    def pooled_margin(lst): return sum(lst[i]["target_they_he"] - lst[j]["target_they_he"] for i, j in pairs)
    keys = [f"{l}.{h}" for l, h in HEADS]
    coef_table = {k: {c: pooled_coef(c, k) for c in coeffs} for k in keys}
    drops = {k: coef_table[k]["native"] - coef_table[k]["verb"] for k in keys}; total_drop = sum(drops.values()); loser = max(keys, key=lambda k: abs(drops[k]))
    m_native = pooled_margin(margins["native"]); dmg_verb = m_native - pooled_margin(margins["verb"]); dmg_all = m_native - pooled_margin(margins["all"])
    unrel = [r for r in readers if r != "target_they_he"]
    moves = {r: sum(abs(a[r] - n[r]) for n, a in zip(margins["native"], margins["verb"])) / len(rows) for r in unrel}
    null_moves = {r: sum(sum(abs(a[r] - n[r]) for n, a in zip(margins["native"], null_margins[h])) / len(rows) for h in null_margins) / len(null_margins) for r in unrel}
    dmg_per_row = dmg_verb / len(pairs); gates = {r: moves[r] <= null_moves[r] + GATE_RATIO * dmg_per_row for r in unrel}
    report = {"coefficients_pooled": coef_table, "drops_verb_edit": drops, "total_drop": total_drop, "loser": loser, "loser_share": drops[loser] / total_drop if total_drop else None, "margin_native_pooled": m_native, "damage_verb_pooled": dmg_verb,
              "damage_all_pooled": dmg_all, "reader_moves": moves, "null_reader_moves": null_moves, "gates": gates, "null_margin_damage": {h: m_native - pooled_margin(null_margins[h]) for h in null_margins}}
    print("coefficients", {k: {c: round(v, 2) for c, v in d.items()} for k, d in coef_table.items()}); print("drops", {k: round(v, 3) for k, v in drops.items()}, "loser", loser, round(report["loser_share"] or 0, 3))
    print("margin native", round(m_native, 2), "damage verb", round(dmg_verb, 2), "all", round(dmg_all, 2), "moves", {r: round(v, 4) for r, v in moves.items()}, "null", {r: round(v, 4) for r, v in null_moves.items()}, "gates", gates)
    predictions = {"pred_a_baseline_replays_v206": abs(m_native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_one_reader_loses_half": bool(report["loser_share"] is not None and report["loser_share"] >= HALF), "pred_c_loser_is_a_contextual_reader": loser != "9.6",
                   "pred_d_edit_selective": all(gates.values()), "pred_e_all_positions_removes_more": dmg_all >= MORE * dmg_verb}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_head45_readers_result_v207b", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
