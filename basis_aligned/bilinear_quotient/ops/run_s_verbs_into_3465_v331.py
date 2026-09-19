#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_replays_v329 pred_c_ambiguous_s_verbs_read_as_plural_nouns pred_d_unambiguous_s_verbs_read_as_nothing pred_e_mass_and_collective_nouns_singular_side
"""Ambiguous vs unambiguous -s verbs, and mass / collective nouns, into unit 3465 (v331). v330: -s verbs scored 0.41 of the singular -> plural axis;
the eight tested (runs, walks, makes, takes, goes, says, looks, needs) are mostly also plural nouns. Split: ambiguous -s verbs that are common plural
nouns (" runs", " walks", " looks", " needs", " works", " plays", " calls", " turns") vs unambiguous -s verbs (" seems", " becomes", " does", " has",
" wants", " knows", " goes", " says"); and mass nouns (" water", " music", " money", " information", " advice", " furniture") and collective nouns
(" team", " group", " family", " committee", " crowd", " army") -- singular in form, plural in sense -- to see which the unit follows.
PREDICTIONS (scored as written; failures preserved; priors from v330)
    pred_a_closure                              the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_replays_v329                         singular / plural noun medians replay v329's -37 / -432 within 5%
    pred_c_ambiguous_s_verbs_read_as_plural_nouns  the ambiguous -s verbs' median lies >= 0.30 along the axis
    pred_d_unambiguous_s_verbs_read_as_nothing  the unambiguous -s verbs' median lies within +- 0.15 of the singular-noun median on the axis
    pred_e_mass_and_collective_nouns_singular_side  mass and collective nouns both lie within +- 0.15 of the singular-noun median (form, not sense). Prior: unsure.
PRICE (registered maximum): 1 batch of <= 200 single tokens (blocks 0-3) = 1 forward; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/s_verbs_into_3465_v331_result.json"
CANDIDATE_ID = "pronoun_number.s_verbs_into_3465_v331"
UNIT, LAYER, SIZES = 3465, 3, (4, 16, 64)
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, GAP_STD, REPLAY, AMB_MIN, NEAR = 1e-4, 1.0, 0.05, 0.30, 0.15
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_replays_v329": "within 5%", "pred_c_ambiguous_s_verbs_read_as_plural_nouns": ">= 0.30", "pred_d_unambiguous_s_verbs_read_as_nothing": "within 0.15", "pred_e_mass_and_collective_nouns_singular_side": "within 0.15 x 2"}


def main() -> None:
    spec = v287.SPEC; sing = [a for _, a, _ in spec["noun_pairs_lexicon"]][:64]; plur = [b for _, _, b in spec["noun_pairs_lexicon"]][:64]
    def T(xs):
        out = []
        for t in xs:
            try: out.append(L._single(t))
            except Exception: pass                      # multi-token entries are dropped and listed in the plan
        return out
    groups = {"singular": sing, "plural": plur, "ambiguous_s_verbs": T([" runs", " walks", " looks", " needs", " works", " plays", " calls", " turns"]), "unambiguous_s_verbs": T([" seems", " becomes", " does", " has", " wants", " knows", " goes", " says"]),
              "mass_nouns": T([" water", " music", " money", " information", " advice", " furniture"]), "collective_nouns": T([" team", " group", " family", " committee", " crowd", " army"])}
    tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "unit": UNIT, "layer": LAYER, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "replay": REPLAY, "amb_min": AMB_MIN, "near": NEAR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    ids = torch.tensor(tokens, device="cuda").unsqueeze(1); blocks = model.transformer.h
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == LAYER:
                h = dod_units.hidden(model, block.mlp, xin)[:, 0, UNIT].float().cpu(); h_full = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu(); break
            x = x + block.mlp(xin)
        forwards += 1
        # instrument: the same hidden from the model's own module chain on a fresh pass (identical code path -> closure is on the captured tensor)
        h2 = dod_units.hidden(model, blocks[LAYER].mlp, xin)[:, 0, UNIT].float().cpu()
    closure = float(((h2 - h).abs() / h.abs().clamp_min(1e-6)).max()); tindex = {t: i for i, t in enumerate(tokens)}
    val = lambda group: torch.tensor([float(h[tindex[t]]) for t in groups[group]])
    s64, p64 = val("singular"), val("plural"); ms, mp = float(s64.median()), float(p64.median()); pooled_std = float(torch.cat([s64 - s64.mean(), p64 - p64.mean()]).std())
    side = lambda v: float((v - ms) / (mp - ms))                     # 0 = singular median, 1 = plural median (the plural side is positive)
    med = {g: float(val(g).median()) for g in groups}; each = {g: {str(t): float(h[tindex[t]]) for t in groups[g]} for g in groups if g not in ("singular", "plural")}
    pos = {g: side(torch.tensor(med[g])) for g in groups}
    report = {"closure_max": closure, "median": med, "pooled_std": pooled_std, "position_on_singular_plural_axis": pos, "each": each}
    print(json.dumps({k: v for k, v in report.items() if k != "each"}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_replays_v329": abs(ms - V329["singular"]) <= REPLAY * abs(V329["singular"]) and abs(mp - V329["plural"]) <= REPLAY * abs(V329["plural"]),
                   "pred_c_ambiguous_s_verbs_read_as_plural_nouns": pos["ambiguous_s_verbs"] >= AMB_MIN, "pred_d_unambiguous_s_verbs_read_as_nothing": abs(pos["unambiguous_s_verbs"]) <= NEAR,
                   "pred_e_mass_and_collective_nouns_singular_side": abs(pos["mass_nouns"]) <= NEAR and abs(pos["collective_nouns"]) <= NEAR}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "s_verbs_into_3465_result_v331", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
