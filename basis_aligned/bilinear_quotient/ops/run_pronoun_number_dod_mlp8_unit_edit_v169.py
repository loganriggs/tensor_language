#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_units_at_noun_are_live pred_c_units_beat_random_unit_triples pred_d_units_selective pred_e_all_positions_removes_more
"""Pronoun number they/he DoD (v169): EDIT the three MLP-8 units that v168 nominated (829, 953, 1030; 65% of MLP 8's number write on 9.6's direction at the noun). Sizing prior from the chain (MLP 8 share of the noun state ~0.50, 9.6 single 0.66 of 1.55, unit share 0.65): ~0.14 of the margin -- LIVE is plausible but not assured.

Lane: Claude circuit lane. Parent: v164 (units 3152 + 3943 carry 76% of MLP 8's gender write on 9.6's reader direction at the noun). Forward with
a hook on block 8's MLP that zeroes chosen hidden units (Lx * Rx)_j at chosen positions; everything else native. Arms on the v71 fresh rows:
native (instrument against the producer), zero {3152, 3943} at the gendered-noun position, zero them at ALL positions, zero the v164 top-10 at the noun,
and 16 random 2-unit sets zeroed at the noun (null). Damage = oriented drop of the he/she margin at the final query; readers will-would / who-which /
night-day as |move|.
PREDICTIONS (scored as written; failures preserved; priors unsure -- 9.6 is one of four heads and MLP 8 one of its inputs)
    pred_a_instrument_replays_native      hooked forward with no edit = producer native <= 1e-4
    pred_b_units_at_noun_are_live     fraction >= 0.10 and positive >= 0.75
    pred_c_units_beat_random_unit_triples  damage > max of 16 random 2-unit zeroings at the noun
    pred_d_units_selective            each reader |move| <= null mean |move| + 0.25 x damage
    pred_e_all_positions_removes_more     zeroing the two units at all positions removes >= 1.5 x the noun-only damage (the verb copy, v82)
PRICE (registered maximum): 2 batches x (native + producer + 3 arms + 16 nulls) = 42 forwards; 0 backwards; 0 fits. Bar <= 46.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp8_unit_edit_v169_result.json"
V164 = ROOT / "circuits/followups/pronoun_number_dod_mlp8_unit_census_v168_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp8_unit_edit_v169"
LAYER, UNITS = 8, (829, 953, 1030)
NULL_SEEDS = tuple(range(6901, 6917))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, MORE, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 1.5, 1e-4
FORWARDS_MAX = 46
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_units_at_noun_are_live": "live", "pred_c_units_beat_random_unit_triples": "> random max", "pred_d_units_selective": "three gates", "pred_e_all_positions_removes_more": ">= 1.5 x noun-only"}


def forward_margins(backend, fw, rows, edits, readers):
    """edits: None or (units, positions_fn) -> zero those MLP-8 hidden units at positions_fn(row) (None = all positions). Returns per-row dicts."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens = fw._tokens(rows); mlp = model.transformer.h[LAYER].mlp
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            x = live + attention
            xin = F.rms_norm(x, (model.config.n_embd,))
            if layer == LAYER and edits is not None:
                units, positions_fn = edits
                h = mlp.Left(xin) * mlp.Right(xin)
                idx = torch.tensor(list(units), device=h.device)
                for i, row in enumerate(rows):
                    pos = positions_fn(row)
                    if pos is None: h[i, :, idx] = 0
                    else: h[i, pos, idx] = 0
                x = x + mlp.Down(h) + mlp.Down_bias
            else:
                x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    out = []
    for i, row in enumerate(rows):
        l = logits[i, row.final].float()
        out.append({"answer": float(l[row.answer_id]), "foil": float(l[row.foil_id]), **{name: float(l[a] - l[b]) for name, (a, b) in readers.items()}})
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    top10 = tuple(j for j, _ in json.loads(V164.read_text())["top_units"][:10])
    readers = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "top10": list(top10), "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "gate_ratio": GATE_RATIO, "more": MORE, "instrument_tol": INSTRUMENT_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fw = L.ManualForward(backend)
    forwards = 0
    def run(edits):
        nonlocal forwards; out = []
        for start in range(0, len(rows), v1.BATCH):
            out.extend(forward_margins(backend, fw, rows[start:start + v1.BATCH], edits, readers)); forwards += 1
        return out
    native = run(None)
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    def summarize(arm):
        d = [(nat["answer"] - nat["foil"]) - (a["answer"] - a["foil"]) for nat, a in zip(native, arm)]
        m = sum(d) / len(d); native_margin = sum(nat["answer"] - nat["foil"] for nat in native) / len(native)
        return {"damage": m, "fraction": m / native_margin, "positive": sum(1 for x in d if x > 0) / len(d), **{f"{k}_abs_move": sum(abs(a[k] - nat[k]) for nat, a in zip(native, arm)) / len(d) for k in readers}}
    arms = {"two_units_noun": summarize(run((UNITS, noun_of))), "two_units_all_positions": summarize(run((UNITS, lambda r: None))), "top10_noun": summarize(run((top10, noun_of)))}
    nulls = []
    for seed in NULL_SEEDS:
        gen = torch.Generator(device="cpu").manual_seed(seed); units = tuple(torch.randperm(4608, generator=gen)[:3].tolist())
        nulls.append(summarize(run((units, noun_of))))
    null_max = max(s["damage"] for s in nulls); null_moves = {k: sum(s[f"{k}_abs_move"] for s in nulls) / len(nulls) for k in readers}
    two = arms["two_units_noun"]
    gates = {k: two[f"{k}_abs_move"] <= null_moves[k] + GATE_RATIO * two["damage"] for k in readers}
    for k, s in arms.items(): print(k, {kk: round(vv, 3) for kk, vv in s.items()})
    print("null max", round(null_max, 3), "null moves", {k: round(v, 3) for k, v in null_moves.items()}, "gates", gates)
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_units_at_noun_are_live": two["fraction"] >= LIVE_FRACTION and two["positive"] >= LIVE_POSITIVE,
                   "pred_c_units_beat_random_unit_triples": two["damage"] > null_max, "pred_d_units_selective": all(gates.values()),
                   "pred_e_all_positions_removes_more": arms["two_units_all_positions"]["damage"] >= MORE * two["damage"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp8_unit_edit_result_v169", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": arms, "null_damage_max": null_max, "null_moves": null_moves,
                               "gates": gates, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
