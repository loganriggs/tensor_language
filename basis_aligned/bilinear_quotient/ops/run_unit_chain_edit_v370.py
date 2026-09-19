#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_chain_lowers_the_margin pred_c_beats_random_sets pred_d_chain_exceeds_mlp3_trio pred_e_upper_four_carry_most
"""Joint edit of the named MLP unit chain (v370). v368 / v369: the noun's number state is rebuilt MLP by MLP through 3465 / 493 (MLP 3), 1036 (MLP 5), 2483
(MLP 6), 1779 (MLP 7) and 829 (MLP 8), all live on text. Edits decide on the native v76 rows: zero all six at every position (units live in different layers,
so the edit is applied layer by layer in one forward), compare with the MLP-3 trio {3465, 114, 493} (v362: -3.6%) and with the upper four {1036, 2483, 1779,
829} alone; null: 12 seeded random 6-unit sets drawn one per named layer position (MLP 3 x 2, 5, 6, 7, 8); readout the oriented they - he margin.
PREDICTIONS (scored as written; failures preserved; priors from v195 / v196 / v362)
    pred_a_baseline_replays      the unedited margin replays 2.048 within 1e-3
    pred_b_chain_lowers_the_margin  zeroing the six lowers the oriented margin by >= 0.08 of its native value
    pred_c_beats_random_sets     |chain change| exceeds 3x the largest |change| among the 12 random sets
    pred_d_chain_exceeds_mlp3_trio  |chain change| > |v362 trio change| (0.074)
    pred_e_upper_four_carry_most |upper-four change| >= 0.5 x |chain change|. Prior: unsure -- MLP 8's 829 alone was small in v169b-era edits.
PRICE (registered maximum): 3 row batches x (1 baseline + 2 edits + 12 null) = 45 forwards; 0 backwards; 0 fits. Bar <= 48.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/unit_chain_edit_v370_result.json"
CANDIDATE_ID = "pronoun_number.unit_chain_edit_v370"
CHAIN = {3: (3465, 493), 5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}; UPPER = {5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}; N_NULL, SEED = 12, 370
PHRASE = (",", " and", " of", " the", " very"); K, N, BATCH = 8, 32, 32
NATIVE_M, M_TOL, MOVE_MIN, NULL_FACTOR, TRIO = 2.0481, 1e-3, 0.08, 3.0, 0.0738
FORWARDS_MAX = 48
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_chain_lowers_the_margin": "<= -0.08 of native", "pred_c_beats_random_sets": "> 3x null", "pred_d_chain_exceeds_mlp3_trio": "> 0.074", "pred_e_upper_four_carry_most": ">= 0.5x chain"}


def margins_multi(backend, fw, rows, edits, readers):
    """they - he margins with `edits` = {layer: units} zeroed at every position (several layers in one forward)."""
    torch, F, model = backend.torch, backend.F, backend.model; tokens = fw._tokens(rows)
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if edits and l in edits:
                h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return [{name: float(logits[i, row.final, a] - logits[i, row.final, b]) for name, (a, b) in readers.items()} for i, row in enumerate(rows)]


def main() -> None:
    rows, he, she, agents, objects = g.build()
    rng = random.Random(SEED); layers = [3, 3, 5, 6, 7, 8]; named = {u for v in CHAIN.values() for u in v}
    null_sets = []
    for _ in range(N_NULL):
        d = {}
        for l in layers: d.setdefault(l, []).append(rng.choice([j for j in range(4608) if j not in named]))
        null_sets.append({l: tuple(v) for l, v in d.items()})
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "chain": {str(k): list(v) for k, v in CHAIN.items()}, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "move_min": MOVE_MIN, "null_factor": NULL_FACTOR, "trio": TRIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); fw = L.ManualForward(backend); forwards = 0
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("chain", CHAIN), ("upper", UPPER)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, edits in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            out += margins_multi(backend, fw, rows[start:start + BATCH], edits, readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum((margins[name][i]["they_he"] if row.present else -margins[name][i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
    base = oriented("baseline"); d_chain = oriented("chain") - base; d_upper = oriented("upper") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_chain": d_chain, "change_upper_four": d_upper, "null_changes": nulls, "null_max_abs": max(abs(v) for v in nulls), "replay_gap": abs(base - NATIVE_M), "chain_rel": d_chain / base}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_chain_lowers_the_margin": d_chain <= -MOVE_MIN * abs(base), "pred_c_beats_random_sets": abs(d_chain) > NULL_FACTOR * report["null_max_abs"], "pred_d_chain_exceeds_mlp3_trio": abs(d_chain) > TRIO, "pred_e_upper_four_carry_most": abs(d_upper) >= 0.5 * abs(d_chain)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "unit_chain_edit_result_v370", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
