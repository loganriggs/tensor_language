#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_site_four_lower_the_margin pred_c_beats_random_sets pred_d_site_four_smaller_than_chain pred_e_both_exceed_either
"""Edit of the post-noun site's own units (v386). v385: the number at the token after the noun is rebuilt by the noun's relays plus site-specific units 3661
(MLP 4), 715 (MLP 5), 69 (MLP 6), 834 (MLP 7). Edits decide on the native v76 rows: zero the four at every position; compare with the six-unit chain (v370,
-14%) and with both sets together (ten units); null: 12 layer-matched random 4-sets; readout the oriented they - he margin.
PREDICTIONS (scored as written; failures preserved; priors from v370 / v385)
    pred_a_baseline_replays          the unedited margin replays 2.048 within 1e-3
    pred_b_site_four_lower_the_margin  zeroing the four lowers the margin by >= 0.03 of its native value
    pred_c_beats_random_sets         |four change| exceeds 3x the largest |change| among the 12 random 4-sets
    pred_d_site_four_smaller_than_chain  |four change| < |six-unit chain change| (0.286)
    pred_e_both_exceed_either        |ten change| > max(|four|, |six|) (the two sites add). Prior: unsure.
PRICE (registered maximum): 3 row batches x (1 baseline + 3 edits + 12 null) = 48 forwards; 0 backwards; 0 fits. Bar <= 52.
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
OUT = ROOT / "circuits/followups/site_units_edit_v386_result.json"
CANDIDATE_ID = "pronoun_number.site_units_edit_v386"
CHAIN = {3: (3465, 493), 5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}; SITE = {4: (3661,), 5: (715,), 6: (69,), 7: (834,)}; BOTH = {3: (3465, 493), 4: (3661,), 5: (1036, 715), 6: (2483, 69), 7: (1779, 834), 8: (829,)}; N_NULL, SEED, BATCH = 12, 386, 32
NATIVE_M, M_TOL, MOVE_MIN, NULL_FACTOR, SIX = 2.0481, 1e-3, 0.03, 3.0, 0.286
FORWARDS_MAX = 52
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_site_four_lower_the_margin": "<= -0.03 of native", "pred_c_beats_random_sets": "> 3x null", "pred_d_site_four_smaller_than_chain": "< 0.286", "pred_e_both_exceed_either": "ten > max"}


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
    rng = random.Random(SEED); layers = [4, 5, 6, 7]; named = {u for v in BOTH.values() for u in v}
    null_sets = []
    for _ in range(N_NULL):
        d = {}
        for l in layers: d.setdefault(l, []).append(rng.choice([j for j in range(4608) if j not in named]))
        null_sets.append({l: tuple(v) for l, v in d.items()})
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "chain": {str(k): list(v) for k, v in CHAIN.items()}, "site": {str(k): list(v) for k, v in SITE.items()}, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "move_min": MOVE_MIN, "null_factor": NULL_FACTOR, "six": SIX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); fw = L.ManualForward(backend); forwards = 0
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("site", SITE), ("chain", CHAIN), ("both", BOTH)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, edits in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            out += margins_multi(backend, fw, rows[start:start + BATCH], edits, readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum((margins[name][i]["they_he"] if row.present else -margins[name][i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
    base = oriented("baseline"); d_site = oriented("site") - base; d_chain = oriented("chain") - base; d_both = oriented("both") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_site_four": d_site, "change_chain_six": d_chain, "change_both_ten": d_both, "null_changes": nulls, "null_max_abs": max(abs(v) for v in nulls), "replay_gap": abs(base - NATIVE_M), "site_rel": d_site / base, "both_rel": d_both / base}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_site_four_lower_the_margin": d_site <= -MOVE_MIN * abs(base), "pred_c_beats_random_sets": abs(d_site) > NULL_FACTOR * report["null_max_abs"], "pred_d_site_four_smaller_than_chain": abs(d_site) < SIX, "pred_e_both_exceed_either": abs(d_both) > max(abs(d_site), abs(d_chain))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "site_units_edit_result_v386", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
