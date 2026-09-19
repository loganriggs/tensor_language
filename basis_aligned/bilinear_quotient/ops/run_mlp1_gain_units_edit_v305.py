#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_alpha_rises_when_gain_units_zeroed pred_c_alpha_rise_beats_random_pairs pred_d_lookup_direction_kept pred_e_number_margin_moves_beyond_null
"""MLP 1: edit the gain units (v305). v303 / v304: units 3289 and 624 of MLP 1 are context-gain units carrying ~30% of the self-cancellation of the
token lookup (v301: the cross term projects -0.44 to -0.75 on the lookup). Edits decide: zero the two units at every position (a) on phrase-A
length-8 rows (224 targets) and read alpha (projection of MLP 1's write on the token's single-token table entry, table unchanged) and the direction
(cosine with the table); (b) on the v76 pronoun-number rows and read the they - he margin at the final token (the whole model downstream is native).
Null: 12 seeded random 2-unit sets of MLP 1 (disjoint from the pair), same positions. Registered reading: alpha rises by roughly the pair's share of
the cut (~0.3 x 0.66 ~ 0.2 at 8 tokens) and the lookup direction is kept; the margin effect is unknown (the number chain reads the table part,
v296, so restoring lookup gain should if anything strengthen the number state).
PREDICTIONS (scored as written; failures preserved; priors from v303 / v304 / v296)
    pred_a_baseline_replays                 the unedited pass reproduces v291's alpha at 8 tokens (0.335) within 0.01 and v76's margins within 1e-3
    pred_b_alpha_rises_when_gain_units_zeroed  median alpha (edit) - alpha (baseline) >= 0.10
    pred_c_alpha_rise_beats_random_pairs    that rise exceeds 3x the largest |median change| among the 12 random pairs
    pred_d_lookup_direction_kept            median cosine(write, table) under the edit >= the baseline's - 0.05
    pred_e_number_margin_moves_beyond_null  |mean they - he margin change| on the v76 rows exceeds 3x the largest among the 12 random pairs. Prior: unsure (reported either way).
PRICE (registered maximum): 1 table + 14 filler passes (baseline, edit, 12 null) + 3 v76 batches x 14 = 57 forwards; 0 backwards; 0 fits. Bar <= 60.
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
OUT = ROOT / "circuits/followups/mlp1_gain_units_edit_v305_result.json"
CANDIDATE_ID = "mlp1.token_table.gain_units_edit_v305"
UNITS, LAYER, N_NULL, SEED = (3289, 624), 1, 12, 305
PHRASE = (",", " and", " of", " the", " very"); K, N, BATCH = 8, 32, 32
REPLAY_ALPHA, REPLAY_TOL, RISE_MIN, NULL_FACTOR, COS_SLACK = 0.335, 0.01, 0.10, 3.0, 0.05
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_baseline_replays": "alpha 0.335 +- 0.01", "pred_b_alpha_rises_when_gain_units_zeroed": ">= 0.10", "pred_c_alpha_rise_beats_random_pairs": "> 3x null", "pred_d_lookup_direction_kept": ">= baseline - 0.05", "pred_e_number_margin_moves_beyond_null": "> 3x null"}


def mlp1_write_with_edit(backend, tokens, pos, units):
    """MLP 1's write at `pos` with the given units zeroed (None = native)."""
    torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h; idx = torch.arange(tokens.shape[0])
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l in (0, 1):
            block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == 1:
                h = dod_units.hidden(model, block.mlp, xin)
                if units is not None: h[:, :, torch.tensor(list(units), device=h.device)] = 0
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else: m = block.mlp(xin)
            x = x + m
    return m[idx, pos].float().cpu()


def main() -> None:
    rows, he, she, agents, objects = g.build()
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = [fill[i % len(fill)] for i in range(K)]; targets = sorted({t for v in cls.values() for t in v})
    rng = random.Random(SEED); pool = [j for j in range(4608) if j not in UNITS]; null_sets = [tuple(sorted(rng.sample(pool, len(UNITS)))) for _ in range(N_NULL)]
    conditions = [("baseline", None), ("edit", UNITS)] + [(f"null{k}", s) for k, s in enumerate(null_sets)]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "targets": len(targets), "units": list(UNITS), "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_alpha": REPLAY_ALPHA, "replay_tol": REPLAY_TOL, "rise_min": RISE_MIN, "null_factor": NULL_FACTOR, "cos_slack": COS_SLACK}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; fw = L.ManualForward(backend); forwards = 0
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; T = tab["mlp1"]
    toks = torch.tensor([filler + [t] for t in targets], device="cuda"); pos = torch.full((len(targets),), K, dtype=torch.long, device="cuda")
    stats = {}
    for name, units in conditions:
        W = mlp1_write_with_edit(backend, toks, pos, units); forwards += 1
        alpha = (W * T).sum(1) / (T * T).sum(1); cos = (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
        stats[name] = {"alpha_median": float(alpha.median()), "cos_median": float(cos.median()), "norm_median": float(W.norm(dim=1).median())}
    readers = {"they_he": (L._single(" they"), L._single(" he"))}
    margins = {}
    for name, units in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; out += dod_units.forward_margins(backend, fw, chunk, LAYER, None if units is None else (units, lambda row: None), readers); forwards += 1
        margins[name] = out
    def oriented(name):
        vals = []
        for i, row in enumerate(rows):
            m = margins[name][i]["they_he"]; vals.append(m if row.present else -m)      # present = plural (they) rows; oriented so positive = correct
        return sum(vals) / len(vals)
    base_m = oriented("baseline"); edit_dm = oriented("edit") - base_m; null_dm = [oriented(f"null{k}") - base_m for k in range(N_NULL)]
    base_a = stats["baseline"]["alpha_median"]; rise = stats["edit"]["alpha_median"] - base_a; null_rise = [stats[f"null{k}"]["alpha_median"] - base_a for k in range(N_NULL)]
    replay = abs(base_a - REPLAY_ALPHA)
    report = {"alpha": {k: v["alpha_median"] for k, v in stats.items()}, "cos": {k: v["cos_median"] for k, v in stats.items()}, "alpha_rise_edit": rise, "alpha_rise_null_max_abs": max(abs(x) for x in null_rise), "baseline_margin_oriented": base_m,
              "margin_change_edit": edit_dm, "margin_change_null_max_abs": max(abs(x) for x in null_dm), "margin_change_nulls": null_dm, "replay_gap": replay, "null_sets": null_sets}
    print(json.dumps({k: v for k, v in report.items() if k not in ("null_sets",)}, indent=1))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_alpha_rises_when_gain_units_zeroed": rise >= RISE_MIN, "pred_c_alpha_rise_beats_random_pairs": rise > NULL_FACTOR * report["alpha_rise_null_max_abs"],
                   "pred_d_lookup_direction_kept": stats["edit"]["cos_median"] >= stats["baseline"]["cos_median"] - COS_SLACK, "pred_e_number_margin_moves_beyond_null": abs(edit_dm) > NULL_FACTOR * report["margin_change_null_max_abs"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_gain_units_edit_result_v305", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
