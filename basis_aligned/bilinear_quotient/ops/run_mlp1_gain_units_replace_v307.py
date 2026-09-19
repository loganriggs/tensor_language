#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_alpha_rises_when_gain_units_restored pred_c_rise_matches_net_census pred_d_rise_beats_random_pairs pred_e_number_margin_moves_beyond_null
"""MLP 1: REPLACE-edit of the gain units (v307). v306: units 3289 and 624 are the two largest net cancellers of the token lookup in context (alone
they write +867 / +664 along the token's entry per row; in context ~ +60 each; 22% of the whole net change). v305's zeroing edit was the wrong
counterfactual (it removes the unit's whole in-context write, already ~ 0 along the entry). The edit that tests the claim replaces each unit's
in-context activation with its single-token activation for the same token, h_j <- h_j(alone), at the target position (a) on phrase-A length-8 rows
(224 targets; alpha and cosine vs the unchanged table) and (b) on the v76 pronoun rows at the noun (they - he margin; downstream native).
Null: 12 seeded random 2-unit sets given the same replacement. Registered: alpha rises by the pair's net share, (807 + 621) / ||T||-scale ~ 0.15 at 8 tokens.
PREDICTIONS (scored as written; failures preserved; priors from v306)
    pred_a_baseline_replays                   the unedited pass reproduces v291's alpha at 8 tokens (0.335) within 0.01
    pred_b_alpha_rises_when_gain_units_restored  median alpha (edit) - alpha (baseline) >= 0.10
    pred_c_rise_matches_net_census            that rise is within 0.05 of the pair's pooled net loss from v306 expressed in alpha (0.169 at 8 tokens)
    pred_d_rise_beats_random_pairs            the rise exceeds 3x the largest |median change| among the 12 random pairs
    pred_e_number_margin_moves_beyond_null    |mean oriented they - he margin change| on the v76 rows exceeds 3x the largest among the 12 random pairs. Prior: unsure.
PRICE (registered maximum): 1 table + 14 filler passes + 1 v76 table + 3 v76 batches x 14 = 58 forwards; 0 backwards; 0 fits. Bar <= 62.
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
OUT = ROOT / "circuits/followups/mlp1_gain_units_replace_v307_result.json"
CANDIDATE_ID = "mlp1.token_table.gain_units_replace_v307"
UNITS, LAYER, N_NULL, SEED = (3289, 624), 1, 12, 305
PHRASE = (",", " and", " of", " the", " very"); K, N, BATCH = 8, 32, 32
REPLAY_ALPHA, REPLAY_TOL, RISE_MIN, NULL_FACTOR, CENSUS_RISE, CENSUS_TOL = 0.335, 0.01, 0.10, 3.0, 0.169, 0.05
FORWARDS_MAX = 62
PREDICTIONS = {"pred_a_baseline_replays": "alpha 0.335 +- 0.01", "pred_b_alpha_rises_when_gain_units_restored": ">= 0.10", "pred_c_rise_matches_net_census": "within 0.05 of 0.169", "pred_d_rise_beats_random_pairs": "> 3x null", "pred_e_number_margin_moves_beyond_null": "> 3x null"}


def mlp1_write_with_replace(backend, tokens, pos, units, h_alone):
    """MLP 1's write at `pos` with the given units' activations at `pos` replaced by `h_alone[row, unit]` (None = native)."""
    torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h; idx = torch.arange(tokens.shape[0], device=tokens.device)
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l in (0, 1):
            block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == 1:
                h = dod_units.hidden(model, block.mlp, xin)
                if units is not None:
                    u = torch.tensor(list(units), device=h.device)
                    for j, uj in enumerate(u.tolist()): h[idx, pos, uj] = h_alone[:, j].to(h.dtype).to(h.device)
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else: m = block.mlp(xin)
            x = x + m
    return m[idx, pos].float().cpu(), x


def margins_with_replace(backend, fw, rows, units, h_alone_rows, noun_of, readers):
    """they - he margins at the final token with MLP-1 units replaced at the noun position by their single-token activations."""
    torch, F, model = backend.torch, backend.F, backend.model; tokens = fw._tokens(rows); pos = torch.tensor([noun_of(r) for r in rows], device=tokens.device)
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; idx = torch.arange(len(rows), device=tokens.device)
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == 1 and units is not None:
                h = dod_units.hidden(model, block.mlp, xin)
                for j, uj in enumerate(units): h[idx, pos, uj] = h_alone_rows[:, j].to(h.dtype).to(h.device)
                x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    out = []
    for i, row in enumerate(rows):
        lg = logits[i, row.final].float(); out.append({name: float(lg[a] - lg[b]) for name, (a, b) in readers.items()})
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = [fill[i % len(fill)] for i in range(K)]; targets = sorted({t for v in cls.values() for t in v})
    rng = random.Random(SEED); pool = [j for j in range(4608) if j not in UNITS]; null_sets = [tuple(sorted(rng.sample(pool, len(UNITS)))) for _ in range(N_NULL)]
    conditions = [("baseline", None), ("edit", UNITS)] + [(f"null{k}", s) for k, s in enumerate(null_sets)]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "targets": len(targets), "units": list(UNITS), "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_alpha": REPLAY_ALPHA, "replay_tol": REPLAY_TOL, "rise_min": RISE_MIN, "null_factor": NULL_FACTOR, "census_rise": CENSUS_RISE, "census_tol": CENSUS_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0
    mlp = model.transformer.h[1].mlp
    def table_h(token_ids):
        ids = torch.tensor(token_ids, device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda"))
        h = dod_units.hidden(model, mlp, F.rms_norm(c["x1"].to("cuda"), (c["x1"].shape[-1],))).float().cpu(); return c["mlp1"], h
    T, h_tab = table_h(targets); forwards += 1
    toks = torch.tensor([filler + [t] for t in targets], device="cuda"); pos = torch.full((len(targets),), K, dtype=torch.long, device="cuda")
    stats = {}
    for name, units in conditions:
        W, _ = mlp1_write_with_replace(backend, toks, pos, units, None if units is None else h_tab[:, torch.tensor(list(units))]); forwards += 1
        alpha = (W * T).sum(1) / (T * T).sum(1); cos = (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
        stats[name] = {"alpha_median": float(alpha.median()), "cos_median": float(cos.median())}
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    noun_ids = sorted({row.ids[noun_of(row)] for row in rows}); _, h_noun = table_h(noun_ids); forwards += 1; nindex = {t: i for i, t in enumerate(noun_ids)}
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    for name, units in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; hr = None if units is None else torch.stack([h_noun[nindex[r.ids[noun_of(r)]]][torch.tensor(list(units))] for r in chunk])
            out += margins_with_replace(backend, fw, chunk, units, hr, noun_of, readers); forwards += 1
        margins[name] = out
    def oriented(name):
        vals = [(margins[name][i]["they_he"] if row.present else -margins[name][i]["they_he"]) for i, row in enumerate(rows)]; return sum(vals) / len(vals)
    base_m = oriented("baseline"); edit_dm = oriented("edit") - base_m; null_dm = [oriented(f"null{k}") - base_m for k in range(N_NULL)]
    base_a = stats["baseline"]["alpha_median"]; rise = stats["edit"]["alpha_median"] - base_a; null_rise = [stats[f"null{k}"]["alpha_median"] - base_a for k in range(N_NULL)]
    replay = abs(base_a - REPLAY_ALPHA)
    report = {"alpha": {k: v["alpha_median"] for k, v in stats.items()}, "cos": {k: v["cos_median"] for k, v in stats.items()}, "alpha_rise_edit": rise, "alpha_rise_null_max_abs": max(abs(x) for x in null_rise), "baseline_margin_oriented": base_m,
              "margin_change_edit": edit_dm, "margin_change_null_max_abs": max(abs(x) for x in null_dm), "margin_change_nulls": null_dm, "replay_gap": replay, "null_sets": null_sets}
    print(json.dumps({k: v for k, v in report.items() if k not in ("null_sets",)}, indent=1))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_alpha_rises_when_gain_units_restored": rise >= RISE_MIN, "pred_c_rise_matches_net_census": abs(rise - CENSUS_RISE) <= CENSUS_TOL,
                   "pred_d_rise_beats_random_pairs": rise > NULL_FACTOR * report["alpha_rise_null_max_abs"], "pred_e_number_margin_moves_beyond_null": abs(edit_dm) > NULL_FACTOR * report["margin_change_null_max_abs"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_gain_units_replace_result_v307", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
