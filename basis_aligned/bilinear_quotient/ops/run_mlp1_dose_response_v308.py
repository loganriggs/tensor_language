#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_rise_matches_census_at_every_k pred_c_rise_beats_random_sets_at_every_k pred_d_top200_restores_half pred_e_direction_improves_monotonically
"""MLP 1: dose-response of the net-unit account (v308). v306 / v307: the net per-unit census delta_j (which sums exactly to (alpha - 1)||T||) named
units 3289 / 624 as the head (22%), and restoring their single-token activations raised alpha by 0.152 (census 0.169). If the per-unit account is
additive for the whole layer, restoring the top-k net units by |pooled delta_j| should raise alpha by the census's cumulative share at every k.
Phrase A, 8 tokens, 224 targets; k = 2, 10, 50, 200 (v306 shares at 8 tokens: 0.22, 0.31, 0.37, 0.47 of the 0.665 loss = 0.15, 0.21, 0.24, 0.31);
null: 4 seeded random k-sets per k with the same replacement. The census is recomputed in this script from the same passes (no dependence on the
v306 file). Reported: alpha rise, census prediction, null max, cosine with the table, per k.
PREDICTIONS (scored as written; failures preserved; priors from v306 / v307)
    pred_a_baseline_replays                 the unedited pass reproduces v291's alpha at 8 tokens (0.335) within 0.01
    pred_b_rise_matches_census_at_every_k   |rise(k) - census(k)| <= 0.05 for k = 2, 10, 50, 200 (per-unit additivity holds through the layer)
    pred_c_rise_beats_random_sets_at_every_k  rise(k) > 3x the largest |median change| among the 4 random k-sets, every k
    pred_d_top200_restores_half             rise(200) >= 0.30 (half of the 0.665 loss from 4% of the units)
    pred_e_direction_improves_monotonically median cosine(write, table) increases with k
PRICE (registered maximum): 1 table + 1 baseline + 4 edits + 16 nulls = 22 forwards; 0 backwards; 0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_dose_response_v308_result.json"
CANDIDATE_ID = "mlp1.token_table.dose_response_v308"
KS, LAYER, N_NULL, SEED = (2, 10, 50, 200), 1, 4, 308
PHRASE = (",", " and", " of", " the", " very"); K, N, BATCH = 8, 32, 32
REPLAY_ALPHA, REPLAY_TOL, CENSUS_TOL, NULL_FACTOR, HALF_MIN = 0.335, 0.01, 0.05, 3.0, 0.30
FORWARDS_MAX = 24
PREDICTIONS = {"pred_a_baseline_replays": "alpha 0.335 +- 0.01", "pred_b_rise_matches_census_at_every_k": "<= 0.05 x 4", "pred_c_rise_beats_random_sets_at_every_k": "> 3x null x 4", "pred_d_top200_restores_half": ">= 0.30", "pred_e_direction_improves_monotonically": "monotone"}


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
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = [fill[i % len(fill)] for i in range(K)]; targets = sorted({t for v in cls.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "targets": len(targets), "ks": list(KS), "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_alpha": REPLAY_ALPHA, "replay_tol": REPLAY_TOL, "census_tol": CENSUS_TOL, "null_factor": NULL_FACTOR, "half_min": HALF_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    mlp = model.transformer.h[1].mlp; Dw = mlp.Down.weight.detach().float().cpu()
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
    T = c["mlp1"]; h_tab = dod_units.hidden(model, mlp, F.rms_norm(c["x1"].to("cuda"), (T.shape[-1],))).float().cpu()
    toks = torch.tensor([filler + [t] for t in targets], device="cuda"); pos = torch.full((len(targets),), K, dtype=torch.long, device="cuda")
    W0, x_ctx = mlp1_write_with_replace(backend, toks, pos, None, None); forwards += 1
    alpha_of = lambda W: (W * T).sum(1) / (T * T).sum(1); cos_of = lambda W: (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
    a0 = alpha_of(W0); replay = abs(float(a0.median()) - REPLAY_ALPHA)
    # census from the same pass: in-context activations at the target
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(toks), (model.config.n_embd,)); x0, v1_ = x, None
        for l in (0, 1):
            block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == 1: h_ctx = dod_units.hidden(model, mlp, xin)[torch.arange(len(targets)), pos].float().cpu()
            x = x + block.mlp(xin)
    That = T / T.norm(dim=1, keepdim=True); delta = (That @ Dw) * (h_ctx - h_tab); pd = delta.sum(0); order = torch.argsort(pd.abs(), descending=True)
    census = {k: float(-(delta[:, order[:k]].sum(1) / T.norm(dim=1)).median()) for k in KS}      # alpha units: restoring the set undoes its net change
    rng = random.Random(SEED); per_k = {}
    for k in KS:
        units = tuple(order[:k].tolist()); We, _ = mlp1_write_with_replace(backend, toks, pos, units, h_tab[:, torch.tensor(units)]); forwards += 1
        rise = float((alpha_of(We) - a0).median()); nulls = []
        for n_ in range(N_NULL):
            rs = tuple(sorted(rng.sample(range(4608), k))); Wn, _ = mlp1_write_with_replace(backend, toks, pos, rs, h_tab[:, torch.tensor(rs)]); forwards += 1; nulls.append(float((alpha_of(Wn) - a0).median()))
        per_k[k] = {"rise": rise, "census": census[k], "null_max_abs": max(abs(v) for v in nulls), "cos_median": float(cos_of(We).median()), "top_units": units[:12]}
    report = {"alpha_baseline_median": float(a0.median()), "cos_baseline_median": float(cos_of(W0).median()), "replay_gap": replay, "per_k": {str(k): v for k, v in per_k.items()}}
    print(json.dumps(report, indent=1))
    coss = [per_k[k]["cos_median"] for k in KS]
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_rise_matches_census_at_every_k": all(abs(v["rise"] - v["census"]) <= CENSUS_TOL for v in per_k.values()), "pred_c_rise_beats_random_sets_at_every_k": all(v["rise"] > NULL_FACTOR * v["null_max_abs"] for v in per_k.values()),
                   "pred_d_top200_restores_half": per_k[200]["rise"] >= HALF_MIN, "pred_e_direction_improves_monotonically": all(coss[i + 1] > coss[i] for i in range(len(coss) - 1)) and coss[0] > report["cos_baseline_median"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_dose_response_result_v308", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
