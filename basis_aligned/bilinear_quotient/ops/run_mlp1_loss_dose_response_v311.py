#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_loss_cost_grows_with_k pred_c_every_k_beats_random_sets pred_d_top200_costs_at_least_020 pred_e_cost_per_unit_of_gain_is_stable
"""MLP 1: loss dose-response of the cancellation (v311). v310: restoring the gain head {3289, 624} (22% of what context takes from the token lookup)
costs the model 0.059 nats of next-token loss on 2,944 natural positions. v308 / v309: the net census orders the units and the top 2 / 10 / 50 / 200
hold 22 / 31 / 37 / 46% of the cancellation on text. Here each set is restored (activations replaced by the single-token values at every position
>= 1) on the same text and the loss cost is read against 4 seeded random sets of the same size, per k. The net census is recomputed in-script from
the same passes (v309's ordering; no dependence on its file). Reported: cost(k), null max(k), and cost per unit of restored lookup gain.
PREDICTIONS (scored as written; failures preserved; priors from v310)
    pred_a_baseline_loss_replays_native   the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_loss_cost_grows_with_k         mean loss change is positive and increases monotonically over k = 2, 10, 50, 200
    pred_c_every_k_beats_random_sets      cost(k) > 3x the largest |change| among the 4 random k-sets, every k
    pred_d_top200_costs_at_least_020      cost(200) >= 0.20 nats (proportional extrapolation of 0.059 for 22% would give ~0.12; registered above it: unsure)
    pred_e_cost_per_unit_of_gain_is_stable  cost(k) / share(k) (share = the set's fraction of the pooled net change) varies by <= 2x across k
PRICE (registered maximum): <= 12 table batches + 2 natural batches x (1 native + 1 manual + 4 restores + 16 nulls) = 56 forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_loss_dose_response_v311_result.json"
CANDIDATE_ID = "mlp1.token_table.loss_dose_response_v311"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
KS, N_NULL, SEED, BATCH = (2, 10, 50, 200), 4, 311, 256
REPLAY_TOL, NULL_FACTOR, COST200_MIN, STABLE_MAX = 1e-4, 3.0, 0.20, 2.0
FORWARDS_MAX = 60
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_loss_cost_grows_with_k": "positive, monotone", "pred_c_every_k_beats_random_sets": "> 3x null x 4", "pred_d_top200_costs_at_least_020": ">= 0.20", "pred_e_cost_per_unit_of_gain_is_stable": "<= 2x spread"}


def losses(backend, tokens, units, h_alone, mode):
    """per-position next-token CE with MLP-1 `units` at positions >= 1 replaced by `h_alone` (mode 'restore') or zeroed ('zero'); None = manual native."""
    torch, F, model = backend.torch, backend.F, backend.model; B, Tn = tokens.shape
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == 1 and units is not None:
                h = dod_units.hidden(model, block.mlp, xin); u = torch.tensor(list(units), device=h.device)
                if mode == "zero": h[:, 1:, u] = 0
                else: h[:, 1:, u] = h_alone.to(h.dtype).to(h.device)
                x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1), reduction="none").view(B, Tn - 1)
    return ce.cpu()


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "ks": list(KS), "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "null_factor": NULL_FACTOR, "cost200_min": COST200_MIN, "stable_max": STABLE_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; mlp = model.transformer.h[1].mlp
    Dw = mlp.Down.weight.detach().float().cpu()
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); uniq = sorted(set(nat[:, 1:].reshape(-1).tolist())); tab, tabw = [], []
    for s0 in range(0, len(uniq), BATCH):
        ids = torch.tensor(uniq[s0:s0 + BATCH], device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
        tab.append(dod_units.hidden(model, mlp, F.rms_norm(c["x1"].to("cuda"), (c["x1"].shape[-1],))).float().cpu()); tabw.append(c["mlp1"])
    h_tab, T_tab = torch.cat(tab), torch.cat(tabw); tindex = {t: i for i, t in enumerate(uniq)}
    # net census from a native manual pass (same as v309)
    hs, ti_all = [], []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == 1: hs.append(dod_units.hidden(model, mlp, xin)[:, 1:].reshape(-1, 4608).float().cpu())
                x = x + block.mlp(xin)
            forwards += 1; ti_all.append(chunk[:, 1:].reshape(-1).cpu())
    h_ctx = torch.cat(hs); ti = torch.tensor([tindex[t] for t in torch.cat(ti_all).tolist()]); T = T_tab[ti]; That = T / T.norm(dim=1, keepdim=True)
    delta = (That @ Dw) * (h_ctx - h_tab[ti]); pd = delta.sum(0); order = torch.argsort(pd.abs(), descending=True); share = {k: float(pd[order[:k]].sum() / pd.sum()) for k in KS}
    rng = random.Random(SEED); sets = {k: tuple(order[:k].tolist()) for k in KS}; nulls = {k: [tuple(sorted(rng.sample(range(4608), k))) for _ in range(N_NULL)] for k in KS}
    conditions = [("manual", None)] + [(f"k{k}", sets[k]) for k in KS] + [(f"null{k}_{n_}", nulls[k][n_]) for k in KS for n_ in range(N_NULL)]
    ce = {name: [] for name, _ in conditions}; native = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            for name, units in conditions:
                ha = None if units is None else torch.stack([h_tab[[tindex[t] for t in row[1:].tolist()]][:, torch.tensor(list(units))] for row in chunk])
                ce[name].append(losses(backend, chunk, units, ha, "restore")); forwards += 1
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    cost = {k: float((ce[f"k{k}"] - base).mean()) for k in KS}; null_max = {k: max(abs(float((ce[f"null{k}_{n_}"] - base).mean())) for n_ in range(N_NULL)) for k in KS}
    per_share = {k: cost[k] / share[k] for k in KS}
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "cost_by_k": {str(k): cost[k] for k in KS}, "null_max_by_k": {str(k): null_max[k] for k in KS}, "share_by_k": {str(k): share[k] for k in KS},
              "cost_per_share_by_k": {str(k): per_share[k] for k in KS}, "top12_units": order[:12].tolist()}
    print(json.dumps(report, indent=1))
    cs = [cost[k] for k in KS]
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_loss_cost_grows_with_k": cs[0] > 0 and all(cs[i + 1] > cs[i] for i in range(len(cs) - 1)), "pred_c_every_k_beats_random_sets": all(cost[k] > NULL_FACTOR * null_max[k] for k in KS),
                   "pred_d_top200_costs_at_least_020": cost[200] >= COST200_MIN, "pred_e_cost_per_unit_of_gain_is_stable": max(per_share.values()) <= STABLE_MAX * min(per_share.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_loss_dose_response_result_v311", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
