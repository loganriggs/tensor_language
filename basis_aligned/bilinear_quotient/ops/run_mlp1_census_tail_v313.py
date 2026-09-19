#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_tail_costs_more_than_head pred_c_costs_are_superadditive pred_d_tail_cost_grows_with_position pred_e_tail_per_share_exceeds_head
"""MLP 1: the tail of the census (v313). v311 / v312: restoring the top-200 net-census units' context-free activations costs 0.156 nats (0.34 nats per
unit share of the cancellation); restoring all 4,608 costs 0.70 -- twice the linear extrapolation -- and more than removing MLP 1 (0.41). So the
tail beyond the top 200 (54% of the cancellation) is worth more per share than the head. Test on the same 2,944 natural positions: restore the tail
only (all units except the top 200 by |pooled net change|, census recomputed in-script), and compare head (top 200), tail, and all: additivity
cost(head) + cost(tail) vs cost(all); per-share prices; growth with position.
PREDICTIONS (scored as written; failures preserved; priors from v311 / v312)
    pred_a_baseline_loss_replays_native   the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_tail_costs_more_than_head      cost(tail) > cost(head = top 200)
    pred_c_costs_are_superadditive        cost(all) > cost(head) + cost(tail) (the two parts interact downstream). Prior: unsure.
    pred_d_tail_cost_grows_with_position  cost(tail) over positions 12-23 exceeds that over 1-6
    pred_e_tail_per_share_exceeds_head    cost(tail) / share(tail) >= 2x cost(head) / share(head)
PRICE (registered maximum): <= 12 table batches + 2 native + 2 census + 2 natural batches x (1 manual + 3 restores) = 24 forwards; 0 backwards; 0 fits. Bar <= 26.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_census_tail_v313_result.json"
CANDIDATE_ID = "mlp1.token_table.census_tail_v313"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
HEADK, BATCH = 200, 256
REPLAY_TOL, RATIO_MIN = 1e-4, 2.0
FORWARDS_MAX = 26
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_tail_costs_more_than_head": "tail > head", "pred_c_costs_are_superadditive": "all > head + tail", "pred_d_tail_cost_grows_with_position": "late > early", "pred_e_tail_per_share_exceeds_head": ">= 2x"}


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
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "head_k": HEADK, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "ratio_min": RATIO_MIN}}
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
    delta = (That @ Dw) * (h_ctx - h_tab[ti]); pd = delta.sum(0); order = torch.argsort(pd.abs(), descending=True)
    head = tuple(order[:HEADK].tolist()); tail = tuple(order[HEADK:].tolist()); allu = tuple(range(4608))
    share = {"head": float(pd[order[:HEADK]].sum() / pd.sum()), "tail": float(pd[order[HEADK:]].sum() / pd.sum())}
    conditions = [("manual", None), ("head", head), ("tail", tail), ("all", allu)]
    ce = {name: [] for name, _ in conditions}; native = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            for name, units in conditions:
                ha = None if units is None else torch.stack([h_tab[[tindex[t] for t in row[1:].tolist()]][:, torch.tensor(list(units))] for row in chunk])
                ce[name].append(losses(backend, chunk, units, ha, "restore")); forwards += 1
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    cost = {k: float((ce[k] - base).mean()) for k in ("head", "tail", "all")}; by_pos = {k: (ce[k] - base).mean(0).tolist() for k in ("head", "tail", "all")}
    early, late = sum(by_pos["tail"][0:6]) / 6, sum(by_pos["tail"][11:23]) / 12
    per_share = {k: cost[k] / share[k] for k in ("head", "tail")}
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "cost": cost, "share": share, "cost_per_share": per_share, "head_plus_tail": cost["head"] + cost["tail"], "tail_early_1_6": early, "tail_late_12_23": late, "by_position": by_pos}
    print(json.dumps({k: v for k, v in report.items() if k != "by_position"}, indent=1))
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_tail_costs_more_than_head": cost["tail"] > cost["head"], "pred_c_costs_are_superadditive": cost["all"] > cost["head"] + cost["tail"],
                   "pred_d_tail_cost_grows_with_position": late > early, "pred_e_tail_per_share_exceeds_head": per_share["tail"] >= RATIO_MIN * per_share["head"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_census_tail_result_v313", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
