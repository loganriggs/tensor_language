#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_self_only_attention_costs_within_2x_of_table pred_c_block1_self_only_costs_more_than_block0 pred_d_self_only_cost_grows_with_position pred_e_mlp1_write_under_self_only_is_the_table
"""MLP 1: does attention 0 / 1 alone do the conversion? (v314). v298: with every head of blocks 0 and 1 reading only its own key, MLP 1's write IS the
token's table entry (alpha = 1.00). v312: replacing MLP 1's write by the table entry everywhere costs 0.70 nats on text. If attention 0 / 1's context
reading is what converts the lookup into the usable write, then making all 18 heads self-only should cost about the same as the table-everywhere edit
(it also removes attention's own direct context writes into the residual, so the cost may exceed 0.70). Conditions on the 2,944 natural positions:
all 18 heads self-only; block-0 heads only; block-1 heads only. Also checked: under all-self-only, MLP 1's write equals the table entry (alpha ~ 1).
PREDICTIONS (scored as written; failures preserved; priors from v298 / v312)
    pred_a_baseline_loss_replays_native            the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_self_only_attention_costs_within_2x_of_table  cost(all self-only) is within 2x of 0.70 nats (0.35-1.40)
    pred_c_block1_self_only_costs_more_than_block0  cost(block-1 self-only) > cost(block-0 self-only) (block 1 carries most of x1's identity, v300). Prior: unsure.
    pred_d_self_only_cost_grows_with_position      cost(all self-only) over positions 12-23 exceeds that over 1-6
    pred_e_mlp1_write_under_self_only_is_the_table  median alpha of MLP 1's write under all-self-only >= 0.95 at the natural positions
PRICE (registered maximum): <= 12 table batches + 2 natural batches x (1 native + 1 manual + 3 edits) = 22 forwards; 0 backwards; 0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_self_only_attention_loss_v314_result.json"
CANDIDATE_ID = "mlp1.token_table.self_only_attention_loss_v314"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
BATCH = 256
REPLAY_TOL, TABLE_COST, FACTOR, ALPHA_MIN = 1e-4, 0.70, 2.0, 0.95
FORWARDS_MAX = 24
HEADS = {"all": [(l, h) for l in (0, 1) for h in range(9)], "block0": [(0, h) for h in range(9)], "block1": [(1, h) for h in range(9)]}
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_self_only_attention_costs_within_2x_of_table": "0.35-1.40", "pred_c_block1_self_only_costs_more_than_block0": "block1 > block0", "pred_d_self_only_cost_grows_with_position": "late > early", "pred_e_mlp1_write_under_self_only_is_the_table": "alpha >= 0.95"}


def losses_self_only(backend, tokens, self_only):
    """per-position next-token CE with the listed heads of blocks 0/1 reading only their own key; also returns MLP 1's write at positions >= 1."""
    torch, F, model = backend.torch, backend.F, backend.model; B, Tn = tokens.shape; blocks = model.transformer.h
    def wrap(l):
        orig = blocks[l].attn.squared_attention
        def f(q, k, v, q2, k2):
            Bq, T, H, D = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0)
            for (ll, h) in self_only:
                if ll == l: pat[:, h] = torch.diag_embed(torch.diagonal(pat[:, h], dim1=-2, dim2=-1))
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)
        return f
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; w1 = None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            if l in (0, 1) and self_only:
                orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
                try: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
                finally: block.attn.squared_attention = orig
            else: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            x = live + attention; m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            if l == 1: w1 = m[:, 1:].reshape(-1, m.shape[-1]).float().cpu()
            x = x + m
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1), reduction="none").view(B, Tn - 1)
    return ce.cpu(), w1


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "table_cost": TABLE_COST, "factor": FACTOR, "alpha_min": ALPHA_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; mlp = model.transformer.h[1].mlp
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); uniq = sorted(set(nat[:, 1:].reshape(-1).tolist())); tabw = []
    for s0 in range(0, len(uniq), BATCH):
        ids = torch.tensor(uniq[s0:s0 + BATCH], device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
        tabw.append(c["mlp1"])
    T_tab = torch.cat(tabw); tindex = {t: i for i, t in enumerate(uniq)}
    ce = {"manual": [], "all": [], "block0": [], "block1": []}; native, w_all, ti_all = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            c0, _ = losses_self_only(backend, chunk, []); ce["manual"].append(c0); forwards += 1
            for name in ("all", "block0", "block1"):
                c_, w_ = losses_self_only(backend, chunk, HEADS[name]); ce[name].append(c_); forwards += 1
                if name == "all": w_all.append(w_); ti_all.append(chunk[:, 1:].reshape(-1).cpu())
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    W = torch.cat(w_all); T = T_tab[torch.tensor([tindex[t] for t in torch.cat(ti_all).tolist()])]; alpha = (W * T).sum(1) / (T * T).sum(1)
    cost = {k: float((ce[k] - base).mean()) for k in ("all", "block0", "block1")}; by_pos = {k: (ce[k] - base).mean(0).tolist() for k in cost}
    early, late = sum(by_pos["all"][0:6]) / 6, sum(by_pos["all"][11:23]) / 12
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "cost": cost, "all_early_1_6": early, "all_late_12_23": late, "alpha_under_all_self_only_median": float(alpha.median()), "alpha_p10_p90": [float(alpha.quantile(0.1)), float(alpha.quantile(0.9))], "by_position": by_pos}
    print(json.dumps({k: v for k, v in report.items() if k != "by_position"}, indent=1))
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_self_only_attention_costs_within_2x_of_table": TABLE_COST / FACTOR <= cost["all"] <= TABLE_COST * FACTOR, "pred_c_block1_self_only_costs_more_than_block0": cost["block1"] > cost["block0"],
                   "pred_d_self_only_cost_grows_with_position": late > early, "pred_e_mlp1_write_under_self_only_is_the_table": float(alpha.median()) >= ALPHA_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_self_only_attention_loss_result_v314", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
