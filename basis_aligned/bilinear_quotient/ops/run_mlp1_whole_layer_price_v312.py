#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_full_restore_matches_linear_price pred_c_zeroing_mlp1_costs_more_than_restoring pred_d_restore_cost_grows_with_position pred_e_table_alone_beats_nothing_at_every_position
"""MLP 1: the whole-layer price (v312). v311: undoing the context cancellation unit by unit costs a stable ~0.34 nats per unit share on text, which
extrapolates to ~0.34 nats for the whole layer. Closing test on the same 2,944 natural positions: (restore) replace MLP 1's write at every position
>= 1 by the token's context-free table entry (all 4,608 units given their single-token activations) and (zero) remove MLP 1's write entirely
(Down bias kept). The manual forward replays the native loss.
PREDICTIONS (scored as written; failures preserved; priors from v311)
    pred_a_baseline_loss_replays_native   the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_full_restore_matches_linear_price  the full-restore cost is within 2x of 0.34 nats (0.17-0.68)
    pred_c_zeroing_mlp1_costs_more_than_restoring  zeroing MLP 1 costs more than the context-free table (the lookup, even uncancelled, is better than nothing). Prior: unsure.
    pred_d_restore_cost_grows_with_position  the full-restore cost over positions 12-23 exceeds that over positions 1-6
    pred_e_table_alone_beats_nothing_at_every_position  the zeroing cost exceeds the restore cost at every one of the 23 positions
PRICE (registered maximum): <= 12 table batches + 2 natural batches x (1 native + 1 manual + 1 restore + 1 zero) = 20 forwards; 0 backwards; 0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_whole_layer_price_v312_result.json"
CANDIDATE_ID = "mlp1.token_table.whole_layer_price_v312"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
BATCH = 256
REPLAY_TOL, PRICE, PRICE_FACTOR = 1e-4, 0.34, 2.0
FORWARDS_MAX = 24
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_full_restore_matches_linear_price": "0.17-0.68", "pred_c_zeroing_mlp1_costs_more_than_restoring": "zero > restore", "pred_d_restore_cost_grows_with_position": "late > early", "pred_e_table_alone_beats_nothing_at_every_position": "23/23"}


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
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "price": PRICE, "price_factor": PRICE_FACTOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; mlp = model.transformer.h[1].mlp
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); uniq = sorted(set(nat[:, 1:].reshape(-1).tolist())); tab = []
    for s0 in range(0, len(uniq), BATCH):
        ids = torch.tensor(uniq[s0:s0 + BATCH], device="cuda").unsqueeze(1); c = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
        tab.append(dod_units.hidden(model, mlp, F.rms_norm(c["x1"].to("cuda"), (c["x1"].shape[-1],))).float().cpu())
    h_tab = torch.cat(tab); tindex = {t: i for i, t in enumerate(uniq)}; ALL = tuple(range(4608))
    ce = {"manual": [], "restore": [], "zero": []}; native = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            ha = torch.stack([h_tab[[tindex[t] for t in row[1:].tolist()]] for row in chunk])
            ce["manual"].append(losses(backend, chunk, None, None, None)); ce["restore"].append(losses(backend, chunk, ALL, ha, "restore")); ce["zero"].append(losses(backend, chunk, ALL, None, "zero")); forwards += 3
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    cost = {k: float((ce[k] - base).mean()) for k in ("restore", "zero")}; by_pos = {k: (ce[k] - base).mean(0).tolist() for k in ("restore", "zero")}
    early, late = sum(by_pos["restore"][0:6]) / 6, sum(by_pos["restore"][11:23]) / 12
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "cost_restore": cost["restore"], "cost_zero": cost["zero"], "restore_early_1_6": early, "restore_late_12_23": late, "by_position": by_pos,
              "zero_beats_restore_positions": int(sum(z > r for z, r in zip(by_pos["zero"], by_pos["restore"])))}
    print(json.dumps({k: v for k, v in report.items() if k != "by_position"}, indent=1))
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_full_restore_matches_linear_price": PRICE / PRICE_FACTOR <= cost["restore"] <= PRICE * PRICE_FACTOR, "pred_c_zeroing_mlp1_costs_more_than_restoring": cost["zero"] > cost["restore"],
                   "pred_d_restore_cost_grows_with_position": late > early, "pred_e_table_alone_beats_nothing_at_every_position": report["zero_beats_restore_positions"] == Tn - 1}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_whole_layer_price_result_v312", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
