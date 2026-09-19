#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_injections_sum_near_total pred_c_direct_path_to_unembedding_small pred_d_reader_is_early pred_e_largest_single_reader_named
"""MLP 1: who reads the tail write? Path-restricted injection (v316). v315: block patch-back cannot localise the reader (cascade through the lambda
recurrence). Here the corrupted MLP-1 write (tail units restored to their context-free activations) is delivered to ONE consumer at a time, with
everything else native: for block l = 2..17, block l's attention and MLP read an input that carries the corrupted MLP-1 write (native residual +
delta at the block's input), while the residual stream itself and every other block stay native; a final condition delivers delta only to the
unembedding (the direct path). delta = W_tail-restored - W_native at every position >= 1, propagated to block l's input with the lambda chain
(exact for the direct residual path). Reported: loss change per consumer; their sum vs the full tail cost (0.473); the direct path; early vs late.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_loss_replays_native   the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_injections_sum_near_total      the per-consumer loss changes (blocks 2..17 + direct) sum to within 2x of the full tail cost (first-order additivity; 0.24-0.95)
    pred_c_direct_path_to_unembedding_small  the direct-to-unembedding delivery changes the loss by <= 0.25 of the full tail cost
    pred_d_reader_is_early                blocks 2-5 together take more of the loss than blocks 6-17 together
    pred_e_largest_single_reader_named    one block takes >= 0.25 of the full tail cost (a nameable principal reader). Prior: unsure.
PRICE (registered maximum): <= 12 table batches + 2 native + 2 census + 2 natural batches x (1 manual + 1 tail + 17 injections) = 54 forwards; 0 backwards; 0 fits. Bar <= 58.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_tail_reader_paths_v316_result.json"
CANDIDATE_ID = "mlp1.token_table.tail_reader_paths_v316"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
HEADK, BATCH = 200, 256
REPLAY_TOL, SUM_LO, SUM_HI, DIRECT_MAX, SINGLE_MIN = 1e-4, 0.5, 2.0, 0.25, 0.25
FORWARDS_MAX = 58
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_injections_sum_near_total": "0.5x-2x of tail cost", "pred_c_direct_path_to_unembedding_small": "<= 0.25", "pred_d_reader_is_early": "2-5 > 6-17", "pred_e_largest_single_reader_named": ">= 0.25"}


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


def mlp1_delta(backend, tokens, units, h_alone):
    """delta = (MLP 1 write with `units` restored) - (native MLP 1 write), all positions; [B, T, D]."""
    torch, F, model = backend.torch, backend.F, backend.model
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l in (0, 1):
            block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            m = block.mlp(xin)
            if l == 1:
                h = dod_units.hidden(model, block.mlp, xin); u = torch.tensor(list(units), device=h.device); h[:, 1:, u] = h_alone.to(h.dtype).to(h.device); m2 = block.mlp.Down(h) + block.mlp.Down_bias; return (m2 - m)
            x = x + m


def losses_inject(backend, tokens, delta, consumer):
    """native forward; the MLP-1 delta (scaled by the lambda chain from block 2 to the consumer's input) is added ONLY to what `consumer` reads:
    consumer = block index l (its attention and MLP inputs) or "direct" (the unembedding's input); the residual stream stays native."""
    torch, F, model = backend.torch, backend.F, backend.model; B, Tn = tokens.shape; blocks = model.transformer.h
    def scale_to(l):
        s = 1.0
        for j in range(2, l + 1): s *= float(blocks[j].lambdas[0])
        return s
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            if consumer == l and delta is not None:
                d = scale_to(l) * delta; attention, v1_ = block.attn(F.rms_norm(live + d, (model.config.n_embd,)), v1_); x = live + attention; m = block.mlp(F.rms_norm(x + d, (model.config.n_embd,)))
            else:
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            x = x + m
        xf = x + (scale_to(17) * delta if (consumer == "direct" and delta is not None) else 0.0)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(xf, (model.config.n_embd,))) / 30)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1), reduction="none").view(B, Tn - 1)
    return ce.cpu()


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "head_k": HEADK, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "sum_lo": SUM_LO, "sum_hi": SUM_HI, "direct_max": DIRECT_MAX, "single_min": SINGLE_MIN}}
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
    tail = tuple(order[HEADK:].tolist()); BLOCKS = list(range(2, 18)); CONS = BLOCKS + ["direct"]
    ce = {"manual": [], "tail": [], **{f"c{c}": [] for c in CONS}}; native = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            ce["manual"].append(losses_inject(backend, chunk, None, None)); forwards += 1
            ha = torch.stack([h_tab[[tindex[t] for t in row[1:].tolist()]][:, torch.tensor(list(tail))] for row in chunk])
            ce["tail"].append(losses(backend, chunk, tail, ha, "restore")); forwards += 1
            delta = mlp1_delta(backend, chunk, tail, ha)
            for c in CONS: ce[f"c{c}"].append(losses_inject(backend, chunk, delta, c)); forwards += 1
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    tail_cost = float((ce["tail"] - base).mean()); take = {str(c): float((ce[f"c{c}"] - base).mean()) for c in CONS}; share = {k: v / tail_cost for k, v in take.items()}
    early = sum(share[str(l)] for l in (2, 3, 4, 5)); late = sum(share[str(l)] for l in range(6, 18)); best = max(BLOCKS, key=lambda l: share[str(l)])
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "tail_cost": tail_cost, "take_share_by_consumer": share, "sum_of_shares": sum(share.values()), "direct_share": share["direct"], "early_2_5": early, "late_6_17": late, "largest_block": best, "largest_share": share[str(best)]}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_injections_sum_near_total": SUM_LO <= sum(share.values()) <= SUM_HI, "pred_c_direct_path_to_unembedding_small": share["direct"] <= DIRECT_MAX, "pred_d_reader_is_early": early > late, "pred_e_largest_single_reader_named": share[str(best)] >= SINGLE_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_tail_reader_paths_result_v316", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
