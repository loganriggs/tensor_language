#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_patch_back_recovers pred_c_reader_is_early pred_d_no_single_block_recovers_most pred_e_tail_cost_replays_v313
"""MLP 1: who reads the tail write? (v315). v313: restoring the census tail (4,408 units) of MLP 1 to its context-free activations costs 0.47 nats on
text; the tail's in-context write is the valuable, context-conditioned part. Which later blocks consume it? Under the tail-restore edit, patch each
later block l = 2..17's full output (attention + MLP write at every position, taken from the native run) back to native and read how much of the
0.47 nats returns. A block whose patch-back recovers a large share reads the tail write (directly or through the blocks between). Reported: the
recovery per block, its cumulative-by-depth profile, and the largest single block.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_loss_replays_native  the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_patch_back_recovers           the sum over blocks 2..17 of single-block recoveries is >= 0.50 of the tail cost (the readers are among them, not only the unembedding)
    pred_c_reader_is_early               blocks 2-5 together recover more than blocks 6-17 together
    pred_d_no_single_block_recovers_most no single block recovers >= 0.50 of the tail cost (the write is read in several places). Prior: unsure.
    pred_e_tail_cost_replays_v313        the tail-restore cost replays v313's 0.473 within 0.01
PRICE (registered maximum): <= 12 table batches + 2 native + 2 census + 2 natural batches x (1 manual + 1 tail + 16 patch-backs) = 52 forwards; 0 backwards; 0 fits. Bar <= 56.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_tail_reader_v315_result.json"
CANDIDATE_ID = "mlp1.token_table.tail_reader_v315"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
HEADK, BATCH = 200, 256
REPLAY_TOL, SUM_MIN, SINGLE_MAX, TAIL_COST, TAIL_TOL = 1e-4, 0.50, 0.50, 0.4729, 0.01
FORWARDS_MAX = 56
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_patch_back_recovers": ">= 0.50 of tail cost", "pred_c_reader_is_early": "2-5 > 6-17", "pred_d_no_single_block_recovers_most": "< 0.50 each", "pred_e_tail_cost_replays_v313": "0.473 +- 0.01"}


def losses_patch(backend, tokens, units, h_alone, native_outs, patch_block):
    """per-position CE with MLP-1 `units` at positions >= 1 replaced by `h_alone`, and (if patch_block is not None) that block's attention + MLP
    writes replaced by the native run's (`native_outs[patch_block]`, a pair of tensors). With units None and patch None this is the manual native pass;
    it also returns every block's (attention, mlp) writes for use as native_outs."""
    torch, F, model = backend.torch, backend.F, backend.model; B, Tn = tokens.shape; outs = {}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            if patch_block == l: attention = native_outs[l][0]
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == 1 and units is not None:
                h = dod_units.hidden(model, block.mlp, xin); u = torch.tensor(list(units), device=h.device); h[:, 1:, u] = h_alone.to(h.dtype).to(h.device); m = block.mlp.Down(h) + block.mlp.Down_bias
            else: m = block.mlp(xin)
            if patch_block == l: m = native_outs[l][1]
            outs[l] = (attention, m); x = x + m
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1), reduction="none").view(B, Tn - 1)
    return ce.cpu(), outs


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "head_k": HEADK, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "sum_min": SUM_MIN, "single_max": SINGLE_MAX, "tail_cost": TAIL_COST, "tail_tol": TAIL_TOL}}
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
    tail = tuple(order[HEADK:].tolist()); BLOCKS = list(range(2, 18))
    ce = {"manual": [], "tail": [], **{f"patch{l}": [] for l in BLOCKS}}; native = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            c0, outs = losses_patch(backend, chunk, None, None, None, None); ce["manual"].append(c0); forwards += 1
            ha = torch.stack([h_tab[[tindex[t] for t in row[1:].tolist()]][:, torch.tensor(list(tail))] for row in chunk])
            ct, _ = losses_patch(backend, chunk, tail, ha, None, None); ce["tail"].append(ct); forwards += 1
            for l in BLOCKS:
                cp, _ = losses_patch(backend, chunk, tail, ha, outs, l); ce[f"patch{l}"].append(cp); forwards += 1
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    tail_cost = float((ce["tail"] - base).mean()); recover = {l: float((ce["tail"] - ce[f"patch{l}"]).mean()) for l in BLOCKS}; share = {l: recover[l] / tail_cost for l in BLOCKS}
    early = sum(share[l] for l in (2, 3, 4, 5)); late = sum(share[l] for l in range(6, 18)); best = max(BLOCKS, key=lambda l: share[l])
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "tail_cost": tail_cost, "recovery_share_by_block": {str(l): share[l] for l in BLOCKS}, "sum_of_shares": sum(share.values()), "early_2_5": early, "late_6_17": late, "largest_block": best, "largest_share": share[best]}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_patch_back_recovers": sum(share.values()) >= SUM_MIN, "pred_c_reader_is_early": early > late, "pred_d_no_single_block_recovers_most": share[best] < SINGLE_MAX, "pred_e_tail_cost_replays_v313": abs(tail_cost - TAIL_COST) <= TAIL_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_tail_reader_result_v315", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
