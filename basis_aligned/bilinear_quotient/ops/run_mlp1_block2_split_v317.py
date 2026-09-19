#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_loss_replays_native pred_b_block2_replays_v316 pred_c_mlp2_takes_more_than_attn2 pred_d_attn2_and_mlp2_add pred_e_mlp2_unit_census_is_spread
"""MLP 1 -> block 2: attention 2 or MLP 2? (v317). v316: delivering MLP 1's un-conditioned (tail-restored) write to block 2's input alone takes 45% of
the 0.473-nat cost at first order; no other consumer past block 3 takes any. Here block 2 is split: the delta delivered to attention 2's input only,
to MLP 2's input only, or to both (v316's block-2 condition), all else native. Also the per-unit first-order response of MLP 2 to the delta at the
block-2 input (the exact product-level term of dod_units.product_unit_census: (L_j . d)(R_j . x^) + (L_j . x^)(R_j . d) - (L_j . d)(R_j . d), over rms^2),
projected on MLP 2's Down columns' loss gradient is NOT taken (no backward); instead the census reports the norm of each unit's response change,
pooled over positions, and its top-k concentration.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_loss_replays_native  the manual forward's mean loss equals the model's native forward within 1e-4
    pred_b_block2_replays_v316           the both-inputs delivery takes 0.45 +- 0.03 of the tail cost
    pred_c_mlp2_takes_more_than_attn2    the MLP-2-only delivery takes more of the cost than the attention-2-only delivery
    pred_d_attn2_and_mlp2_add            attn-only + mlp-only is within 0.10 (share) of the both-inputs value
    pred_e_mlp2_unit_census_is_spread    the 200 units of MLP 2 with the largest pooled |response change| carry <= 0.50 of the total (the reading is layer-wide, like MLP 1's writing)
PRICE (registered maximum): <= 12 table batches + 2 native + 2 census + 2 natural batches x (1 manual + 1 tail + 3 deliveries + 1 census pass) = 28 forwards; 0 backwards; 0 fits. Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_block2_split_v317_result.json"
CANDIDATE_ID = "mlp1.token_table.block2_split_v317"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
HEADK, BATCH = 200, 256
REPLAY_TOL, B2_SHARE, B2_TOL, ADD_TOL, TOP200_MAX = 1e-4, 0.4519, 0.03, 0.10, 0.50
FORWARDS_MAX = 30
PREDICTIONS = {"pred_a_baseline_loss_replays_native": "<= 1e-4", "pred_b_block2_replays_v316": "0.45 +- 0.03", "pred_c_mlp2_takes_more_than_attn2": "mlp > attn", "pred_d_attn2_and_mlp2_add": "within 0.10", "pred_e_mlp2_unit_census_is_spread": "<= 0.50"}


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
    if not hasattr(losses_inject, "census"): losses_inject.census = []
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
            if isinstance(consumer, tuple) and consumer[0] == l and delta is not None:
                d = scale_to(l) * delta; part = consumer[1]
                attention, v1_ = block.attn(F.rms_norm(live + (d if part in ("attn", "both") else 0.0), (model.config.n_embd,)), v1_); x = live + attention
                xin0 = F.rms_norm(x, (model.config.n_embd,)); xin1 = F.rms_norm(x + d, (model.config.n_embd,)); m = block.mlp(xin1 if part in ("mlp", "both") else xin0)
                if part == "census":
                    h0 = dod_units.hidden(model, block.mlp, xin0); h1 = dod_units.hidden(model, block.mlp, xin1); losses_inject.census.append((h1 - h0)[:, 1:].reshape(-1, h0.shape[-1]).abs().sum(0).float().cpu()); m = block.mlp(xin0)
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
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "b2_share": B2_SHARE, "b2_tol": B2_TOL, "add_tol": ADD_TOL, "top200_max": TOP200_MAX}}
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
    tail = tuple(order[HEADK:].tolist()); CONS = [(2, "attn"), (2, "mlp"), (2, "both"), (2, "census")]
    ce = {"manual": [], "tail": [], **{f"c{c[1]}": [] for c in CONS}}; native = []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; B, Tn = chunk.shape
            nl = model(chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()); forwards += 1; native.append(torch.full((B, Tn - 1), float(nl if not isinstance(nl, tuple) else nl[-1])))
            ce["manual"].append(losses_inject(backend, chunk, None, None)); forwards += 1
            ha = torch.stack([h_tab[[tindex[t] for t in row[1:].tolist()]][:, torch.tensor(list(tail))] for row in chunk])
            ce["tail"].append(losses(backend, chunk, tail, ha, "restore")); forwards += 1
            delta = mlp1_delta(backend, chunk, tail, ha)
            for c in CONS: ce[f"c{c[1]}"].append(losses_inject(backend, chunk, delta, c)); forwards += 1
    native = torch.cat(native); ce = {k: torch.cat(v) for k, v in ce.items()}; base = ce["manual"]; replay = float((base.mean() - native.mean()).abs())
    tail_cost = float((ce["tail"] - base).mean()); share = {c[1]: float((ce[f"c{c[1]}"] - base).mean()) / tail_cost for c in CONS[:3]}
    resp = sum(losses_inject.census); order2 = torch.argsort(resp, descending=True); top = {str(k): float(resp[order2[:k]].sum() / resp.sum()) for k in (10, 50, 200, 500)}
    report = {"native_loss": float(native.mean()), "manual_loss": float(base.mean()), "replay_gap": replay, "tail_cost": tail_cost, "share_by_part": share, "attn_plus_mlp": share["attn"] + share["mlp"], "mlp2_response_top_shares": top, "mlp2_top12_units": order2[:12].tolist()}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_loss_replays_native": replay <= REPLAY_TOL, "pred_b_block2_replays_v316": abs(share["both"] - B2_SHARE) <= B2_TOL, "pred_c_mlp2_takes_more_than_attn2": share["mlp"] > share["attn"],
                   "pred_d_attn2_and_mlp2_add": abs(share["attn"] + share["mlp"] - share["both"]) <= ADD_TOL, "pred_e_mlp2_unit_census_is_spread": top["200"] <= TOP200_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_block2_split_result_v317", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
