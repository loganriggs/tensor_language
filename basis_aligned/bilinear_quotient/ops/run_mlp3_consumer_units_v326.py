#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_mlp3_only_replays_v325 pred_c_top50_units_carry_half pred_d_named_units_in_top50 pred_e_unit_edit_recovers
"""MLP 3's consumption of MLP 1's conditioned write, at unit grain (v326). v325: fed to MLP 3's input alone, MLP 1's un-conditioned write costs 65% of the
they - he margin change on the v76 rows. Which MLP-3 units? For each row the exact per-unit response change of MLP 3 to the delta at the noun,
r_j = h_j(x3 + d) - h_j(x3), and its write D_j r_j; the margin is then re-read with only the top-k responding units (by pooled |r_j| . ||D_j||) receiving the
delta (all other units native), k = 10, 50, 200: a unit-restricted version of the MLP-3-only delivery. Reported: the top-k shares of the response norm,
the rank of units 3465 / 493, and the margin change recovered by k.
PREDICTIONS (scored as written; failures preserved; priors from v296 / v325)
    pred_a_baseline_replays       native margin 2.048 +- 1e-3 and the full-delta change -0.655 +- 0.03
    pred_b_mlp3_only_replays_v325  the MLP-3-only delivery replays v325's share 0.647 within 0.03
    pred_c_top50_units_carry_half  the 50 units with the largest pooled response-write norm carry >= 0.50 of MLP 3's total response-write norm. Prior: unsure.
    pred_d_named_units_in_top50    3465 and 493 are both in that top 50
    pred_e_unit_edit_recovers      delivering the delta to the top-200 units only reproduces >= 0.60 of the MLP-3-only margin change
PRICE (registered maximum): 3 row batches x (1 native + 1 delta + 1 full + 1 mlp3-only + 1 census + 3 unit-restricted) = 24 forwards; 0 backwards; 0 fits. Bar <= 28.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_context_gain_decomposition_v289 as v289
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp3_consumer_units_v326_result.json"
CANDIDATE_ID = "mlp3.consumer_units_v326"
UNITS, DST, SRC, BATCH = (3465, 493), 3, 1, 32
NATIVE_M, M_TOL, FULL_DM, FULL_TOL, V325_MLP3, V325_TOL, TOP50_MIN, RECOVER_MIN = 2.0481, 1e-3, -0.6554, 0.03, 0.647, 0.03, 0.50, 0.60
KS = (10, 50, 200)
FORWARDS_MAX = 28
HEADS_ALL = [(l, h) for l in (0, 1) for h in range(9)]
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3; -0.655 +- 0.03", "pred_b_mlp3_only_replays_v325": "0.647 +- 0.03", "pred_c_top50_units_carry_half": ">= 0.50", "pred_d_named_units_in_top50": "3465 and 493", "pred_e_unit_edit_recovers": ">= 0.60"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "src": SRC, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "full_dm": FULL_DM, "full_tol": FULL_TOL, "v325_mlp3": V325_MLP3, "v325_tol": V325_TOL, "top50_min": TOP50_MIN, "recover_min": RECOVER_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0
    blocks = model.transformer.h; scale = 1.0
    for l in range(SRC + 1, DST + 1): scale *= float(blocks[l].lambdas[0])
    def wrap(l):
        orig = blocks[l].attn.squared_attention
        def f(q, k, v, q2, k2):
            B, T, H, D = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0)
            pat = torch.diag_embed(torch.diagonal(pat, dim1=-2, dim2=-1)); return torch.einsum("bhqk,bkhd->bhqd", pat, v)
        return f
    def scale_to(l):
        sc = 1.0
        for j in range(2, l + 1): sc *= float(blocks[j].lambdas[0])
        return sc
    def delta_for(tokens):
        """(un-conditioned MLP-1 write) - (native MLP-1 write) at every position: MLP 1 fed the parallel self-only stream."""
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; xs, v1s = x, None
            for l in (0, 1):
                block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                lives = block.lambdas[0] * xs + block.lambdas[1] * x0; orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
                try: atts, v1s = block.attn(F.rms_norm(lives, (model.config.n_embd,)), v1s)
                finally: block.attn.squared_attention = orig
                xs = lives + atts
                if l == 1: return block.mlp(F.rms_norm(xs, (model.config.n_embd,))) - block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,))); xs = xs + block.mlp(F.rms_norm(xs, (model.config.n_embd,)))
    def margins_with(tokens, chunk, delta, consumer):
        """consumer: None (native), "full" (delta added to the residual at block 1's output), int l (delta only to block l's attention + MLP inputs), "direct"."""
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                if isinstance(consumer, tuple) and consumer[0] == "units" and consumer[1] == l:
                    d = scale_to(l) * delta; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                    h0 = dod_units.hidden(model, block.mlp, F.rms_norm(x, (model.config.n_embd,))); h1 = dod_units.hidden(model, block.mlp, F.rms_norm(x + d, (model.config.n_embd,)))
                    h = h0.clone(); h[:, :, consumer[2]] = h1[:, :, consumer[2]]; m = block.mlp.Down(h) + block.mlp.Down_bias
                elif isinstance(consumer, tuple) and consumer[0] == l:
                    d = scale_to(l) * delta; part = consumer[1]
                    attention, v1_ = block.attn(F.rms_norm(live + (d if part in ("attn", "both") else 0.0), (model.config.n_embd,)), v1_); x = live + attention
                    m = block.mlp(F.rms_norm(x + (d if part in ("mlp", "both") else 0.0), (model.config.n_embd,)))
                else:
                    attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + m
                if consumer == "full" and l == 1: x = x + delta
            xf = x + (scale_to(17) * delta if consumer == "direct" else 0.0)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(xf, (model.config.n_embd,))) / 30)
        return [float(logits[i, row.final, L._single(" they")] - logits[i, row.final, L._single(" he")]) for i, row in enumerate(chunk)]
    out = {"native": [], "full": [], "mlp3": [], **{f"k{k}": [] for k in KS}}; resp_pool = torch.zeros(4608); Dn = blocks[3].mlp.Down.weight.detach().float().cpu().norm(dim=0)
    # pass 1: census of MLP 3's per-unit response to the delta at the noun
    chunks = [rows[start:start + BATCH] for start in range(0, len(rows), BATCH)]; deltas = []
    for chunk in chunks:
        tokens = fw._tokens(chunk); delta = delta_for(tokens); forwards += 1; deltas.append(delta); pos = torch.tensor([noun_of(r) for r in chunk], device=tokens.device); idx = torch.arange(len(chunk))
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                if l == 3:
                    d = scale_to(3) * delta; h0 = dod_units.hidden(model, block.mlp, F.rms_norm(x, (model.config.n_embd,))); h1 = dod_units.hidden(model, block.mlp, F.rms_norm(x + d, (model.config.n_embd,)))
                    resp_pool += ((h1 - h0)[idx, pos].float().cpu().abs() * Dn).sum(0); break
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
        forwards += 1
    order = torch.argsort(resp_pool, descending=True); shares = {str(k): float(resp_pool[order[:k]].sum() / resp_pool.sum()) for k in (10, 50, 200, 500)}; ranks = {str(u): int((resp_pool > resp_pool[u]).sum()) + 1 for u in (3465, 493)}
    for chunk, delta in zip(chunks, deltas):
        tokens = fw._tokens(chunk)
        out["native"] += margins_with(tokens, chunk, None, None); forwards += 1; out["full"] += margins_with(tokens, chunk, delta, "full"); forwards += 1
        out["mlp3"] += margins_with(tokens, chunk, delta, (3, "mlp")); forwards += 1
        for k in KS: out[f"k{k}"] += margins_with(tokens, chunk, delta, ("units", 3, order[:k].to(tokens.device))); forwards += 1
    def oriented(m): return sum((v if row.present else -v) for v, row in zip(m, rows)) / len(rows)
    base = oriented(out["native"]); full = oriented(out["full"]) - base; mlp3 = (oriented(out["mlp3"]) - base) / full; byk = {str(k): (oriented(out[f"k{k}"]) - base) / full for k in KS}
    report = {"native_margin": base, "full_delta_change": full, "mlp3_only_share": mlp3, "unit_restricted_share_by_k": byk, "recovered_fraction_by_k": {k: v / mlp3 for k, v in byk.items()}, "response_top_shares": shares, "named_unit_ranks": ranks, "top12_units": order[:12].tolist()}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": abs(base - NATIVE_M) <= M_TOL and abs(full - FULL_DM) <= FULL_TOL, "pred_b_mlp3_only_replays_v325": abs(mlp3 - V325_MLP3) <= V325_TOL, "pred_c_top50_units_carry_half": shares["50"] >= TOP50_MIN,
                   "pred_d_named_units_in_top50": all(r <= 50 for r in ranks.values()), "pred_e_unit_edit_recovers": byk["200"] / mlp3 >= RECOVER_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_consumer_units_result_v326", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
