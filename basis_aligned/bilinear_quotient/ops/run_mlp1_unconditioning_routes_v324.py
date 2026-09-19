#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_full_delta_replays_v323 pred_c_first_order_sum_within_2x pred_d_block2_or_3_is_principal pred_e_direct_path_small
"""Who carries MLP 1's un-conditioning to the pronoun-number margin? (v324). v323: feeding MLP 1 a self-only-attention input (its write becomes the
table entry) costs the they - he margin 32% on the v76 rows while MLP 1's carriage into MLP-3 units 3465 / 493 stays flat. v316's instrument with the
margin as readout: delta = (MLP 1's un-conditioned write) - (native write) at every position, delivered with the lambda chain to ONE consumer's input at a
time (blocks 2..17, and the direct path to the unembedding), residual and all other blocks native. Reported: the oriented margin change per consumer as a
share of the full-delta change; their first-order sum; the principal consumer.
PREDICTIONS (scored as written; failures preserved; priors from v316 / v323)
    pred_a_baseline_replays            native oriented margin replays v323's 2.048 within 1e-3
    pred_b_full_delta_replays_v323     adding the full delta to the residual (all consumers) changes the margin by -0.655 +- 0.03
    pred_c_first_order_sum_within_2x   the per-consumer changes sum to within 0.5x-2x of the full change
    pred_d_block2_or_3_is_principal    the largest single consumer share is block 2 or block 3 (as on text, v316). Prior: unsure -- the pronoun rows may use a later reader.
    pred_e_direct_path_small           the direct-to-unembedding share is <= 0.25
PRICE (registered maximum): 1 table batch + 3 row batches x (1 native + 1 delta pass + 1 full + 17 consumers) = 61 forwards; 0 backwards; 0 fits. Bar <= 64.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_context_gain_decomposition_v289 as v289
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_unconditioning_routes_v324_result.json"
CANDIDATE_ID = "mlp1.token_table.unconditioning_routes_v324"
UNITS, DST, SRC, BATCH = (3465, 493), 3, 1, 32
NATIVE_M, M_TOL, FULL_DM, FULL_TOL, SUM_LO, SUM_HI, DIRECT_MAX = 2.0481, 1e-3, -0.6554, 0.03, 0.5, 2.0, 0.25
FORWARDS_MAX = 64
HEADS_ALL = [(l, h) for l in (0, 1) for h in range(9)]
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_full_delta_replays_v323": "-0.655 +- 0.03", "pred_c_first_order_sum_within_2x": "0.5x-2x", "pred_d_block2_or_3_is_principal": "block 2 or 3", "pred_e_direct_path_small": "<= 0.25"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "src": SRC, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "full_dm": FULL_DM, "full_tol": FULL_TOL, "sum_lo": SUM_LO, "sum_hi": SUM_HI, "direct_max": DIRECT_MAX}}
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
                if consumer == l:
                    d = scale_to(l) * delta; attention, v1_ = block.attn(F.rms_norm(live + d, (model.config.n_embd,)), v1_); x = live + attention; m = block.mlp(F.rms_norm(x + d, (model.config.n_embd,)))
                else:
                    attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                x = x + m
                if consumer == "full" and l == 1: x = x + delta
            xf = x + (scale_to(17) * delta if consumer == "direct" else 0.0)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(xf, (model.config.n_embd,))) / 30)
        return [float(logits[i, row.final, L._single(" they")] - logits[i, row.final, L._single(" he")]) for i, row in enumerate(chunk)]
    CONS = list(range(2, 18)) + ["direct"]; out = {"native": [], "full": [], **{f"c{c}": [] for c in CONS}}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); delta = delta_for(tokens); forwards += 1
        out["native"] += margins_with(tokens, chunk, None, None); forwards += 1; out["full"] += margins_with(tokens, chunk, delta, "full"); forwards += 1
        for c in CONS: out[f"c{c}"] += margins_with(tokens, chunk, delta, c); forwards += 1
    def oriented(m): return sum((v if row.present else -v) for v, row in zip(m, rows)) / len(rows)
    base = oriented(out["native"]); full = oriented(out["full"]) - base; take = {str(c): oriented(out[f"c{c}"]) - base for c in CONS}; share = {k: v / full for k, v in take.items()}
    best = max((str(c) for c in CONS if c != "direct"), key=lambda k: share[k])
    report = {"native_margin": base, "full_delta_change": full, "take_by_consumer": take, "share_by_consumer": share, "sum_of_shares": sum(share.values()), "principal_block": best, "principal_share": share[best], "direct_share": share["direct"]}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": abs(base - NATIVE_M) <= M_TOL, "pred_b_full_delta_replays_v323": abs(full - FULL_DM) <= FULL_TOL, "pred_c_first_order_sum_within_2x": SUM_LO <= sum(share.values()) <= SUM_HI, "pred_d_block2_or_3_is_principal": best in ("2", "3"), "pred_e_direct_path_small": share["direct"] <= DIRECT_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_unconditioning_routes_result_v324", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
