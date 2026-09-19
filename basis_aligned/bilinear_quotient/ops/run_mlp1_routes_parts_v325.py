#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_parts_replay_v324_blocks pred_c_mlp_parts_exceed_attention_parts pred_d_parts_additive pred_e_mlp2_is_the_largest_part
"""Blocks 2-4 split into attention and MLP consumers (v325). v324: on the pronoun rows MLP 1's un-conditioning reaches the they - he margin through
block 2 (0.23), block 4 (0.15), block 3 (0.12) and block 5 (0.05) at first order. v317 (text, loss): within block 2 the MLP took 0.35 and attention 0.08.
Here, for blocks 2, 3 and 4, the delta is delivered to the attention input only, the MLP input only, or both (= v324's per-block condition), with
the residual and all other blocks native; readout the oriented margin change as a share of the full-delta change (-0.655).
PREDICTIONS (scored as written; failures preserved; priors from v317 / v324)
    pred_a_baseline_replays               native oriented margin 2.048 +- 1e-3 and full-delta change -0.655 +- 0.03
    pred_b_parts_replay_v324_blocks       the both-inputs shares replay v324's 0.229 / 0.116 / 0.151 for blocks 2 / 3 / 4 within 0.02
    pred_c_mlp_parts_exceed_attention_parts  for each of blocks 2, 3, 4 the MLP-only share exceeds the attention-only share
    pred_d_parts_additive                 attention-only + MLP-only is within 0.05 of both-inputs for each block
    pred_e_mlp2_is_the_largest_part       MLP 2's share is the largest of the six parts. Prior: unsure -- block 4 was larger than block 3 on the rows.
PRICE (registered maximum): 3 row batches x (1 native + 1 delta + 1 full + 9 parts) = 36 forwards; 0 backwards; 0 fits. Bar <= 40.
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
OUT = ROOT / "circuits/followups/mlp1_routes_parts_v325_result.json"
CANDIDATE_ID = "mlp1.token_table.routes_parts_v325"
UNITS, DST, SRC, BATCH = (3465, 493), 3, 1, 32
NATIVE_M, M_TOL, FULL_DM, FULL_TOL, BLOCK_TOL, ADD_TOL = 2.0481, 1e-3, -0.6554, 0.03, 0.02, 0.05
V324 = {2: 0.2293, 3: 0.116, 4: 0.151}
FORWARDS_MAX = 40
HEADS_ALL = [(l, h) for l in (0, 1) for h in range(9)]
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3; -0.655 +- 0.03", "pred_b_parts_replay_v324_blocks": "within 0.02 x 3", "pred_c_mlp_parts_exceed_attention_parts": "mlp > attn x 3", "pred_d_parts_additive": "within 0.05 x 3", "pred_e_mlp2_is_the_largest_part": "MLP 2 largest"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "src": SRC, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "full_dm": FULL_DM, "full_tol": FULL_TOL, "block_tol": BLOCK_TOL, "add_tol": ADD_TOL}}
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
                if isinstance(consumer, tuple) and consumer[0] == l:
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
    CONS = [(l, part) for l in (2, 3, 4) for part in ("attn", "mlp", "both")]; out = {"native": [], "full": [], **{f"c{l}{part}": [] for l, part in CONS}}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); delta = delta_for(tokens); forwards += 1
        out["native"] += margins_with(tokens, chunk, None, None); forwards += 1; out["full"] += margins_with(tokens, chunk, delta, "full"); forwards += 1
        for l, part in CONS: out[f"c{l}{part}"] += margins_with(tokens, chunk, delta, (l, part)); forwards += 1
    def oriented(m): return sum((v if row.present else -v) for v, row in zip(m, rows)) / len(rows)
    base = oriented(out["native"]); full = oriented(out["full"]) - base; share = {f"{l}.{part}": (oriented(out[f"c{l}{part}"]) - base) / full for l, part in CONS}
    largest = max(share, key=share.get)
    report = {"native_margin": base, "full_delta_change": full, "share": share, "largest_part": largest}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": abs(base - NATIVE_M) <= M_TOL and abs(full - FULL_DM) <= FULL_TOL, "pred_b_parts_replay_v324_blocks": all(abs(share[f"{l}.both"] - V324[l]) <= BLOCK_TOL for l in (2, 3, 4)),
                   "pred_c_mlp_parts_exceed_attention_parts": all(share[f"{l}.mlp"] > share[f"{l}.attn"] for l in (2, 3, 4)), "pred_d_parts_additive": all(abs(share[f"{l}.attn"] + share[f"{l}.mlp"] - share[f"{l}.both"]) <= ADD_TOL for l in (2, 3, 4)), "pred_e_mlp2_is_the_largest_part": largest == "2.mlp"}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_routes_parts_result_v325", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
