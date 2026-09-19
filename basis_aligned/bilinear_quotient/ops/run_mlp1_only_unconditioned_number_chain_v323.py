#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_mlp1_write_becomes_table pred_c_margin_change_small_when_only_mlp1_unconditioned pred_d_carriage_flat pred_e_margin_change_is_a_fraction_of_v322
"""The number chain when only MLP 1 is un-conditioned (v323). v322: with attention 0/1 all-self-only on the v76 rows the they - he margin fell 37%, but
that edit changes what every downstream consumer receives from blocks 0/1 attention. Here the self-only attention is computed in parallel and fed ONLY
to MLP 1's input (MLP 1 reads the residual as it would be under self-only attention, so its write is the table entry); the residual stream itself keeps
the native attention writes, and every other block is native. Read: MLP 1's noun write (alpha, cosine), its exact number carriage into MLP-3 units
3465 / 493, and the they - he margin. The margin change here is attributable to MLP 1's un-conditioning alone (v310-v313 priced it at 0.7 nats on text).
PREDICTIONS (scored as written; failures preserved; priors from v322 / v296 / v312)
    pred_a_baseline_replays                        native alpha 0.540 +- 0.01 and native table share for 3465 0.736 +- 0.02
    pred_b_mlp1_write_becomes_table                under the edit, median alpha >= 0.95 and median cosine >= 0.95 at the noun
    pred_c_margin_change_small_when_only_mlp1_unconditioned  |oriented margin change| <= 0.15 of the native margin (the number chain reads the entry's direction, which the edit keeps). Prior: unsure -- v312 says the raw lookup is harmful on text.
    pred_d_carriage_flat                            MLP 1's carriage into 3465 under the edit is within 0.8x-1.25x of native
    pred_e_margin_change_is_a_fraction_of_v322      |margin change| here <= 0.5 x v322's 0.766 (most of v322's drop was attention's other consumers)
PRICE (registered maximum): 3 row batches x (native + edited) + 1 table batch = 7 forwards; 0 backwards; 0 fits. Bar <= 9.
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
OUT = ROOT / "circuits/followups/mlp1_only_unconditioned_number_chain_v323_result.json"
CANDIDATE_ID = "mlp1.token_table.only_unconditioned_number_chain_v323"
UNITS, DST, SRC, BATCH = (3465, 493), 3, 1, 32
REPLAY_A, REPLAY_TOL, REPLAY_SHARE, SHARE_TOL, ALPHA_MIN, MARGIN_MAX, FLAT_LO, FLAT_HI, V322_DROP = 0.540, 0.01, 0.736, 0.02, 0.95, 0.15, 0.8, 1.25, 0.7657
FORWARDS_MAX = 9
HEADS_ALL = [(l, h) for l in (0, 1) for h in range(9)]
PREDICTIONS = {"pred_a_baseline_replays": "alpha 0.540 +- 0.01; share 0.736 +- 0.02", "pred_b_mlp1_write_becomes_table": ">= 0.95", "pred_c_margin_change_small_when_only_mlp1_unconditioned": "<= 0.15 of native", "pred_d_carriage_flat": "0.8x-1.25x", "pred_e_margin_change_is_a_fraction_of_v322": "<= 0.5 x 0.766"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "src": SRC, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_alpha": REPLAY_A, "replay_tol": REPLAY_TOL, "replay_share": REPLAY_SHARE, "share_tol": SHARE_TOL, "alpha_min": ALPHA_MIN, "margin_max": MARGIN_MAX, "flat_lo": FLAT_LO, "flat_hi": FLAT_HI, "v322_drop": V322_DROP}}
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
    def run(self_only):
        W, X3, margins = [], [], []
        with torch.no_grad():
            for start in range(0, len(rows), BATCH):
                chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); pos = torch.tensor([noun_of(r) for r in chunk], device=tokens.device); idx = torch.arange(len(chunk))
                x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; xs, v1s = x, None    # xs: the parallel self-only stream
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if self_only and l in (0, 1):
                        lives = block.lambdas[0] * xs + block.lambdas[1] * x0; orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
                        try: atts, v1s = block.attn(F.rms_norm(lives, (model.config.n_embd,)), v1s)
                        finally: block.attn.squared_attention = orig
                        xs = lives + atts
                    if l == DST: X3.append(x[idx, pos].float().cpu())
                    if l == SRC and self_only: m = block.mlp(F.rms_norm(xs, (model.config.n_embd,)))        # MLP 1 reads the self-only stream; the residual keeps native attention
                    else: m = block.mlp(xin)
                    if l == SRC: W.append(m[idx, pos].float().cpu())
                    if self_only and l == 0: xs = xs + block.mlp(F.rms_norm(xs, (model.config.n_embd,)))
                    x = x + m
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
                for i, row in enumerate(chunk):
                    lg = logits[i, row.final].float(); margins.append(float(lg[L._single(" they")] - lg[L._single(" he")]))
        return torch.cat(W), torch.cat(X3), margins
    for _ in range(2): forwards += 3
    nat = run(False); ed = run(True)
    noun_ids = sorted({row.ids[noun_of(row)] for row in rows}); tok = torch.tensor(noun_ids, device="cuda").unsqueeze(1)
    tab = v289.capture(backend, tok, torch.zeros(len(noun_ids), dtype=torch.long, device="cuda")); forwards += 1; tindex = {t: i for i, t in enumerate(noun_ids)}
    T = torch.stack([tab["mlp1"][tindex[row.ids[noun_of(row)]]] for row in rows])
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    dst = blocks[DST].mlp
    def carriage(W, X3, unit):
        Lr, Rr = dst.Left.weight.detach().float()[unit].cpu(), dst.Right.weight.detach().float()[unit].cpu(); rms2 = X3.pow(2).mean(1); Lx, Rx = X3 @ Lr, X3 @ Rr
        ws = scale * W; lw, rw = ws @ Lr, ws @ Rr; tot = (lw * Rx + Lx * rw - lw * rw) / rms2
        A = ((W * T).sum(1) / (T * T).sum(1))[:, None] * T; wa = scale * A; la, ra = wa @ Lr, wa @ Rr; ca = (la * Rx + Lx * ra - la * ra) / rms2
        acc, acc_a, pairs = 0.0, 0.0, []
        for i, row in enumerate(rows):
            if not row.present: continue
            j = partner[(row.construction, row.group, False)]; d = float(tot[i] - tot[j]); acc += d; acc_a += float(ca[i] - ca[j]); pairs.append(d)
        return acc, acc_a, pairs
    def oriented(m): return sum((v if row.present else -v) for v, row in zip(m, rows)) / len(rows)
    out = {}
    for name, (W, X3, margins) in (("native", nat), ("edit", ed)):
        alpha = (W * T).sum(1) / (T * T).sum(1); cos = (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
        out[name] = {"alpha_median": float(alpha.median()), "cos_median": float(cos.median()), "margin_oriented": oriented(margins)}
        for unit in UNITS:
            tot, ta, pairs = carriage(W, X3, unit); out[name][f"carriage_{unit}"] = tot; out[name][f"table_share_{unit}"] = ta / tot; out[name][f"pairs_{unit}"] = pairs
    sign_ok = min(sum(1 for a_, b_ in zip(out["edit"][f"pairs_{u}"], out["native"][f"pairs_{u}"]) if (a_ > 0) == (b_ > 0)) / len(out["native"][f"pairs_{u}"]) for u in UNITS)
    rise = out["edit"]["carriage_3465"] / out["native"]["carriage_3465"]; dm = out["edit"]["margin_oriented"] - out["native"]["margin_oriented"]
    report = {k: {a_: b_ for a_, b_ in v.items() if not a_.startswith("pairs_")} for k, v in out.items()}; report.update({"carriage_rise_3465": rise, "carriage_rise_493": out["edit"]["carriage_493"] / out["native"]["carriage_493"], "margin_change": dm, "margin_change_rel": dm / out["native"]["margin_oriented"], "sign_agreement_min": sign_ok})
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": abs(out["native"]["alpha_median"] - REPLAY_A) <= REPLAY_TOL and abs(out["native"]["table_share_3465"] - REPLAY_SHARE) <= SHARE_TOL, "pred_b_mlp1_write_becomes_table": out["edit"]["alpha_median"] >= ALPHA_MIN and out["edit"]["cos_median"] >= ALPHA_MIN,
                   "pred_c_margin_change_small_when_only_mlp1_unconditioned": abs(dm) <= MARGIN_MAX * abs(out["native"]["margin_oriented"]), "pred_d_carriage_flat": FLAT_LO <= rise <= FLAT_HI, "pred_e_margin_change_is_a_fraction_of_v322": abs(dm) <= 0.5 * V322_DROP}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_only_unconditioned_number_chain_result_v323", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
