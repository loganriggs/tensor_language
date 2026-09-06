"""v56: remove each head set's block diff-in-means direction at EVERY position and measure per-token collateral over whole
sentences (rubric row 4 across the sentence, not only at the answer token).

v50/v51 set the coordinate along the block direction to its background value at the prediction position only and measured
collateral at the answer token only. Two open questions follow. (1) Is the direction a final-position readout carrier, or is
it also used at earlier positions (heads reading it there through later attention)? All-position removal on the set's own
sentences answers this: if the answer-CE damage rises well beyond the final-only damage, earlier positions carry it too.
(2) Off-target across the sentence: per-token CE increase over all positions of the C sentences and of the other sets' A1
sentences under all-position removal -- the honest row-4 statement for a direction that is removed everywhere.
Intervention (all four sets of v23.SETS, rank 1 per block): live coordinate c = x.q replaced by the background coordinate;
background = mean over the set's A1 tokens (base + donor): at the final position the v51 final-position mean (so the
final-only arm is a replica of v51 -- instrument control), at other positions the all-token mean. Random rank-1 direction
control. New code path (own full forward with c_proj pre-hooks) is controlled against g.forward_units on the no-hook path.

REGISTERED BEFORE THE RUN
    pred_a_instrument      no-hook forward matches g.forward_units answer/foil to 1e-4 on every set AND the final-only arm's
                           own answer-CE damage is within 0.02 nat of v51's recorded value on every set. Worked: 0.355 vs 0.3549 True; 0.30 False.
    pred_b_final_carrier   all-position / final-only own answer-CE damage in [0.8, 1.5] on all four sets (direction is a
                           final-position carrier). Worked: 1.1 True; 2.5 False (earlier positions also carry it).
    pred_c_offtarget_C     all-position removal, per-token CE increase over the C sentences: 97.5% UB <= 0.01 nat on all
                           four sets. Worked: 0.004 True; 0.03 False.
    pred_d_cross_tokens    per-token CE increase over the other sets' A1 sentences: UB <= 0.10 x own answer damage (point)
                           on all 12 pairs. Worked: 0.010 vs 0.0355 True; 0.05 False.
    pred_e_random          random rank-1 direction, all positions, own A1 per-token CE increase UB <= 0.01 nat on all sets.
                           Worked: 0.002 True; 0.02 False.
    Reading rule. b False with c/d True: the set's direction carries the behaviour at earlier positions too -- the terminal
    row-3 statement is per-position and must be reported as such; no unit is added. c or d False: row 4 fails across the
    sentence even though it passed at the answer token (v51) -- the tier list entry is downgraded, not the bar.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_polarity_selective_removal_v50 as v50
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_all_position_removal_v56_result.json"
V51 = ROOT / "circuits/followups/unit_selective_removal_four_sets_v51_result.json"
SETS, HEAD_DIM, N_EMBD = v23.SETS, 128, 1152
INSTR_TOL, V51_TOL, CARRIER_BAND, OFF_UB, CROSS_FRAC, RAND_UB = 1e-4, 0.02, (0.8, 1.5), 0.01, 0.10, 0.01
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 6000


def _plan():
    return {"candidate_id": "corpus.unit_all_position_removal_v56", "sets": {k: v[1] for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def spans_of(units):
    """{layer: [(s, e), ...]} in the given unit order (the block concatenation order)."""
    out = {}
    for u in units:
        h = int(u.rsplit(":", 1)[1])
        out.setdefault(g.unit_layer(u), []).append((h * HEAD_DIM, (h + 1) * HEAD_DIM))
    return out


def forward_all(backend, batch, hooks=()):
    """The producer's exact forward; returns all-position logits (n, T, V). hooks: [(layer, fn)] c_proj pre-hooks."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    handles = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(fn) for l, fn in hooks]
    try:
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (N_EMBD,))
            x0, v1 = x, None
            for block in model.transformer.h:
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (N_EMBD,)), v1)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (N_EMBD,)))
            logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (N_EMBD,))) / 30.0)
    finally:
        for h in handles:
            h.remove()
    return logits.float(), tokens, lengths


def removal_hooks(torch, spans, q, c_final, c_all, lengths, mode):
    """mode 'final' -> replace the coordinate at the last position only; 'all' -> at every position (final uses c_final)."""
    last = torch.tensor([l - 1 for l in lengths])
    hooks = []
    for l, sp in spans.items():
        qb = q[(l, "heads")][:, 0]                                            # (D_blk,)

        def fn(m, a, l=l, sp=sp, qb=qb):
            v = a[0]
            n, T, _ = v.shape
            blk = torch.cat([v[:, :, s:e] for s, e in sp], dim=2).float()      # (n, T, D_blk)
            c = blk @ qb                                                       # (n, T)
            target = torch.full_like(c, float(c_all[l])) if mode == "all" else c.clone()
            target[torch.arange(n), last.to(v.device)] = float(c_final[l])
            delta = (target - c)[..., None] * qb                               # (n, T, D_blk)
            out = v.clone(); o = 0
            for s, e in sp:
                out[:, :, s:e] = v[:, :, s:e] + delta[..., o:o + (e - s)].to(v.dtype); o += e - s
            return (out,) + tuple(a[1:])
        hooks.append((l, fn))
    return hooks


def per_token_ce(F, logits, tokens, lengths):
    """Per-document mean next-token CE over positions 0..len-2."""
    lp = F.log_softmax(logits[:, :-1], -1)
    nll = -lp.gather(-1, tokens[:, 1:, None])[..., 0]
    return [float(nll[i, :l - 1].mean()) for i, l in enumerate(lengths)]


def coords(backend, prep, spans, q):
    """Background coordinates along q: final-position mean (v51) and all-token mean, over A1 base + donor."""
    torch = backend.torch
    c_final, c_all = {}, {}
    grabbed = {}
    for l, sp in spans.items():
        def fn(m, a, l=l, sp=sp):
            grabbed.setdefault(l, []).append(torch.cat([a[0][:, :, s:e] for s, e in sp], dim=2).float())
        grabbed.setdefault(l, [])
        hooks = [(l, fn)]
        for batch in (prep.base_batch, prep.donor_batch):
            _, _, lengths = forward_all(backend, batch, hooks)
            grabbed[l][-1] = (grabbed[l][-1], lengths)
        qb = q[(l, "heads")][:, 0]
        fin, alltok = [], []
        for blk, lengths in grabbed[l]:
            c = blk @ qb
            for i, L in enumerate(lengths):
                fin.append(c[i, L - 1]); alltok.append(c[i, :L])
        c_final[l] = float(torch.stack(fin).mean()); c_all[l] = float(torch.cat(alltok).mean())
    return c_final, c_all


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    t0 = time.perf_counter()
    v51 = json.loads(V51.read_text())["behaviours"]
    preps = {n: g.prepare(backend, g.rows_of(m, "A1")) for n, (m, _) in SETS.items()}
    c_prep = g.prepare(backend, g.rows_of(SETS["polarity_licensing"][0], "C"))

    def natives(prep):
        out = {}
        for side, batch in (("base", prep.base_batch), ("donor", prep.donor_batch)):
            logits, tokens, lengths = forward_all(backend, batch)
            out[side] = (logits, tokens, lengths)
        return out
    nat = {n: natives(p) for n, p in preps.items()}
    nat["C"] = natives(c_prep)

    def measure(prep, native, hooks):
        """Answer-CE damage per doc (final position) and per-token CE increase per doc, over base + donor sentences."""
        ans_d, tok_d = [], []
        for side, batch in (("base", prep.base_batch), ("donor", prep.donor_batch)):
            l0, tokens, lengths = native[side]
            l1, _, _ = forward_all(backend, batch, hooks)
            i = torch.arange(len(lengths), device=l0.device); last = torch.tensor([l - 1 for l in lengths], device=l0.device)
            ans = torch.tensor(batch.answer_ids, device=l0.device)
            ans_d += (F.log_softmax(l0[i, last], -1)[i, ans] - F.log_softmax(l1[i, last], -1)[i, ans]).tolist()
            a, b = per_token_ce(F, l0, tokens, lengths), per_token_ce(F, l1, tokens, lengths)
            tok_d += [y - x for x, y in zip(a, b)]
        return ans_d, tok_d

    def stat(x):
        p, lb, ub = v50._boot(torch, x)
        return {"point": p, "lb975": lb, "ub975": ub, "documents": len(x)}

    report, instr = {}, {}
    for n, (m, units) in SETS.items():
        units = list(units); prep = preps[n]
        # instrument: the no-hook path vs g.forward_units at the answer position
        mine, _, lengths = forward_all(backend, prep.base_batch)
        i = torch.arange(len(lengths), device=mine.device); last = torch.tensor([l - 1 for l in lengths], device=mine.device)
        af_mine = torch.stack([mine[i, last, torch.tensor(prep.base_batch.answer_ids, device=mine.device)],
                               mine[i, last, torch.tensor(prep.base_batch.foil_ids, device=mine.device)]], 1)
        instr[n] = float((af_mine - g.forward_units(backend, prep.base_batch).float()).abs().max())
        spans = spans_of(units)
        q = g.block_diff_in_means(backend, prep, units)
        q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
        c_final, c_all = coords(backend, prep, spans, q)
        cr_final, cr_all = coords(backend, prep, spans, q_rand)
        def run(prep_, native, mode, qq=q, cf=c_final, ca=c_all):
            # lengths differ per batch; build the hooks per side inside measure via a closure over each batch
            ans_d, tok_d = [], []
            for side, batch in (("base", prep_.base_batch), ("donor", prep_.donor_batch)):
                l0, tokens, lengths = native[side]
                hooks = removal_hooks(torch, spans, qq, cf, ca, lengths, mode)
                l1, _, _ = forward_all(backend, batch, hooks)
                i = torch.arange(len(lengths), device=l0.device); last = torch.tensor([l - 1 for l in lengths], device=l0.device)
                ans = torch.tensor(batch.answer_ids, device=l0.device)
                ans_d += (F.log_softmax(l0[i, last], -1)[i, ans] - F.log_softmax(l1[i, last], -1)[i, ans]).tolist()
                a, b = per_token_ce(F, l0, tokens, lengths), per_token_ce(F, l1, tokens, lengths)
                tok_d += [y - x for x, y in zip(a, b)]
            return {"answer_ce": stat(ans_d), "token_ce": stat(tok_d)}
        r = {"units": units, "instrument_max_abs_err": instr[n], "v51_final_only_ce": v51[n]["target_A1"]["ce_damage"],
             "own_final_only": run(prep, nat[n], "final"), "own_all": run(prep, nat[n], "all"),
             "C_all": run(c_prep, nat["C"], "all"), "own_random_all": run(prep, nat[n], "all", q_rand, cr_final, cr_all), "cross_all": {}}
        for t in SETS:
            if t != n:
                r["cross_all"][t] = run(preps[t], nat[t], "all")
        r["carrier_ratio"] = r["own_all"]["answer_ce"]["point"] / r["own_final_only"]["answer_ce"]["point"]
        report[n] = r
    predictions = {
        'pred_a_instrument': all(r["instrument_max_abs_err"] <= INSTR_TOL and abs(r["own_final_only"]["answer_ce"]["point"] - r["v51_final_only_ce"]) <= V51_TOL for r in report.values()),
        'pred_b_final_carrier': all(CARRIER_BAND[0] <= r["carrier_ratio"] <= CARRIER_BAND[1] for r in report.values()),
        'pred_c_offtarget_C': all(r["C_all"]["token_ce"]["ub975"] <= OFF_UB for r in report.values()),
        'pred_d_cross_tokens': all(c["token_ce"]["ub975"] <= CROSS_FRAC * r["own_all"]["answer_ce"]["point"] for r in report.values() for c in r["cross_all"].values()),
        'pred_e_random': all(r["own_random_all"]["token_ce"]["ub975"] <= RAND_UB for r in report.values()),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_all_position_removal_result_v1", "candidate_id": "corpus.unit_all_position_removal_v56",
              "bars": {"instr_tol": INSTR_TOL, "v51_tol": V51_TOL, "carrier_band": CARRIER_BAND, "off_ub": OFF_UB, "cross_frac": CROSS_FRAC, "rand_ub": RAND_UB},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    brief = {n: {"instr": round(r["instrument_max_abs_err"], 6), "v51": round(r["v51_final_only_ce"], 3),
                 "final_only": round(r["own_final_only"]["answer_ce"]["point"], 3), "all": round(r["own_all"]["answer_ce"]["point"], 3),
                 "ratio": round(r["carrier_ratio"], 2), "own_tok": round(r["own_all"]["token_ce"]["point"], 4),
                 "C_tok_ub": round(r["C_all"]["token_ce"]["ub975"], 4), "C_ans": round(r["C_all"]["answer_ce"]["point"], 4),
                 "rand_tok_ub": round(r["own_random_all"]["token_ce"]["ub975"], 4),
                 "cross_tok_ub": {t: round(c["token_ce"]["ub975"], 4) for t, c in r["cross_all"].items()}} for n, r in report.items()}
    print(json.dumps({"predictions": predictions, "brief": brief, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
