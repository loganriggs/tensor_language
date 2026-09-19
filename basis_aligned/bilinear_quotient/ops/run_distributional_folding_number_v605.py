#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_positive_on_almost_every_row pred_c_cv_moderate pred_d_panel_and_text_means_overlap pred_e_panel_variance_smaller
"""Distributional folding: expected circuit and its variance (v605; approved 19 Sep 22:12 UTC). Every fold so far has reported a POOLED fraction (sum of
numerators over sum of denominators across rows) -- a single number, as if the circuit had one fixed size. Here the five pronoun readers' joint value
swap at the noun (the number circuit's best-established single number, panel 0.48 / text 0.60 pooled) is scored PER ROW instead, on both the panel (a
synthetic template with the noun as the only varying slot -- the closest thing this repo has to a controlled/Gaussian-moment-style distribution over one
factor) and the 122 natural text rows (the empirical distribution: real sentences, real co-occurring context, real noun frequencies) -- giving E[fraction],
Var[fraction], CV, and a same-vs-different check between the two distributions' means (the "load-bearing on-distribution vs OOD-only" question, done as a
genuine two-sample comparison rather than a single pooled number standing in for both).
PREDICTIONS (scored as written; failures preserved; priors from the pooled figures already on record: panel 0.483, text 0.600 (v433))
    pred_a_baseline_replays        the unedited manual forward replays the model's they - he margins within 1e-3, both sets
    pred_b_positive_on_almost_every_row  the per-row fraction is positive (oriented the right way) on >= 0.85 of rows, both sets
    pred_c_cv_moderate             the coefficient of variation (std/|mean|) of the per-row fraction is <= 1.5 on the natural text rows -- the pooled
                                    0.60 is not an artefact of a few huge rows swamping many near-zero ones. Prior: unsure
    pred_d_panel_and_text_means_overlap  the panel's per-row mean fraction and the text's per-row mean fraction are within 2 x the LARGER set's standard
                                    error of each other (no evidence of a real distributional shift, beyond what v433/v406's pooled gap already showed).
                                    Prior: unlikely to hold exactly -- the pooled numbers already differ (0.48 vs 0.60)
    pred_e_panel_variance_smaller  the panel's per-row variance is smaller than the text's (the synthetic template controls everything but the noun;
                                    natural text varies noun, verb, distance and surrounding words all at once). Prior: likely
PRICE (registered maximum): panel 3 batches + text 4 batches, x 2 passes (native + swap) + 1 replay-check forward = 15 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, math, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/distributional_folding_number_v605_result.json"
CANDIDATE_ID = "distributional.number_reader_fraction_v605"
N_HEAD = 9
FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, POS_MIN, CV_MAX = 1e-3, 0.85, 1.5
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_positive_on_almost_every_row": ">= 0.85 x 2", "pred_c_cv_moderate": "<= 1.5",
               "pred_d_panel_and_text_means_overlap": "within 2 SE", "pred_e_panel_variance_smaller": "panel var < text var"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "pos_min": POS_MIN, "cv_max": CV_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE = L._single(" they"), L._single(" he")
    forwards = 0

    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, THEY] - z[:, HE]

    def run(seqs, pos, swap_it):
        nonlocal forwards
        tokens = torch.tensor([q + [0] * (max(len(t) for t in seqs) - len(q)) for q in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if swap_it and l in FIVE:
                    hs = FIVE[l]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    v1n = v if v1_ is None else v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in hs: v[idx, pp, h] = v[swap, pp, h].clone()
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
            return logits_margin(x[idx, fin]).cpu()

    def per_row(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        nat, ed = [], []
        for s0 in range(0, len(seqs), 2 * batch):
            nat.append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], False))
            ed.append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], True))
        nat, ed = torch.cat(nat), torch.cat(ed)
        gap = nat[0::2] - nat[1::2]
        frac = ((nat[0::2] - ed[0::2]) + (ed[1::2] - nat[1::2])) / (2 * gap)   # per-row closed fraction
        return frac, gap, nat

    frac_panel, gap_panel, nat_panel = per_row(panel_items, 16)
    frac_text, gap_text, nat_text = per_row(text_items, 32)

    with torch.no_grad():
        n = min(32, len(text_items)); seqs = [list(p) for p, _, _ in text_items[:n]] + [list(s) for _, s, _ in text_items[:n]]
        T_ = max(len(q) for q in seqs); tokens = torch.tensor([q + [0] * (T_ - len(q)) for q in seqs], device="cuda"); fin_ = torch.tensor([len(q) - 1 for q in seqs], device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[torch.arange(2 * n, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_text[0:2 * n:2], nat_text[1:2 * n:2]]); replay = float((mine - ref).abs().max())

    def stats(frac):
        f = frac.double()
        mean = float(f.mean()); std = float(f.std(unbiased=True)); se = std / math.sqrt(len(f))
        cv = std / abs(mean) if abs(mean) > 1e-9 else float("inf")
        pos_frac = float((f > 0).float().mean())
        return {"n": len(f), "mean": mean, "std": std, "se": se, "cv": cv, "positive_fraction": pos_frac, "var": std ** 2, "values": f.tolist()}

    s_panel, s_text = stats(frac_panel), stats(frac_text)
    diff = abs(s_panel["mean"] - s_text["mean"]); combined_se = max(s_panel["se"], s_text["se"])
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_positive_on_almost_every_row": s_panel["positive_fraction"] >= POS_MIN and s_text["positive_fraction"] >= POS_MIN,
                   "pred_c_cv_moderate": s_text["cv"] <= CV_MAX, "pred_d_panel_and_text_means_overlap": diff <= 2 * combined_se,
                   "pred_e_panel_variance_smaller": s_panel["var"] < s_text["var"]}
    print("panel", {k: round(v, 4) if isinstance(v, float) else v for k, v in s_panel.items() if k != "values"})
    print("text", {k: round(v, 4) if isinstance(v, float) else v for k, v in s_text.items() if k != "values"})
    print(json.dumps(predictions, indent=2))
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "distributional_folding_number_result_v605", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "panel": s_panel, "text": s_text, "replay_max_abs": replay, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
