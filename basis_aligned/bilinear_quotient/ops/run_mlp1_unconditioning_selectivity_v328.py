#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_margin_drop_replays_v323 pred_c_harm_is_not_number_selective pred_d_final_token_distribution_moves pred_e_correct_answer_logprob_falls
"""Is the un-conditioning harm number-selective? (v328). v323: feeding MLP 1 a self-only-attention input on the v76 rows cuts the they - he margin 32%.
Is that a number effect or a general degradation of the final-token prediction? Same edit (MLP 1 alone un-conditioned; residual and all other blocks
native); readouts at the final token: (i) the oriented they - he margin; (ii) a non-number control contrast on the same rows, the oriented he - she
margin (gender, which the rows do not manipulate: its mean should not move); (iii) the KL divergence of the final-token distribution, edited vs native;
(iv) the log-probability of the correct pronoun. Registered reading (from v312: the raw lookup is broadly harmful): the harm is NOT selective -- the KL
is large and the control contrast moves too.
PREDICTIONS (scored as written; failures preserved; priors from v312 / v323)
    pred_a_baseline_replays                 native oriented margin 2.048 +- 1e-3
    pred_b_margin_drop_replays_v323         edited oriented margin change -0.655 +- 0.03
    pred_c_harm_is_not_number_selective     |mean change of the he - she control| >= 0.25 x |they - he change| (a general disturbance, not a number-specific one). Prior: unsure.
    pred_d_final_token_distribution_moves   mean KL(native || edited) at the final token >= 0.05 nats
    pred_e_correct_answer_logprob_falls     mean log-probability of the correct pronoun falls under the edit
PRICE (registered maximum): 3 row batches x (native + edited) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
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
OUT = ROOT / "circuits/followups/mlp1_unconditioning_selectivity_v328_result.json"
CANDIDATE_ID = "mlp1.token_table.unconditioning_selectivity_v328"
UNITS, DST, SRC, BATCH = (3465, 493), 3, 1, 32
NATIVE_M, M_TOL, V323_DM, DM_TOL, CONTROL_RATIO, KL_MIN = 2.0481, 1e-3, -0.6554, 0.03, 0.25, 0.05
FORWARDS_MAX = 8
HEADS_ALL = [(l, h) for l in (0, 1) for h in range(9)]
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_margin_drop_replays_v323": "-0.655 +- 0.03", "pred_c_harm_is_not_number_selective": ">= 0.25x", "pred_d_final_token_distribution_moves": ">= 0.05 nats", "pred_e_correct_answer_logprob_falls": "falls"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "src": SRC, "dst": DST, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "v323_dm": V323_DM, "dm_tol": DM_TOL, "control_ratio": CONTROL_RATIO, "kl_min": KL_MIN}}
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
                margins.append(torch.stack([logits[i, row.final].float().cpu() for i, row in enumerate(chunk)]))
        return torch.cat(W), torch.cat(X3), torch.cat(margins)
    for _ in range(2): forwards += 3
    nat = run(False); ed = run(True); Ln, Le = nat[2], ed[2]
    they, he, she = L._single(" they"), L._single(" he"), L._single(" she")
    def oriented(Lg, a_, b_): return float(sum((Lg[i, a_] - Lg[i, b_]) * (1 if row.present else -1) for i, row in enumerate(rows)) / len(rows))
    base = oriented(Ln, they, he); dm = oriented(Le, they, he) - base; ctrl_n = oriented(Ln, he, she); dctrl = oriented(Le, he, she) - ctrl_n
    Pn, Pe = torch.log_softmax(Ln, -1), torch.log_softmax(Le, -1); kl = float((Pn.exp() * (Pn - Pe)).sum(-1).mean())
    correct = torch.tensor([row.answer_id for row in rows]); lp_n = float(Pn[torch.arange(len(rows)), correct].mean()); lp_e = float(Pe[torch.arange(len(rows)), correct].mean())
    report = {"native_margin": base, "margin_change": dm, "control_he_she_native": ctrl_n, "control_he_she_change": dctrl, "control_ratio": abs(dctrl) / abs(dm), "kl_native_edited_mean": kl, "kl_median": float((Pn.exp() * (Pn - Pe)).sum(-1).median()),
              "correct_logprob_native": lp_n, "correct_logprob_edited": lp_e, "top1_agreement": float((Ln.argmax(-1) == Le.argmax(-1)).float().mean())}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": abs(base - NATIVE_M) <= M_TOL, "pred_b_margin_drop_replays_v323": abs(dm - V323_DM) <= DM_TOL, "pred_c_harm_is_not_number_selective": abs(dctrl) >= CONTROL_RATIO * abs(dm), "pred_d_final_token_distribution_moves": kl >= KL_MIN, "pred_e_correct_answer_logprob_falls": lp_e < lp_n}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_unconditioning_selectivity_result_v328", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
