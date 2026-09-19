#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pair_closure pred_b_classes_differ_in_gain pred_c_function_words_and_punctuation_cancel_most pred_d_numbers_keep_lookup_most pred_e_class_order_shared_by_mlp1_and_mlp2
"""Which token classes keep their lookup? MLP 1 and MLP 2 by class (v320). The cancellation law (v301 / v319) was measured pooled over seven token classes.
Logan's question was about classes: fold the paths class by class. Phrase A, 8 tokens, 32 targets per class (lexicon nouns, vocabulary nouns, verbs, numbers,
punctuation, function words, adjectives): per class and per layer (MLP 1, MLP 2), the lookup gain alpha, gamma^2, the cross projection and the cosine
with the table entry. v290 already showed alpha_1's class medians 0.41 (punctuation) to 0.57 (nouns) at one context token.
PREDICTIONS (scored as written; failures preserved; priors from v290 / v302 word-vs-rest split)
    pred_a_pair_closure                          gamma^2 T + cross + context^2 = W within relative 2e-2 on every row, both layers
    pred_b_classes_differ_in_gain                the spread of alpha_1's class medians (max - min) >= 0.10
    pred_c_function_words_and_punctuation_cancel_most  function words and punctuation have the two lowest alpha_1 class medians
    pred_d_numbers_keep_lookup_most              numbers have the highest alpha_1 class median. Prior: unsure.
    pred_e_class_order_shared_by_mlp1_and_mlp2   Spearman rank correlation of the seven class medians of alpha between MLP 1 and MLP 2 >= 0.60
PRICE (registered maximum): 2 table batches (blocks 0-1 and 0-2 outputs, one pass) + 1 length batch = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp12_class_gain_v320_result.json"
CANDIDATE_ID = "mlp1.token_table.class_gain_v320"
PHRASE = (",", " and", " of", " the", " very")
K, N, BATCH = 8, 32, 256
CLOSURE_TOL, SPREAD_MIN, RANK_MIN = 2e-2, 0.10, 0.60
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_pair_closure": "<= 2e-2", "pred_b_classes_differ_in_gain": ">= 0.10", "pred_c_function_words_and_punctuation_cancel_most": "two lowest", "pred_d_numbers_keep_lookup_most": "highest", "pred_e_class_order_shared_by_mlp1_and_mlp2": "rho >= 0.60"}


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; targets = sorted({t for v in cls.values() for t in v}); tindex = {t: i for i, t in enumerate(targets)}
    plan = {"candidate_id": CANDIDATE_ID, "phrase": list(PHRASE), "length": K, "class_sizes": {k: len(v) for k, v in cls.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "spread_min": SPREAD_MIN, "rank_min": RANK_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    def capture12(tokens, pos):
        idx = torch.arange(tokens.shape[0]); out = {}
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1, 2):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l in (1, 2): out[f"x{l}"] = x[idx, pos].float().cpu(); out[f"mlp{l}"] = m[idx, pos].float().cpu()
                x = x + m
        return out
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = capture12(ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1
    toks = torch.tensor([filler(K) + [t] for t in targets], device="cuda"); c = capture12(toks, torch.full((len(targets),), K, dtype=torch.long, device="cuda")); forwards += 1
    closure, per = 0.0, {}
    for l in (1, 2):
        mlp = model.transformer.h[l].mlp; Lw, Rw, Dw = mlp.Left.weight.detach().float().cpu(), mlp.Right.weight.detach().float().cpu(), mlp.Down.weight.detach().float().cpu()
        T = tab[f"mlp{l}"]; n_tab = F.rms_norm(tab[f"x{l}"], (T.shape[-1],)); W = c[f"mlp{l}"]; n = F.rms_norm(c[f"x{l}"], (T.shape[-1],))
        gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
        Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
        quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T; only = (Lc * Rc) @ Dw.T
        closure = max(closure, float((((quad + cross + only) - W).norm(dim=1) / W.norm(dim=1)).max()))
        pT = lambda A: (A * T).sum(1) / (T * T).sum(1); alpha, g2, pc, po = pT(W), pT(quad), pT(cross), pT(only); cos = (W * T).sum(1) / (W.norm(dim=1) * T.norm(dim=1))
        for name, v in cls.items():
            i = torch.tensor([tindex[t] for t in v]); per[f"mlp{l}|{name}"] = {"alpha": float(alpha[i].median()), "gamma2": float(g2[i].median()), "cross": float(pc[i].median()), "only": float(po[i].median()), "cos": float(cos[i].median()), "c_norm": float(cc[i].norm(dim=1).median())}
    a1 = {name: per[f"mlp1|{name}"]["alpha"] for name in cls}; a2 = {name: per[f"mlp2|{name}"]["alpha"] for name in cls}
    order1 = sorted(cls, key=lambda n_: a1[n_]); order2 = sorted(cls, key=lambda n_: a2[n_])
    r1 = {n_: i for i, n_ in enumerate(order1)}; r2 = {n_: i for i, n_ in enumerate(order2)}; d2 = sum((r1[n_] - r2[n_]) ** 2 for n_ in cls); rho = 1 - 6 * d2 / (len(cls) * (len(cls) ** 2 - 1))
    report = {"closure_max": closure, "alpha1_by_class": a1, "alpha2_by_class": a2, "alpha1_spread": max(a1.values()) - min(a1.values()), "order_mlp1_low_to_high": order1, "order_mlp2_low_to_high": order2, "spearman_alpha1_alpha2": rho, "per": per}
    print(json.dumps({k: v for k, v in report.items() if k != "per"}, indent=1))
    predictions = {"pred_a_pair_closure": closure <= CLOSURE_TOL, "pred_b_classes_differ_in_gain": report["alpha1_spread"] >= SPREAD_MIN, "pred_c_function_words_and_punctuation_cancel_most": set(order1[:2]) == {"function_words", "punctuation"},
                   "pred_d_numbers_keep_lookup_most": order1[-1] == "numbers", "pred_e_class_order_shared_by_mlp1_and_mlp2": rho >= RANK_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_class_gain_result_v320", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
