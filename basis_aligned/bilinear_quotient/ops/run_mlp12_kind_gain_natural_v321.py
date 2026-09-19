#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_alphabetic_keeps_more_than_punctuation_mlp1 pred_c_numeric_lowest_at_mlp2 pred_d_class_gaps_replay_filler_sign pred_e_mlp2_cross_negative_every_kind
"""Lookup gain by token kind on natural text, MLP 1 and MLP 2 (v321). v320 (filler contexts): content words keep more of their lookup than
function words and punctuation at both layers; numbers lose theirs fastest by MLP 2 (0.14). OOD on the 2,944 natural positions with a lexicon-free
kind split of the predicted token: alphabetic word tokens, numeric tokens (all digits), punctuation / symbol tokens (no alphanumerics), other.
Per kind and layer: alpha, cosine, gamma^2, cross projection. Kinds with < 30 positions are reported but not scored.
PREDICTIONS (scored as written; failures preserved; priors from v320)
    pred_a_closure                                the block-2 recurrence reproduces the captured MLP-2 write within relative 1e-4 (instrument)
    pred_b_alphabetic_keeps_more_than_punctuation_mlp1  MLP 1: alpha median (alphabetic) >= alpha median (punctuation) + 0.03
    pred_c_numeric_lowest_at_mlp2                 MLP 2: alpha median (numeric) is the lowest of the scored kinds
    pred_d_class_gaps_replay_filler_sign          MLP 1 and MLP 2 both rank alphabetic above punctuation (the filler-context order survives text)
    pred_e_mlp2_cross_negative_every_kind         MLP 2's cross term projects negatively at >= 0.90 of positions within every scored kind
PRICE (registered maximum): 2 natural batches + <= 12 unique-token table batches = 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp12_kind_gain_natural_v321_result.json"
CANDIDATE_ID = "mlp1.token_table.kind_gain_natural_v321"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, GAP_MIN, NEG_MIN, MIN_N, BATCH = 1e-4, 0.03, 0.90, 30, 256
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_alphabetic_keeps_more_than_punctuation_mlp1": ">= +0.03", "pred_c_numeric_lowest_at_mlp2": "lowest", "pred_d_class_gaps_replay_filler_sign": "both layers", "pred_e_mlp2_cross_negative_every_kind": ">= 0.90 x kinds"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "natural": [p.name for p in NATURAL], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_min": GAP_MIN, "neg_min": NEG_MIN, "min_n": MIN_N}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); Tlen = nat.shape[1]
    writes, x1s, toks, w2s, x2s = [], [], [], [], []
    with torch.no_grad():
        for s0 in range(0, len(recs), 64):
            chunk = nat[s0:s0 + 64]; x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1, 2):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l == 1: x1, m1 = x, m
                if l == 2: x2, m2 = x, m
                x = x + m
            forwards += 1
            writes.append(m1[:, 1:].reshape(-1, m1.shape[-1]).float().cpu()); x1s.append(x1[:, 1:].reshape(-1, m1.shape[-1]).float().cpu()); toks.append(chunk[:, 1:].reshape(-1).cpu())
            w2s.append(m2[:, 1:].reshape(-1, m2.shape[-1]).float().cpu()); x2s.append(x2[:, 1:].reshape(-1, m2.shape[-1]).float().cpu())
    W, X1, tok, W2, X2 = torch.cat(writes), torch.cat(x1s), torch.cat(toks), torch.cat(w2s), torch.cat(x2s)
    uniq = sorted(set(tok.tolist())); tab = {"mlp1": [], "x1": [], "mlp2": [], "x2": []}
    with torch.no_grad():
        for s0 in range(0, len(uniq), BATCH):
            ids = torch.tensor(uniq[s0:s0 + BATCH], device="cuda").unsqueeze(1); x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
            for l in (0, 1, 2):
                block = model.transformer.h[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
                if l == 1: tab["x1"].append(x[:, 0].float().cpu()); tab["mlp1"].append(m[:, 0].float().cpu())
                if l == 2: tab["x2"].append(x[:, 0].float().cpu()); tab["mlp2"].append(m[:, 0].float().cpu())
                x = x + m
            forwards += 1
    tab = {k: torch.cat(v) for k, v in tab.items()}; tindex = {t: i for i, t in enumerate(uniq)}; ti = torch.tensor([tindex[t] for t in tok.tolist()])
    T1 = tab["mlp1"][ti]; alpha1 = (W * T1).sum(1) / (T1 * T1).sum(1)
    mlp2 = model.transformer.h[2].mlp; Lw, Rw, Dw = mlp2.Left.weight.detach().float().cpu(), mlp2.Right.weight.detach().float().cpu(), mlp2.Down.weight.detach().float().cpu()
    rec = mlp2(F.rms_norm(X2.to("cuda"), (X2.shape[-1],))).float().cpu(); closure = float(((rec - W2).norm(dim=1) / W2.norm(dim=1)).max())
    T2 = tab["mlp2"][ti]; alpha2 = (W2 * T2).sum(1) / (T2 * T2).sum(1); cos2 = (W2 * T2).sum(1) / (W2.norm(dim=1) * T2.norm(dim=1))
    n = F.rms_norm(X2, (X2.shape[-1],)); n_tab = F.rms_norm(tab["x2"][ti], (X2.shape[-1],)); gamma = (n * n_tab).sum(1) / (n_tab * n_tab).sum(1); tp = gamma[:, None] * n_tab; cc = n - tp
    Lt2, Rt2, Lc, Rc = tp @ Lw.T, tp @ Rw.T, cc @ Lw.T, cc @ Rw.T
    quad = (Lt2 * Rt2) @ Dw.T; cross = (Lt2 * Rc + Lc * Rt2) @ Dw.T; only = (Lc * Rc) @ Dw.T
    pT = lambda A: (A * T2).sum(1) / (T2 * T2).sum(1); g2, pc, po = pT(quad), pT(cross), pT(only)
    # MLP 1's three terms too
    mlp1 = model.transformer.h[1].mlp; L1, R1, D1 = mlp1.Left.weight.detach().float().cpu(), mlp1.Right.weight.detach().float().cpu(), mlp1.Down.weight.detach().float().cpu()
    n1 = F.rms_norm(X1, (X1.shape[-1],)); n1t = F.rms_norm(tab["x1"][ti], (X1.shape[-1],)); g1 = (n1 * n1t).sum(1) / (n1t * n1t).sum(1); tp1 = g1[:, None] * n1t; c1 = n1 - tp1
    q1 = ((tp1 @ L1.T) * (tp1 @ R1.T)) @ D1.T; x1c = ((tp1 @ L1.T) * (c1 @ R1.T) + (c1 @ L1.T) * (tp1 @ R1.T)) @ D1.T
    pT1 = lambda A: (A * T1).sum(1) / (T1 * T1).sum(1); g1sq, pc1 = pT1(q1), pT1(x1c); cos1 = (W * T1).sum(1) / (W.norm(dim=1) * T1.norm(dim=1))
    import tiktoken
    tk = tiktoken.get_encoding("gpt2"); texts = [tk.decode([t]).strip() for t in tok.tolist()]
    def kind(t):
        if t and t.isalpha(): return "alphabetic"
        if t and t.isdigit(): return "numeric"
        if t and not any(ch.isalnum() for ch in t): return "punctuation"
        return "other"
    kinds = [kind(t) for t in texts]; K = sorted(set(kinds)); per = {}
    for kd in K:
        i = torch.tensor([j for j, x_ in enumerate(kinds) if x_ == kd]); per[kd] = {"n": int(len(i)), "mlp1": {"alpha": float(alpha1[i].median()), "cos": float(cos1[i].median()), "gamma2": float(g1sq[i].median()), "cross": float(pc1[i].median()), "cross_negative_fraction": float((pc1[i] < 0).float().mean())},
                                                                        "mlp2": {"alpha": float(alpha2[i].median()), "cos": float(cos2[i].median()), "gamma2": float(g2[i].median()), "cross": float(pc[i].median()), "cross_negative_fraction": float((pc[i] < 0).float().mean())}}
    scored = [kd for kd in K if per[kd]["n"] >= MIN_N]
    report = {"positions": int(W2.shape[0]), "closure_max": closure, "kinds_scored": scored, "per_kind": per}
    print(json.dumps(report, indent=1))
    a1 = {kd: per[kd]["mlp1"]["alpha"] for kd in scored}; a2 = {kd: per[kd]["mlp2"]["alpha"] for kd in scored}
    have = all(k_ in scored for k_ in ("alphabetic", "punctuation")); havenum = "numeric" in scored
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_alphabetic_keeps_more_than_punctuation_mlp1": have and a1["alphabetic"] >= a1["punctuation"] + GAP_MIN, "pred_c_numeric_lowest_at_mlp2": havenum and min(a2, key=a2.get) == "numeric",
                   "pred_d_class_gaps_replay_filler_sign": have and a1["alphabetic"] > a1["punctuation"] and a2["alphabetic"] > a2["punctuation"], "pred_e_mlp2_cross_negative_every_kind": all(per[kd]["mlp2"]["cross_negative_fraction"] >= NEG_MIN for kd in scored)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_kind_gain_natural_result_v321", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
