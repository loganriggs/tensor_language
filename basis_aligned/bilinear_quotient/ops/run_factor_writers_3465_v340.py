#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_writer_closure pred_c_mlp1_mlp2_carry_half_of_each_factor pred_d_embedding_carries_a_fifth pred_e_attention_3_carries_little
"""Which writers move the two factors of unit 3465 with the noun's number? (v340). v339: on the v76 rows both factors of 3465 carry the plural - singular
contrast (L 30%, R 70%). Each factor is linear in the block-3 input up to the rms scale: L . x3^ = sum_w (L . w) / rms(x3), so the pair contrast of each
factor splits exactly by writer -- embedding (lambda-chain terms of x0), attention 0 / 1 / 2 and MLP 0 / 1 / 2 totals (lambda-scaled), attention 3 --
plus the rms-change term (a scalar per pair, reported). v201 did this for the PRODUCT (mlp:01 + mlp:02 >= 0.50; embedding >= 0.20); here per factor.
PREDICTIONS (scored as written; failures preserved; priors from v201 / v296)
    pred_a_closure                              the manual forward reproduces the model's they - he margin (2.048) within 1e-3 (instrument)
    pred_b_writer_closure                       the writer terms plus the rms term reproduce each factor's pair contrast within relative 1e-3 on every pair
    pred_c_mlp1_mlp2_carry_half_of_each_factor  MLP 1 + MLP 2 carry >= 0.50 of the pooled contrast of BOTH factors of 3465
    pred_d_embedding_carries_a_fifth            the embedding terms carry >= 0.20 of the pooled contrast of each factor of 3465
    pred_e_attention_3_carries_little           attention 3 carries <= 0.10 of each factor's pooled contrast (the factors read the residual the block receives, not the block's own attention). Prior: unsure.
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/factor_writers_3465_v340_result.json"
CANDIDATE_ID = "pronoun_number.factor_writers_3465_v340"
UNIT, BATCH = 3465, 32
CLOSURE_TOL, NATIVE_M, WRITER_TOL, MLP12_MIN, EMB_MIN, ATTN3_MAX = 1e-3, 2.0481, 1e-3, 0.50, 0.20, 0.10
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_writer_closure": "<= 1e-3", "pred_c_mlp1_mlp2_carry_half_of_each_factor": ">= 0.50 x 2", "pred_d_embedding_carries_a_fifth": ">= 0.20 x 2", "pred_e_attention_3_carries_little": "<= 0.10 x 2"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "unit": UNIT, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "writer_tol": WRITER_TOL, "mlp12_min": MLP12_MIN, "emb_min": EMB_MIN, "attn3_max": ATTN3_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; mlp3 = blocks[3].mlp
    Lr, Rr = mlp3.Left.weight.detach().float()[UNIT].cpu(), mlp3.Right.weight.detach().float()[UNIT].cpu()
    W = {"embedding": [], "attn0": [], "mlp0": [], "attn1": [], "mlp1": [], "attn2": [], "mlp2": [], "attn3": []}; X3 = []; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None; parts = {k: torch.zeros_like(x) for k in W}
            parts["embedding"] = x.clone()                       # x0 seeds the stream; its lambda-chain re-injections are added below
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention
                if l <= 3: parts[f"attn{l}"] = parts[f"attn{l}"] + attention
                if l == 3:
                    X3.append(x[idx, pn].float().cpu())
                    for k in W: W[k].append(parts[k][idx, pn].float().cpu())
                m = block.mlp(F.rms_norm(x, (model.config.n_embd,))); x = x + m
                if l < 3: parts[f"mlp{l}"] = parts[f"mlp{l}"] + m
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    X3 = torch.cat(X3); W = {k: torch.cat(v) for k, v in W.items()}; recon = sum(W.values()); assert float(((recon - X3).norm(dim=1) / X3.norm(dim=1)).max()) < 1e-3, "writer sum != x3"
    rms = X3.pow(2).mean(1).sqrt(); partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    per, wclos = {}, 0.0
    for name, vec in (("L", Lr), ("R", Rr)):
        f = (X3 @ vec) / rms; df = f[plural] - f[sing]; tot = float(df.sum()); terms = {}
        for k, w in W.items():
            proj = (w @ vec); terms[k] = float((proj[plural] / rms[plural] - proj[sing] / rms[sing]).sum()) / tot           # each writer's term, with the pair's own rms on each side (exact split; no separate rms term)
        wclos = max(wclos, abs(sum(terms.values()) - 1.0))
        per[name] = {"pooled_contrast": tot, "writer_shares": terms, "mlp1_plus_mlp2": terms["mlp1"] + terms["mlp2"], "embedding": terms["embedding"], "attn3": terms["attn3"]}
    report = {"closure_margin_gap": closure, "writer_closure": wclos, "per_factor": per}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_writer_closure": wclos <= WRITER_TOL, "pred_c_mlp1_mlp2_carry_half_of_each_factor": all(per[n]["mlp1_plus_mlp2"] >= MLP12_MIN for n in ("L", "R")),
                   "pred_d_embedding_carries_a_fifth": all(per[n]["embedding"] >= EMB_MIN for n in ("L", "R")), "pred_e_attention_3_carries_little": all(abs(per[n]["attn3"]) <= ATTN3_MAX for n in ("L", "R"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "factor_writers_3465_result_v340", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
