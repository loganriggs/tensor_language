#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_writer_closure pred_c_attention_4_leads_at_the_next_token pred_d_mlps_at_next_token_secondary pred_e_embedding_small_at_next_token
"""Who writes the number at the token after the noun? (v383). v382: reader 9.6 takes half its final number write from the key one token after the noun.
That position's state, as 9.6's value reads it, is v = W_v rms(live_9) at that key; the number there was put by earlier writers. Exact writer fold of
9.6's value at the noun+1 key projected through its output onto the they - he unembedding direction: embedding (x0 terms), attention 0-8 totals, MLP 0-8
totals; plural - singular over aligned pairs on the v76 rows. Section 4.8 named head 4.5 as the copier to this site.
PREDICTIONS (scored as written; failures preserved; priors from section 4.8 / v357-v366)
    pred_a_closure                     the manual forward reproduces the they - he margin (2.048) within 1e-3 (instrument)
    pred_b_writer_closure              the writer terms reproduce 9.6's value-borne contrast at the noun+1 key within relative 1e-3 on every pair
    pred_c_attention_4_leads_at_the_next_token  attention 4 is the largest single writer of the number at the noun+1 key
    pred_d_mlps_at_next_token_secondary  MLPs 5-8 together carry >= 0.20 (the copied state is re-processed at that position). Prior: unsure.
    pred_e_embedding_small_at_next_token  the embedding terms carry <= 0.10 (the next token's own identity carries no number)
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
OUT = ROOT / "circuits/followups/next_token_number_writers_v383_result.json"
CANDIDATE_ID = "pronoun_number.next_token_number_writers_v383"
DST, HEAD, BATCH = 9, 6, 32
CLOSURE_TOL, NATIVE_M, WRITER_TOL, MLP_MIN, EMB_MAX = 1e-3, 2.0481, 1e-3, 0.20, 0.10
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_writer_closure": "<= 1e-3", "pred_c_attention_4_leads_at_the_next_token": "largest", "pred_d_mlps_at_next_token_secondary": ">= 0.20", "pred_e_embedding_small_at_next_token": "<= 0.10"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "head": f"{DST}.{HEAD}", "key": "noun+1", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "writer_tol": WRITER_TOL, "mlp_min": MLP_MIN, "emb_max": EMB_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; attn = blocks[DST].attn; D = model.config.n_embd; hd = D // 9
    u = (model.lm_head.weight.detach().float()[L._single(" they")] - model.lm_head.weight.detach().float()[L._single(" he")]).cpu()
    Wv_h = attn.c_v.weight.detach().float()[HEAD * hd:(HEAD + 1) * hd].cpu(); Wo_h = attn.c_proj.weight.detach().float()[:, HEAD * hd:(HEAD + 1) * hd].cpu(); lamv = float(attn.lamb)
    W = {"embedding": []}; W.update({f"attn{l}": [] for l in range(DST + 1)}); W.update({f"mlp{l}": [] for l in range(DST)}); LIVE, V1 = [], []; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pk = torch.tensor([noun_of(r) + 1 for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {k: torch.zeros_like(x) for k in W}; parts["embedding"] = x.clone()
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                if l == DST:
                    LIVE.append(live[idx, pk].float().cpu()); V1.append(v1_[idx, pk, HEAD].float().cpu())
                    for k in W: W[k].append(parts[k][idx, pk].float().cpu())
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention
                if l <= DST: parts[f"attn{l}"] = parts[f"attn{l}"] + attention
                m = block.mlp(F.rms_norm(x, (D,))); x = x + m
                if l < DST: parts[f"mlp{l}"] = parts[f"mlp{l}"] + m
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    LIVE = torch.cat(LIVE); V1 = torch.cat(V1); W = {k: torch.cat(v) for k, v in W.items()}; rms = LIVE.pow(2).mean(1).sqrt()
    # the head's value at the noun+1 key, projected through O_h onto u (pattern weight left out: it is a per-row scalar common to all writers)
    val = lambda M: (((1 - lamv) * (M / rms.unsqueeze(1)) @ Wv_h.T) @ Wo_h.T) @ u
    full = val(LIVE) + ((lamv * V1) @ Wo_h.T) @ u
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    con = lambda t: float((t[plural] - t[sing]).sum())
    terms = {k: val(M) for k, M in W.items()}; terms["embedding"] = terms["embedding"] + ((lamv * V1) @ Wo_h.T) @ u
    recon = sum(terms.values()); wclos = float(((recon - full).abs() / full.abs().clamp_min(1e-6)).max()); tot = con(full)
    share = {k: con(v) / tot for k, v in terms.items()}
    report = {"closure_margin_gap": closure, "writer_closure": wclos, "value_contrast_total": tot, "writer_shares": share, "largest": max(share, key=lambda k: share[k]), "mlps5_8": sum(share[f"mlp{l}"] for l in range(5, 9)), "embedding": share["embedding"]}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_writer_closure": wclos <= WRITER_TOL, "pred_c_attention_4_leads_at_the_next_token": report["largest"] == "attn4", "pred_d_mlps_at_next_token_secondary": report["mlps5_8"] >= MLP_MIN, "pred_e_embedding_small_at_next_token": abs(share["embedding"]) <= EMB_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "next_token_number_writers_result_v383", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
