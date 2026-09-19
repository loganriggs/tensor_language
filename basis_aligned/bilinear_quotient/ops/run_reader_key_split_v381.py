#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_key_split_closure pred_b_9_6_reads_the_noun pred_c_12_4_reads_elsewhere pred_d_cut_falls_on_the_noun_key pred_e_noun_key_share_of_9_6_loss
"""Where the readers read from (v381). v380: at the final token heads 12.4 and 9.6 each write ~2050 of number contrast on the they - he unembedding direction,
15.1 ~1090; the six-unit MLP chain cut costs 9.6 19% and 12.4 5%. Exact key split of each reader's write at the final token: sum over keys j of
p(final, j) . O_h V_h v_j -- grouped as the NOUN key, all other keys -- native and under the chain cut, projected on the logit reader; plural - singular per pair.
PREDICTIONS (scored as written; failures preserved; priors from v82 / v380)
    pred_a_key_split_closure       noun-key term + other-keys term = the head's write within relative 1e-4 on every row, both heads, both conditions
    pred_b_9_6_reads_the_noun      natively, 9.6's noun-key term carries >= 0.60 of its contrast
    pred_c_12_4_reads_elsewhere    natively, 12.4's noun-key term carries <= 0.50 of its contrast (its number comes from other positions -- the verb site of section 4.8 or the determiner)
    pred_d_cut_falls_on_the_noun_key  under the chain cut, >= 0.70 of 9.6's contrast loss is on the noun-key term
    pred_e_noun_key_share_of_9_6_loss  the noun-key term's own relative loss for 9.6 is >= 0.20 (the noun state it copies lost a fifth). Prior: unsure.
PRICE (registered maximum): 3 row batches x (1 native + 1 edited) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/reader_key_split_v381_result.json"
CANDIDATE_ID = "pronoun_number.reader_key_split_v381"
N_HEAD = 9; HEADS = ((9, 6), (12, 4))
CHAIN = {3: (3465, 493), 5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}
CLOSURE_TOL, NOUN_96, NOUN_124, LOSS_ON_NOUN, NOUN_LOSS, BATCH = 1e-4, 0.60, 0.50, 0.70, 0.20, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_key_split_closure": "<= 1e-4", "pred_b_9_6_reads_the_noun": ">= 0.60", "pred_c_12_4_reads_elsewhere": "<= 0.50", "pred_d_cut_falls_on_the_noun_key": ">= 0.70", "pred_e_noun_key_share_of_9_6_loss": ">= 0.20"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "noun_96": NOUN_96, "noun_124": NOUN_124, "loss_on_noun": LOSS_ON_NOUN, "noun_loss": NOUN_LOSS}, "heads": [list(h) for h in HEADS], "chain": {str(k): list(v) for k, v in CHAIN.items()}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    D = model.config.n_embd; hd = D // N_HEAD; u = (model.lm_head.weight.detach().float()[L._single(" they")] - model.lm_head.weight.detach().float()[L._single(" he")]).cpu()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    store = {}
    def wrap(orig, l):
        def f(q, k, v, q2, k2):
            B, T, H, Dh = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dh) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dh)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0); store[l] = (pat.float(), v.float()); return orig(q, k, v, q2, k2)
        return f
    def run(edits):
        out = {h: {"noun": [], "other": [], "total": []} for h in HEADS}; closure = 0.0
        with torch.no_grad():
            for start in range(0, len(rows), BATCH):
                chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pf = torch.tensor([r_.final for r_ in chunk]); pn = torch.tensor([noun_of(r_) for r_ in chunk])
                x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,)); heads_here = [h for (bl, h) in HEADS if bl == l]
                    if heads_here:
                        orig = block.attn.squared_attention; block.attn.squared_attention = wrap(orig, l); captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                        try: attention, v1_ = block.attn(xin_a, v1_)
                        finally: hook.remove(); block.attn.squared_attention = orig
                        pat, v = store[l]; Wp = block.attn.c_proj.weight.detach().float()
                        for h in heads_here:
                            Wo = Wp[:, h * hd:(h + 1) * hd]; y = captured["y"][idx, pf].float(); total = y[:, h * hd:(h + 1) * hd] @ Wo.T
                            pfh = pat[idx, h, pf]                                    # [B, T] pattern row at the final query
                            noun_term = (pfh[idx, pn].unsqueeze(1) * v[idx, pn, h]) @ Wo.T
                            allsum = torch.einsum("bt,bthd->bhd", pfh, v[:, :, h:h + 1]).squeeze(1) @ Wo.T; other = allsum - noun_term
                            closure = max(closure, float(((noun_term + other - total).norm(dim=1) / total.norm(dim=1)).max()))
                            out[(l, h)]["noun"].append((noun_term @ u.to(noun_term.device)).cpu()); out[(l, h)]["other"].append((other @ u.to(other.device)).cpu()); out[(l, h)]["total"].append((total @ u.to(total.device)).cpu())
                    else: attention, v1_ = block.attn(xin_a, v1_)
                    x = live + attention; xin = F.rms_norm(x, (D,))
                    if edits and l in edits:
                        hh = dod_units.hidden(model, block.mlp, xin); hh[:, :, torch.tensor(list(edits[l]), device=hh.device)] = 0; x = x + block.mlp.Down(hh) + block.mlp.Down_bias
                    else: x = x + block.mlp(xin)
        return {h: {k: torch.cat(v) for k, v in d.items()} for h, d in out.items()}, closure
    P0, c0 = run(None); forwards += 3; P1, c1 = run(CHAIN); forwards += 3; closure = max(c0, c1)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    con = lambda t: float((t[plural] - t[sing]).sum())
    report = {"closure_max": closure}
    for h in HEADS:
        key = f"{h[0]}.{h[1]}"; n0, o0, t0 = con(P0[h]["noun"]), con(P0[h]["other"]), con(P0[h]["total"]); n1, o1, t1 = con(P1[h]["noun"]), con(P1[h]["other"]), con(P1[h]["total"])
        report[key] = {"native": {"noun": n0, "other": o0, "total": t0, "noun_share": n0 / t0}, "cut": {"noun": n1, "other": o1, "total": t1}, "loss_total": t1 - t0, "loss_on_noun_share": (n1 - n0) / (t1 - t0) if t1 != t0 else None, "noun_rel_loss": (n0 - n1) / abs(n0)}
    print(json.dumps(report, indent=1))
    r96, r124 = report["9.6"], report["12.4"]
    predictions = {"pred_a_key_split_closure": closure <= CLOSURE_TOL, "pred_b_9_6_reads_the_noun": r96["native"]["noun_share"] >= NOUN_96, "pred_c_12_4_reads_elsewhere": r124["native"]["noun_share"] <= NOUN_124,
                   "pred_d_cut_falls_on_the_noun_key": (r96["loss_on_noun_share"] or 0) >= LOSS_ON_NOUN, "pred_e_noun_key_share_of_9_6_loss": r96["noun_rel_loss"] >= NOUN_LOSS}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "reader_key_split_result_v381", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
