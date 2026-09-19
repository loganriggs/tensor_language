#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_key_split_closure pred_b_few_positions_carry_the_rest pred_c_12_4_top_key_is_not_the_noun pred_d_top_non_noun_key_is_after_the_noun pred_e_cut_hits_non_noun_keys_too
"""Which positions carry the copied number to the readers? (v382). v381: reader 9.6 takes 52% and 12.4 76% of their final number write from keys other than
the noun. Exact per-key split of each reader's final-token write (pattern x value per key, projected on the they - he unembedding direction), pooled plural -
singular per key position measured relative to the noun (offset = key - noun: negative = before the noun, 0 = the noun, positive = after), native and under the
six-unit chain cut. Which offsets carry the number, and does the cut remove it there too.
PREDICTIONS (scored as written; failures preserved; priors from section 4.8 / v381)
    pred_a_key_split_closure          the per-key terms sum to the head's write within relative 1e-4 on every row, both heads, both conditions
    pred_b_few_positions_carry_the_rest  for each reader, the top 3 non-noun offsets carry >= 0.60 of the non-noun contrast
    pred_c_12_4_top_key_is_not_the_noun  12.4's largest single offset by |contrast| is not the noun
    pred_d_top_non_noun_key_is_after_the_noun  for each reader the largest non-noun offset is positive (a copy downstream of the noun -- the verb site -- rather than the determiner). Prior: unsure.
    pred_e_cut_hits_non_noun_keys_too under the chain cut, the non-noun keys' summed contrast falls for both readers
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
OUT = ROOT / "circuits/followups/reader_key_positions_v382_result.json"
CANDIDATE_ID = "pronoun_number.reader_key_positions_v382"
N_HEAD = 9; HEADS = ((9, 6), (12, 4))
CHAIN = {3: (3465, 493), 5: (1036,), 6: (2483,), 7: (1779,), 8: (829,)}
CLOSURE_TOL, TOP3_MIN, BATCH = 1e-4, 0.60, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_key_split_closure": "<= 1e-4", "pred_b_few_positions_carry_the_rest": ">= 0.60 x 2", "pred_c_12_4_top_key_is_not_the_noun": "not offset 0", "pred_d_top_non_noun_key_is_after_the_noun": "offset > 0 x 2", "pred_e_cut_hits_non_noun_keys_too": "falls x 2"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top3_min": TOP3_MIN}, "heads": [list(h) for h in HEADS], "chain": {str(k): list(v) for k, v in CHAIN.items()}}
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
        out = {h: {} for h in HEADS}; tot = {h: [] for h in HEADS}; closure = 0.0
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
                            Wo = Wp[:, h * hd:(h + 1) * hd]; y = captured["y"][idx, pf].float(); total = (y[:, h * hd:(h + 1) * hd] @ Wo.T) @ u.to(y.device)
                            pfh = pat[idx, h, pf]                                                            # [B, T]
                            per_key = pfh.unsqueeze(2) * v[:, :, h] @ Wo.T @ u.to(v.device)                   # [B, T] each key's term on the reader
                            closure = max(closure, float(((per_key.sum(1) - total).abs() / total.abs().clamp_min(1e-6)).max()))
                            for i, r_ in enumerate(chunk):
                                for j in range(int(r_.final) + 1):
                                    out[(l, h)].setdefault(j - int(pn[i]), []).append((i + start, float(per_key[i, j])))
                            tot[(l, h)].append(total.cpu())
                    else: attention, v1_ = block.attn(xin_a, v1_)
                    x = live + attention; xin = F.rms_norm(x, (D,))
                    if edits and l in edits:
                        hh = dod_units.hidden(model, block.mlp, xin); hh[:, :, torch.tensor(list(edits[l]), device=hh.device)] = 0; x = x + block.mlp.Down(hh) + block.mlp.Down_bias
                    else: x = x + block.mlp(xin)
        return out, closure
    O0, c0 = run(None); forwards += 3; O1, c1 = run(CHAIN); forwards += 3; closure = max(c0, c1)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = set(i for i, r_ in enumerate(rows) if r_.present); sing_of = {i: partner[(rows[i].construction, rows[i].group, False)] for i in plural}
    def pooled(o):
        res = {}
        for off, vals in o.items():
            d = dict(vals); res[off] = sum(d[i] - d[sing_of[i]] for i in plural if i in d and sing_of[i] in d)
        return res
    report = {"closure_max": closure}
    for h in HEADS:
        key = f"{h[0]}.{h[1]}"; p0, p1 = pooled(O0[h]), pooled(O1[h]); non = {k: v for k, v in p0.items() if k != 0}; order = sorted(non, key=lambda k: -abs(non[k])); tot_non = sum(abs(v) for v in non.values())
        report[key] = {"by_offset_native": {str(k): p0[k] for k in sorted(p0)}, "by_offset_cut": {str(k): p1[k] for k in sorted(p1)}, "top3_non_noun": order[:3], "top3_share_of_non_noun": sum(abs(non[k]) for k in order[:3]) / tot_non, "largest_offset": max(p0, key=lambda k: abs(p0[k])),
                       "non_noun_sum_native": sum(non.values()), "non_noun_sum_cut": sum(v for k, v in p1.items() if k != 0)}
    print(json.dumps({k: ({a_: b_ for a_, b_ in v.items() if not a_.startswith("by_offset")} if isinstance(v, dict) else v) for k, v in report.items()}, indent=1))
    r96, r124 = report["9.6"], report["12.4"]
    predictions = {"pred_a_key_split_closure": closure <= CLOSURE_TOL, "pred_b_few_positions_carry_the_rest": all(r["top3_share_of_non_noun"] >= TOP3_MIN for r in (r96, r124)), "pred_c_12_4_top_key_is_not_the_noun": r124["largest_offset"] != 0,
                   "pred_d_top_non_noun_key_is_after_the_noun": all(r["top3_non_noun"][0] > 0 for r in (r96, r124)), "pred_e_cut_hits_non_noun_keys_too": all(abs(r["non_noun_sum_cut"]) < abs(r["non_noun_sum_native"]) for r in (r96, r124))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "reader_key_positions_result_v382", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
