#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_head_sum_closure pred_b_4_5_leads_at_the_final pred_c_4_8_carries_at_the_noun pred_d_top2_at_final pred_e_signs_negative_for_plural
"""Block-4 heads in context on the pronoun rows (v357). v356: from the token alone, block 4's number contrast on head 9.6's they - he reader is carried by
heads 4.8 (43%) and 4.5 (31%); section 4.8 named only 4.5 (the copier at the verb / final site). On the v76 rows: each block-4 head's write at the NOUN
and at the FINAL position, projected on r, plural - singular over aligned pairs; head shares at each position.
PREDICTIONS (scored as written; failures preserved; priors from v205-v215 / v356)
    pred_a_head_sum_closure         the nine per-head writes sum to block 4's attention output within relative 1e-4 at both positions
    pred_b_4_5_leads_at_the_final   at the final position, head 4.5 carries the largest |pooled contrast| (the copier named in section 4.8)
    pred_c_4_8_carries_at_the_noun  at the noun, head 4.8 carries >= 0.30 of the summed |contrast| (its single-token number write is present in context)
    pred_d_top2_at_final            at the final position the top two heads carry >= 0.60 of the summed |contrast|
    pred_e_signs_negative_for_plural  4.5's and 4.8's pooled contrasts are negative at the noun (plural = negative on this reader, v356 / v168)
PRICE (registered maximum): 3 row batches x 1 forward (blocks 0-4) = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/block4_heads_on_rows_v357_result.json"
CANDIDATE_ID = "pronoun_number.block4_heads_on_rows_v357"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, SHARE_48, TOP2_MIN, BATCH = 1e-4, 0.30, 0.60, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_head_sum_closure": "<= 1e-4", "pred_b_4_5_leads_at_the_final": "rank 1", "pred_c_4_8_carries_at_the_noun": ">= 0.30", "pred_d_top2_at_final": ">= 0.60", "pred_e_signs_negative_for_plural": "both < 0 at the noun"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "share_48": SHARE_48, "top2_min": TOP2_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    proj = {"noun": [], "final": []}; closure = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pos = {"noun": torch.tensor([noun_of(r_) for r_ in chunk]), "final": torch.tensor([r_.final for r_ in chunk])}
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: hook.remove()
                    y = captured["y"].float()
                    for pname, p in pos.items():
                        yp = y[idx, p]; per_head = torch.stack([(yp[:, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T) for h in range(N_HEAD)], 1)
                        closure = max(closure, float(((per_head.sum(1) - attention[idx, p].float()).norm(dim=1) / attention[idx, p].float().norm(dim=1)).max()))
                        proj[pname].append((per_head @ r.to(per_head.device)).cpu())
                    break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    report = {"closure_max": closure}
    for pname in proj:
        P = torch.cat(proj[pname]); per = {f"4.{h}": float((P[plural, h] - P[sing, h]).sum()) for h in range(N_HEAD)}; total = sum(abs(v) for v in per.values()); order = sorted(per, key=lambda k: -abs(per[k]))
        report[pname] = {"pooled_contrast": per, "order": order, "shares": {k: abs(per[k]) / total for k in order}, "top2_share": sum(abs(per[k]) for k in order[:2]) / total}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_head_sum_closure": closure <= CLOSURE_TOL, "pred_b_4_5_leads_at_the_final": report["final"]["order"][0] == "4.5", "pred_c_4_8_carries_at_the_noun": report["noun"]["shares"]["4.8"] >= SHARE_48,
                   "pred_d_top2_at_final": report["final"]["top2_share"] >= TOP2_MIN, "pred_e_signs_negative_for_plural": report["noun"]["pooled_contrast"]["4.5"] < 0 and report["noun"]["pooled_contrast"]["4.8"] < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "block4_heads_on_rows_result_v357", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
