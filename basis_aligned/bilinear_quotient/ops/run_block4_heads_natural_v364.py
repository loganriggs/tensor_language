#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_head_sum_closure pred_b_4_5_leads_at_natural_nouns pred_c_4_5_positive_for_plural_on_text pred_d_4_5_separates_cue_rows pred_e_final_position_led_by_4_4_or_4_8
"""Block-4 heads at the cue noun and final token of natural sentences (v364). v357 / v358: on the panel rows head 4.5 carries ~90% of block 4's number write on
9.6's they - he reader at the noun, positive for plural, once a determiner precedes. OOD on the 128 natural verb rows (cue noun at cue_offset; final token):
per-head write projected on r, plural-cue minus singular-cue rows (unpaired means), head shares; 4.5's separation in pooled std.
PREDICTIONS (scored as written; failures preserved; priors from v357)
    pred_a_head_sum_closure            the nine per-head writes sum to block 4's attention output within relative 1e-4 at both positions
    pred_b_4_5_leads_at_natural_nouns  at the cue noun head 4.5 carries the largest |contrast| of the nine
    pred_c_4_5_positive_for_plural_on_text  4.5's cue-noun contrast is positive (they-for-plural), as on the panel
    pred_d_4_5_separates_cue_rows      4.5's cue-noun projection separates plural-cue from singular-cue rows by >= 0.5 pooled std
    pred_e_final_position_led_by_4_4_or_4_8  at the final token the largest head is 4.4 or 4.8 (as on the panel). Prior: unsure -- natural finals vary.
PRICE (registered maximum): 2 natural batches (blocks 0-4) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/block4_heads_natural_v364_result.json"
CANDIDATE_ID = "pronoun_number.block4_heads_natural_v364"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, SEP_MIN, BATCH = 1e-4, 0.5, 64
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_head_sum_closure": "<= 1e-4", "pred_b_4_5_leads_at_natural_nouns": "rank 1", "pred_c_4_5_positive_for_plural_on_text": "positive", "pred_d_4_5_separates_cue_rows": ">= 0.5 std", "pred_e_final_position_led_by_4_4_or_4_8": "4.4 or 4.8"}


def main() -> None:
    rows, he, she, agents, objects = g.build(); recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "natural_rows": len(recs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "sep_min": SEP_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda")
    cue = [int(r_["cue_offset"]) for r_ in recs]; fin = [len(r_["ids"]) - 1 for r_ in recs]; plural = [i for i, r_ in enumerate(recs) if r_["cue"] == "plural"]; sing = [i for i, r_ in enumerate(recs) if r_["cue"] != "plural"]
    proj = {"noun": [], "final": []}; closure = 0.0
    with torch.no_grad():
        for s0 in range(0, len(recs), BATCH):
            tokens = nat[s0:s0 + BATCH]; idx = torch.arange(tokens.shape[0]); pos = {"noun": torch.tensor(cue[s0:s0 + BATCH]), "final": torch.tensor(fin[s0:s0 + BATCH])}
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
    report = {"closure_max": closure}
    for pname in proj:
        P = torch.cat(proj[pname]); per = {f"4.{h}": float(P[plural, h].mean() - P[sing, h].mean()) for h in range(N_HEAD)}; total = sum(abs(v) for v in per.values()); order = sorted(per, key=lambda k: -abs(per[k]))
        s45 = P[:, 5]; std = float(torch.cat([s45[plural] - s45[plural].mean(), s45[sing] - s45[sing].mean()]).std())
        report[pname] = {"mean_contrast": per, "order": order, "shares": {k: abs(per[k]) / total for k in order}, "sep_4_5_over_std": float((s45[plural].median() - s45[sing].median()) / std)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_head_sum_closure": closure <= CLOSURE_TOL, "pred_b_4_5_leads_at_natural_nouns": report["noun"]["order"][0] == "4.5", "pred_c_4_5_positive_for_plural_on_text": report["noun"]["mean_contrast"]["4.5"] > 0,
                   "pred_d_4_5_separates_cue_rows": report["noun"]["sep_4_5_over_std"] >= SEP_MIN, "pred_e_final_position_led_by_4_4_or_4_8": report["final"]["order"][0] in ("4.4", "4.8")}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "block4_heads_natural_result_v364", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
