#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linear_matches_exact pred_c_mlp7_opposes_direct pred_d_attention_response_small pred_e_direct_plus_mlp7_within_020
"""Pronoun number they/he DoD (v227): WHO responds between block 6 and block 8? v226: the in-place leave-out of the MLP-6 trio {2483, 2826, 4131} predicts
15-28% of u_829's contrast, the edit (v196) removes 6.3%. Response census AT THE NOUN: native vs edited forward (trio zeroed at the noun), capturing at the
noun the writes of attn:06, mlp:06, attn:07, mlp:07 and the nine block-8 heads, plus the block-8 input x_8 (before MLP 8) and u_829. Exact split: delta x_8 =
sum of scaled write deltas (lambda0 recurrence; closure). Linearised effect on u_829 of each writer's delta: g . (w delta) with g = grad u_829 at the native
x_8 (rms included); DIRECT = mlp:06's delta (the removed units themselves, plus MLP 6's own renormalisation), RESPONSES = attn:07, mlp:07, block-8 heads.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_recurrence_closure        the writer deltas reconstruct delta x_8 within relative 1e-3, every row
    pred_b_linear_matches_exact      |linear total - exact delta u_829| <= 0.25 x |exact| (pooled; the product is quadratic, so the linearisation may miss)
    pred_c_mlp7_opposes_direct       mlp:07's response has the sign opposite to the direct term and |response| >= 0.30 x |direct|
    pred_d_attention_response_small  |attn:07 + block-8 heads| <= 0.20 x |direct|
    pred_e_direct_plus_mlp7_within_020  |direct + mlp:07 - exact| <= 0.20 x |exact| (the two terms are the story)
PRICE (registered maximum): 3 batches x (native + edited) = 6 forwards; gradients on captured 1152-d vectors only; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp6_edit_noun_response_v227_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp6_edit_noun_response_v227"
UNITS, LAYER, UNIT, BATCH = (2483, 2826, 4131), 6, 829, 32
CLOSURE_TOL, LIN_TOL, OPPOSE_MIN, ATTN_MAX, STORY_TOL = 1e-3, 0.25, 0.30, 0.20, 0.20
FORWARDS_MAX = 8
WRITERS = ["attn:06", "mlp:06", "attn:07", "mlp:07"] + [f"attnhead:08:{h}" for h in range(9)]
PREDICTIONS = {"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linear_matches_exact": "<= 0.25", "pred_c_mlp7_opposes_direct": "opposite, >= 0.30", "pred_d_attention_response_small": "<= 0.20", "pred_e_direct_plus_mlp7_within_020": "<= 0.20"}


def trace(backend, fw, chunk, noun_of, edit):
    torch, F, model = backend.torch, backend.F, backend.model
    tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk)); out = [dict() for _ in chunk]; pre = {}
    blocks = model.transformer.h
    def hook(_m, args): pre["z"] = args[0].detach().clone(); return None
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            handle = block.attn.c_proj.register_forward_pre_hook(hook) if l == 8 else None
            try: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            finally:
                if handle: handle.remove()
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            for i in range(len(chunk)):
                if l in (6, 7): out[i][f"attn:{l:02d}"] = attention[i, pos[i]].float().clone()
                out[i][f"lambda0:{l:02d}"] = float(block.lambdas[0])
            if l == 8:
                W = block.attn.c_proj.weight.detach().float()
                for i in range(len(chunk)):
                    z = pre["z"][i, pos[i]].float()
                    for h in range(9): out[i][f"attnhead:08:{h}"] = W[:, h * L.HEAD_DIM:(h + 1) * L.HEAD_DIM] @ z[h * L.HEAD_DIM:(h + 1) * L.HEAD_DIM]
                    out[i]["x8"] = x[i, pos[i]].float().clone(); out[i]["u"] = float(dod_units.hidden(model, block.mlp, xin)[i, pos[i], UNIT])
                break
            if l == LAYER and edit:
                h = dod_units.hidden(model, block.mlp, xin).clone(); ui = torch.tensor(list(UNITS), device=h.device)
                for i in range(len(chunk)): h[i, pos[i], ui] = 0
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else:
                m = block.mlp(xin)
            for i in range(len(chunk)):
                if l in (6, 7): out[i][f"mlp:{l:02d}"] = m[i, pos[i]].float().clone()
            x = x + m
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "lin_tol": LIN_TOL, "oppose_min": OPPOSE_MIN, "attn_max": ATTN_MAX, "story_tol": STORY_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); forwards = 0; native, edited = [], []
    mlp8 = model.transformer.h[8].mlp; Lrow, Rrow = mlp8.Left.weight.detach().float()[UNIT], mlp8.Right.weight.detach().float()[UNIT]
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        native.extend(trace(backend, fw, chunk, noun_of, False)); forwards += 1
        edited.extend(trace(backend, fw, chunk, noun_of, True)); forwards += 1
    per_row = []
    for row, nat, ed in zip(rows, native, edited):
        w6 = nat["lambda0:07"] * nat["lambda0:08"]; w7 = nat["lambda0:08"]
        weights = {"attn:06": w6, "mlp:06": w6, "attn:07": w7, "mlp:07": w7, **{f"attnhead:08:{h}": 1.0 for h in range(9)}}
        deltas = {m: ed[m] - nat[m] for m in WRITERS}; recon = sum(weights[m] * deltas[m] for m in WRITERS); true = ed["x8"] - nat["x8"]
        closure = float((recon - true).norm() / max(float(true.norm()), 1e-9))
        x = nat["x8"].clone().requires_grad_(True); xin = F.rms_norm(x, (x.shape[-1],)); u = (Lrow.to(x.device) @ xin) * (Rrow.to(x.device) @ xin); u.backward(); gvec = x.grad.detach()
        attribution = {m: float(gvec @ (weights[m] * deltas[m])) for m in WRITERS}
        per_row.append({"closure": closure, "exact": ed["u"] - nat["u"], "linear": sum(attribution.values()), "attribution": attribution, "u_native": nat["u"]})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    pooled = lambda key: sum(per_row[i][key] - per_row[j][key] for i, j in pairs)
    contrast = pooled("u_native"); exact = pooled("exact") / contrast; linear = pooled("linear") / contrast
    attr = {m: sum(per_row[i]["attribution"][m] - per_row[j]["attribution"][m] for i, j in pairs) / contrast for m in WRITERS}
    direct = attr["mlp:06"]; mlp7 = attr["mlp:07"]; attn = attr["attn:07"] + sum(attr[f"attnhead:08:{h}"] for h in range(9)) + attr["attn:06"]
    report = {"u829_contrast": contrast, "exact_change_fraction": exact, "linear_total_fraction": linear, "attribution_fraction": attr, "direct_mlp6": direct, "mlp7_response": mlp7, "attention_response": attn, "max_closure": max(r["closure"] for r in per_row)}
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report.items() if k != "attribution_fraction"}); print({m: round(v, 4) for m, v in attr.items() if abs(v) > 1e-3})
    predictions = {"pred_a_recurrence_closure": report["max_closure"] <= CLOSURE_TOL, "pred_b_linear_matches_exact": abs(linear - exact) <= LIN_TOL * abs(exact), "pred_c_mlp7_opposes_direct": (mlp7 * direct < 0) and abs(mlp7) >= OPPOSE_MIN * abs(direct),
                   "pred_d_attention_response_small": abs(attn) <= ATTN_MAX * abs(direct), "pred_e_direct_plus_mlp7_within_020": abs(direct + mlp7 - exact) <= STORY_TOL * abs(exact)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp6_edit_noun_response_result_v227", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
