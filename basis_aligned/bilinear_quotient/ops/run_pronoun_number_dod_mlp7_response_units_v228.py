#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_top_10_units_carry_half pred_c_top_50_units_carry_080 pred_d_largest_responder_opposes_direct
"""Pronoun number they/he DoD (v228): WHICH MLP-7 UNITS compensate? v227: under the MLP-6 trio removal at the noun, MLP 7's response restores 71% of the direct
linear effect on the plural detector 829 (+9.6% of the contrast against the direct -13.5%). Exactly by unit: MLP 7's write delta at the noun is
sum_j (h7_j^edited - h7_j^native) Down7[:, j] + 0 (bias cancels), so unit j's response term is  g . (w7 (delta h7_j) Down7[:, j])  with g the gradient of u_829
at the native block-8 input and w7 = lambda0_8 (the same linearisation as v227; their sum is v227's mlp:07 term -- closure). Pooled plural - singular.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_unit_closure                the unit terms sum to v227's mlp:07 response within relative 1e-3, every row
    pred_b_top_10_units_carry_half     top-10 |pooled| units carry >= 0.50 of the MLP-7 response
    pred_c_top_50_units_carry_080      top-50 >= 0.80
    pred_d_largest_responder_opposes_direct  the largest |pooled| unit has the sign of the MLP-7 response (it pushes back, not with the removal)
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp7_response_units_v228_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp7_response_units_v228"
UNITS, LAYER, UNIT, BATCH = (2483, 2826, 4131), 6, 829, 32
CLOSURE_TOL, TOP10_MIN, TOP50_MIN = 1e-3, 0.50, 0.80
FORWARDS_MAX = 8
WRITERS = ["attn:06", "mlp:06", "attn:07", "mlp:07"] + [f"attnhead:08:{h}" for h in range(9)]
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_top_10_units_carry_half": ">= 0.50", "pred_c_top_50_units_carry_080": ">= 0.80", "pred_d_largest_responder_opposes_direct": "sign of the MLP-7 response"}


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
            if l == 7:
                h7 = dod_units.hidden(model, block.mlp, xin)
                for i in range(len(chunk)): out[i]["h7"] = h7[i, pos[i]].float().clone()
            for i in range(len(chunk)):
                if l in (6, 7): out[i][f"mlp:{l:02d}"] = m[i, pos[i]].float().clone()
            x = x + m
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "writers": WRITERS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN, "top50_min": TOP50_MIN}}
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
    D7 = model.transformer.h[7].mlp.Down.weight.detach().float()
    per_row = []
    for row, nat, ed in zip(rows, native, edited):
        w7 = nat["lambda0:08"]
        x = nat["x8"].clone().requires_grad_(True); xin = F.rms_norm(x, (x.shape[-1],)); u = (Lrow.to(x.device) @ xin) * (Rrow.to(x.device) @ xin); u.backward(); gvec = x.grad.detach()
        gD = (gvec.to(D7.device) @ D7) * w7                                  # per-unit: g . (w7 Down7[:, j])
        dh = (ed["h7"] - nat["h7"]).to(D7.device); terms = gD * dh
        block_term = float(gvec @ (w7 * (ed["mlp:07"] - nat["mlp:07"])))
        closure = abs(float(terms.sum()) - block_term) / max(abs(block_term), 1e-6)
        per_row.append({"closure": closure, "terms": terms.cpu(), "block_term": block_term, "u_native": nat["u"], "exact": ed["u"] - nat["u"]})
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    contrast = sum(per_row[i]["u_native"] - per_row[j]["u_native"] for i, j in pairs)
    pooled = sum(per_row[i]["terms"] - per_row[j]["terms"] for i, j in pairs) / contrast; response = float(pooled.sum())
    order = torch.argsort(pooled.abs(), descending=True); share = lambda k: float(pooled[order[:k]].sum()) / response
    top = [(int(j), float(pooled[j])) for j in order[:40]]
    report = {"u829_contrast": contrast, "mlp7_response_fraction": response, "exact_change_fraction": sum(per_row[i]["exact"] - per_row[j]["exact"] for i, j in pairs) / contrast, "shares": {str(k): share(k) for k in (5, 10, 20, 50, 100, 500)}, "top_units": top, "max_closure": max(r["closure"] for r in per_row)}
    print("MLP-7 response", round(response, 4), "top-10", round(share(10), 3), "top-50", round(share(50), 3), "top", [(j, round(v, 4)) for j, v in top[:10]], "closure", report["max_closure"])
    predictions = {"pred_a_unit_closure": report["max_closure"] <= CLOSURE_TOL, "pred_b_top_10_units_carry_half": share(10) >= TOP10_MIN, "pred_c_top_50_units_carry_080": share(50) >= TOP50_MIN, "pred_d_largest_responder_opposes_direct": (top[0][1] > 0) == (response > 0)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp6_edit_mlp7_response_units_result_v228", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
