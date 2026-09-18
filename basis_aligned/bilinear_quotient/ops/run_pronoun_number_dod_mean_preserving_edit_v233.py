#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_mean_preserving_removes_more_than_zeroing pred_c_829_drops_at_least_010 pred_d_mlp7_response_shrinks pred_e_margin_drops
"""Pronoun number they/he DoD (v233): a MEAN-PRESERVING edit decides v232's mechanism. Zeroing the MLP-6 trio {2483, 2826, 4131} at the noun removes a write that is
large in both members of a pair; v232 showed the resulting push-back of MLP-7 unit 1779 is mean-driven and RMS-driven (first-order +0.15 of its contrast, exact -0.36).
Here the trio's hidden values at the noun are set to their PAIR MEAN (h <- (h_plural + h_singular)/2, per aligned pair): the plural - singular difference is removed,
the mean write and the block-7 input norm are kept. Registered readings: the plural detector 829 loses MORE than under zeroing (6.3%, v196) -- near the
first-order MLP-6 carriage into 829 (v182 / v188: 20% x 57% ~ 11%) -- because the compensation is switched off; MLP 7's response at the noun (v227's
quantity, linearised on u_829) shrinks in magnitude; the they - he margin drops. Two passes: native (to read the trio's h at the noun and pair them) and edited.
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays_v206              native pooled they - he margin = 196.62 within relative 1e-3
    pred_b_mean_preserving_removes_more_than_zeroing  |u_829 contrast drop| under the mean-preserving edit > 0.063 (v196's zeroing)
    pred_c_829_drops_at_least_010             the drop is >= 0.10 of u_829's contrast
    pred_d_mlp7_response_shrinks              |MLP 7's linearised response on u_829| <= 0.50 x v227's +0.096
    pred_e_margin_drops                       the they - he margin drops by > 0.005 (zeroing: 0.0067, v196)
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_mean_preserving_edit_v233_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mean_preserving_edit_v233"
UNITS, LAYER, UNIT, V206_MARGIN, V196_DROP, V227_MLP7, DROP_MIN, SHRINK, MARGIN_MIN, BATCH = (2483, 2826, 4131), 6, 829, 196.62, 0.0633, 0.0963, 0.10, 0.50, 0.005, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_mean_preserving_removes_more_than_zeroing": "> 0.063", "pred_c_829_drops_at_least_010": ">= 0.10", "pred_d_mlp7_response_shrinks": "<= 0.5 x 0.096", "pred_e_margin_drops": "> 0.005"}


def run(backend, fw, chunk, noun_of, he, she, target=None):
    """Native forward; if target (per-row 3-vectors) is given, set the trio's hidden values at the noun to it. Returns per-row h6[trio], u_829 at the noun,
    the block-8 input x8 and MLP-7 write at the noun (for the linearised MLP-7 response), and the they - he margin."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens = fw._tokens(chunk); pos = [noun_of(r) for r in chunk]; idx = torch.arange(len(chunk)); out = [dict() for _ in chunk]; blocks = model.transformer.h
    ui = torch.tensor(list(UNITS))
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == LAYER:
                h = dod_units.hidden(model, block.mlp, xin).clone()
                for i in range(len(chunk)):
                    out[i]["h_trio"] = h[i, pos[i], ui.to(h.device)].float().cpu().clone()
                    if target is not None: h[i, pos[i], ui.to(h.device)] = target[i].to(h.device, h.dtype)
                m = block.mlp.Down(h) + block.mlp.Down_bias
            else:
                m = block.mlp(xin)
            if l == 7:
                for i in range(len(chunk)): out[i]["mlp7"] = m[i, pos[i]].float().clone(); out[i]["lambda0_8"] = float(blocks[8].lambdas[0])
            if l == 8:
                for i in range(len(chunk)): out[i]["x8"] = x[i, pos[i]].float().clone(); out[i]["u"] = float(dod_units.hidden(model, block.mlp, xin)[i, pos[i], UNIT])
            x = x + m
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
        fin = torch.tensor([r.final for r in chunk]); lg = logits[idx, fin].float()
        for i in range(len(chunk)): out[i]["margin"] = float(lg[i, he] - lg[i, she])
    return out


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"v196_drop": V196_DROP, "v227_mlp7": V227_MLP7, "drop_min": DROP_MIN, "shrink": SHRINK, "margin_min": MARGIN_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend); forwards = 0
    native = []
    for start in range(0, len(rows), BATCH):
        native.extend(run(backend, fw, rows[start:start + BATCH], noun_of, he, she)); forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    target = [(native[i]["h_trio"] + native[partner[(row.construction, row.group, not row.present)]]["h_trio"]) / 2 for i, row in enumerate(rows)]
    edited = []
    for start in range(0, len(rows), BATCH):
        edited.extend(run(backend, fw, rows[start:start + BATCH], noun_of, he, she, target[start:start + BATCH])); forwards += 1
    mlp8 = model.transformer.h[8].mlp; Lrow, Rrow = mlp8.Left.weight.detach().float()[UNIT], mlp8.Right.weight.detach().float()[UNIT]
    pairs = [(i, partner[(row.construction, row.group, False)]) for i, row in enumerate(rows) if row.present]
    def pooled(lst, key): return sum(lst[i][key] - lst[j][key] for i, j in pairs)
    m_native, m_edit = pooled(native, "margin"), pooled(edited, "margin"); u_native, u_edit = pooled(native, "u"), pooled(edited, "u")
    resp = []
    for nat, ed in zip(native, edited):
        x = nat["x8"].clone().requires_grad_(True); xin = F.rms_norm(x, (x.shape[-1],)); u = (Lrow.to(x.device) @ xin) * (Rrow.to(x.device) @ xin); u.backward()
        resp.append(float(x.grad.detach() @ (nat["lambda0_8"] * (ed["mlp7"] - nat["mlp7"]))))
    mlp7_response = sum(resp[i] - resp[j] for i, j in pairs) / u_native
    drop = (u_native - u_edit) / u_native; margin_drop = (m_native - m_edit) / m_native
    report = {"margin_native_pooled": m_native, "margin_drop_fraction": margin_drop, "u829_contrast": u_native, "u829_drop_fraction": drop, "mlp7_linearised_response_fraction": mlp7_response,
              "difference_removed_check": float(sum(((edited[i]["h_trio"] - edited[j]["h_trio"]).abs().sum()) for i, j in pairs))}
    print({k: round(v, 4) for k, v in report.items()})
    predictions = {"pred_a_baseline_replays_v206": abs(m_native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_mean_preserving_removes_more_than_zeroing": drop > V196_DROP, "pred_c_829_drops_at_least_010": drop >= DROP_MIN,
                   "pred_d_mlp7_response_shrinks": abs(mlp7_response) <= SHRINK * V227_MLP7, "pred_e_margin_drops": margin_drop > MARGIN_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mean_preserving_edit_result_v233", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
