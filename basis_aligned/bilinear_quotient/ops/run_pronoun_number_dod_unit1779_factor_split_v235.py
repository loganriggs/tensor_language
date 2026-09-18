#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_factor_closure pred_c_factors_move_oppositely pred_d_one_factor_dominates pred_e_1779_contrast_moves_against_native
"""Pronoun number they/he DoD (v235): the FACTORS of unit 1779's restoring response. v234: under the mean-preserving edit of the MLP-6 trio at the noun, MLP-7 unit
1779 carries 65% of MLP 7's push-back on the plural detector although the trio feeds it (v231). For a bilinear unit u = (L.x)(R.x) with x the normalised block-7
input at the noun, the exact change under the edit splits as  delta u = (delta a) b_bar + a_bar (delta b)  with a = L.x, b = R.x, delta = edited - native and
bar = the mean of the two states (exact for a product). Pooled plural - singular over the 48 pairs, as fractions of 1779's native contrast (-715): if the two
factor terms have opposite signs and the restoring one dominates, the response is an internal cancellation of the unit's own two reads of the number.
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays_v206         native pooled they - he margin = 196.62 within relative 1e-3
    pred_b_factor_closure                the two factor terms sum to the exact change of u_1779 within relative 1e-3, every row
    pred_c_factors_move_oppositely       the pooled Left-factor term and Right-factor term have opposite signs
    pred_d_one_factor_dominates          the larger |factor term| is >= 1.5 x the smaller
    pred_e_1779_contrast_moves_against_native  the pooled change of u_1779's plural - singular contrast has the sign opposite to its native contrast's first-order loss (i.e. the unit's number contrast GROWS in magnitude or flips toward restoring): pooled delta u / native contrast < 0 means shrink; registered: > 0 (grows)
PRICE (registered maximum): 3 batches x (native + edited) = 6 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit1779_factor_split_v235_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit1779_factor_split_v235"
UNITS, LAYER, UNIT, U1779, V206_MARGIN, CLOSURE_TOL, DOMINATE, BATCH = (2483, 2826, 4131), 6, 829, 1779, 196.62, 1e-3, 1.5, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_factor_closure": "<= 1e-3", "pred_c_factors_move_oppositely": "opposite signs", "pred_d_one_factor_dominates": ">= 1.5 x", "pred_e_1779_contrast_moves_against_native": "> 0"}


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
                for i in range(len(chunk)): out[i]["xin7"] = xin[i, pos[i]].float().clone()
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
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "dominate": DOMINATE}}
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
    mlp7 = model.transformer.h[7].mlp; L7, R7 = mlp7.Left.weight.detach().float()[U1779], mlp7.Right.weight.detach().float()[U1779]
    per_row = []
    for nat, ed in zip(native, edited):
        aN, bN = float(L7 @ nat["xin7"].to(L7.device)), float(R7 @ nat["xin7"].to(L7.device)); aE, bE = float(L7 @ ed["xin7"].to(L7.device)), float(R7 @ ed["xin7"].to(L7.device))
        left = (aE - aN) * (bE + bN) / 2; right = (aE + aN) / 2 * (bE - bN); exact = aE * bE - aN * bN
        per_row.append({"u": aN * bN, "left": left, "right": right, "exact": exact, "closure": abs(left + right - exact) / max(abs(exact), 1e-6)})
    c1779 = sum(per_row[i]["u"] - per_row[j]["u"] for i, j in pairs)
    left = sum(per_row[i]["left"] - per_row[j]["left"] for i, j in pairs) / c1779; right = sum(per_row[i]["right"] - per_row[j]["right"] for i, j in pairs) / c1779; exact = sum(per_row[i]["exact"] - per_row[j]["exact"] for i, j in pairs) / c1779
    report = {"margin_native_pooled": m_native, "margin_drop_fraction": (m_native - m_edit) / m_native, "u829_drop_fraction": (u_native - u_edit) / u_native, "u1779_contrast": c1779, "left_factor_term": left, "right_factor_term": right, "exact_change_fraction": exact, "max_closure": max(r["closure"] for r in per_row)}
    print({k: round(v, 4) for k, v in report.items()})
    big, small = max(abs(left), abs(right)), min(abs(left), abs(right))
    predictions = {"pred_a_baseline_replays_v206": abs(m_native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_factor_closure": report["max_closure"] <= CLOSURE_TOL, "pred_c_factors_move_oppositely": left * right < 0, "pred_d_one_factor_dominates": big >= DOMINATE * max(small, 1e-9), "pred_e_1779_contrast_moves_against_native": exact > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit1779_factor_split_result_v235", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
