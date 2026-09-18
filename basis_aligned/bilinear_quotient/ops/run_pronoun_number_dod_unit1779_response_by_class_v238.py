#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_row_split_closure pred_c_gradients_differ_by_class pred_d_singular_rows_carry_the_restoring_response pred_e_plural_rows_lose
"""Pronoun number they/he DoD (v238): WHY a shrinking relay restores the detector. Under the mean-preserving edit of the MLP-6 trio, MLP-7 unit 1779's plural -
singular contrast shrinks 16% (v235) yet its write moves the plural detector 829 in the restoring direction (+0.064 of 829's contrast, v234); its relay sign is genuine
(v237). Resolution candidate: 829's gradient g (through rms_norm, at each row's native x_8) is not the same on plural and singular rows, so the pooled response
sum_pairs [G_P dh_P - G_S dh_S] (G_j = g . lambda0_8 Down7[:, 1779], dh = edited - native h_1779) is not (dh_P - dh_S) x a common G. Reported: the response split into
its plural-row and singular-row halves, the per-class means of G and of dh, and the closure of the split.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_replays_v206     native pooled they - he margin = 196.62 within relative 1e-3
    pred_b_row_split_closure         plural half + singular half = 1779's pooled response (v234: +0.064 of 829's contrast) within relative 1e-3
    pred_c_gradients_differ_by_class |mean G_P - mean G_S| >= 0.50 x |mean G| over all rows
    pred_d_singular_rows_carry_the_restoring_response  the singular-row half of the response is > 0 and >= 0.50 of the total
    pred_e_plural_rows_lose          the plural-row half is < 0 (on plural rows the shrunken relay does move 829 down)
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_unit1779_response_by_class_v238_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_unit1779_response_by_class_v238"
UNITS, LAYER, UNIT, U1779, V206_MARGIN, CLOSURE_TOL, DIFF_MIN, HALF, BATCH = (2483, 2826, 4131), 6, 829, 1779, 196.62, 1e-3, 0.50, 0.50, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_row_split_closure": "<= 1e-3", "pred_c_gradients_differ_by_class": ">= 0.50", "pred_d_singular_rows_carry_the_restoring_response": "> 0, >= 0.50", "pred_e_plural_rows_lose": "< 0"}


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
                h7 = dod_units.hidden(model, block.mlp, xin)
                for i in range(len(chunk)): out[i]["mlp7"] = m[i, pos[i]].float().clone(); out[i]["lambda0_8"] = float(blocks[8].lambdas[0]); out[i]["h7"] = h7[i, pos[i]].float().clone()
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
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "diff_min": DIFF_MIN, "half": HALF}}
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
    D7 = model.transformer.h[7].mlp.Down.weight.detach().float(); per_row = []
    for row, nat, ed in zip(rows, native, edited):
        x = nat["x8"].clone().requires_grad_(True); xin = F.rms_norm(x, (x.shape[-1],)); u = (Lrow.to(x.device) @ xin) * (Rrow.to(x.device) @ xin); u.backward(); gvec = x.grad.detach()
        G = float(gvec.to(D7.device) @ D7[:, U1779]) * nat["lambda0_8"]; dh = float(ed["h7"][U1779] - nat["h7"][U1779])
        per_row.append({"G": G, "dh": dh, "term": G * dh, "present": row.present})
    total = sum(per_row[i]["term"] - per_row[j]["term"] for i, j in pairs) / u_native
    plural_half = sum(per_row[i]["term"] for i, j in pairs) / u_native; singular_half = -sum(per_row[j]["term"] for i, j in pairs) / u_native
    G_P = sum(per_row[i]["G"] for i, j in pairs) / len(pairs); G_S = sum(per_row[j]["G"] for i, j in pairs) / len(pairs); G_all = (G_P + G_S) / 2
    dh_P = sum(per_row[i]["dh"] for i, j in pairs) / len(pairs); dh_S = sum(per_row[j]["dh"] for i, j in pairs) / len(pairs)
    report = {"margin_native_pooled": m_native, "u829_contrast": u_native, "u1779_response_total": total, "plural_half": plural_half, "singular_half": singular_half, "G_plural_mean": G_P, "G_singular_mean": G_S, "dh_plural_mean": dh_P, "dh_singular_mean": dh_S,
              "split_closure": abs(plural_half + singular_half - total) / max(abs(total), 1e-9)}
    print({k: round(v, 4) for k, v in report.items()})
    predictions = {"pred_a_baseline_replays_v206": abs(m_native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_row_split_closure": report["split_closure"] <= CLOSURE_TOL, "pred_c_gradients_differ_by_class": abs(G_P - G_S) >= DIFF_MIN * abs(G_all),
                   "pred_d_singular_rows_carry_the_restoring_response": singular_half > 0 and singular_half >= HALF * total, "pred_e_plural_rows_lose": plural_half < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_unit1779_response_by_class_result_v238", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
