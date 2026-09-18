#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays_v206 pred_b_split_closure pred_c_radial_part_dominates pred_d_tangential_part_has_feed_sign pred_e_1779_radial_share
"""Pronoun number they/he DoD (v236): is MLP 7's push-back RADIAL? v234 / v235: under the mean-preserving edit of the MLP-6 trio, MLP-7 unit 1779 carries 65% of
MLP 7's restoring response on the plural detector 829 while its own number contrast shrinks 16%. The detector reads x_hat = x_8 / rms(x_8): a write along x_8
itself changes only the rms (rescaling u_829), a write orthogonal to x_8 moves its factors. Exact split of MLP 7's response, per row: with d = lambda0_8 (mlp7_edited -
mlp7_native) at the noun and x = x_8 native, d_par = (d . x_hat) x_hat, d_perp = d - d_par;  tangential = u(x + d_perp) - u(x),  radial = u(x + d) - u(x + d_perp)
(sums to the exact effect of MLP 7's response on u_829). Also for unit 1779 alone (d_1779 = lambda0_8 delta h_1779 Down7[:, 1779]).
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_replays_v206     native pooled they - he margin = 196.62 within relative 1e-3
    pred_b_split_closure             tangential + radial = exact effect of MLP 7's response on u_829 within relative 1e-3, every row (an identity; a code check)
    pred_c_radial_part_dominates     pooled plural - singular: |radial| >= 0.50 x |tangential + radial| for MLP 7's whole response
    pred_d_tangential_part_has_feed_sign  the tangential part is < 0 (a shrinking relay moves the factors AGAINST 829's contrast)
    pred_e_1779_radial_share         for unit 1779 alone the radial part carries >= 0.50 of its response
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
OUT = ROOT / "circuits/followups/pronoun_number_dod_mlp7_response_radial_split_v236_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_mlp7_response_radial_split_v236"
UNITS, LAYER, UNIT, U1779, V206_MARGIN, CLOSURE_TOL, HALF, BATCH = (2483, 2826, 4131), 6, 829, 1779, 196.62, 1e-3, 0.50, 32
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_baseline_replays_v206": "<= 1e-3", "pred_b_split_closure": "<= 1e-3", "pred_c_radial_part_dominates": ">= 0.50", "pred_d_tangential_part_has_feed_sign": "< 0", "pred_e_1779_radial_share": ">= 0.50"}


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
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "half": HALF}}
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
    D7 = model.transformer.h[7].mlp.Down.weight.detach().float(); dev = Lrow.device
    def u_of(x):
        xin = F.rms_norm(x, (x.shape[-1],)); return float((Lrow @ xin) * (Rrow @ xin))
    def split(x, d):
        xh = x / x.norm(); d_par = float(d @ xh) * xh; d_perp = d - d_par
        tang = u_of(x + d_perp) - u_of(x); rad = u_of(x + d) - u_of(x + d_perp); return tang, rad, u_of(x + d) - u_of(x)
    per_row = []
    for nat, ed in zip(native, edited):
        x = nat["x8"].to(dev); lam = nat["lambda0_8"]; d_all = lam * (ed["mlp7"] - nat["mlp7"]).to(dev); d_1779 = lam * float(ed["h7"][U1779] - nat["h7"][U1779]) * D7[:, U1779]
        t_all, r_all, e_all = split(x, d_all); t_u, r_u, e_u = split(x, d_1779)
        per_row.append({"tang": t_all, "rad": r_all, "exact": e_all, "tang_1779": t_u, "rad_1779": r_u, "exact_1779": e_u, "closure": abs(t_all + r_all - e_all) / max(abs(e_all), 1e-6)})
    P = lambda key: sum(per_row[i][key] - per_row[j][key] for i, j in pairs) / u_native
    report = {"margin_native_pooled": m_native, "u829_contrast": u_native, "mlp7_tangential": P("tang"), "mlp7_radial": P("rad"), "mlp7_exact": P("exact"), "u1779_tangential": P("tang_1779"), "u1779_radial": P("rad_1779"), "u1779_exact": P("exact_1779"), "max_closure": max(r["closure"] for r in per_row)}
    print({k: round(v, 4) for k, v in report.items()})
    predictions = {"pred_a_baseline_replays_v206": abs(m_native - V206_MARGIN) / V206_MARGIN <= 1e-3, "pred_b_split_closure": report["max_closure"] <= CLOSURE_TOL, "pred_c_radial_part_dominates": abs(report["mlp7_radial"]) >= HALF * abs(report["mlp7_exact"]),
                   "pred_d_tangential_part_has_feed_sign": report["mlp7_tangential"] < 0, "pred_e_1779_radial_share": abs(report["u1779_radial"]) >= HALF * abs(report["u1779_exact"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_mlp7_response_radial_split_result_v236", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
