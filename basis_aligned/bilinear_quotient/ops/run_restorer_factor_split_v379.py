#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_factor_closure pred_c_rise_is_on_one_factor pred_d_rise_factor_reads_the_missing_state pred_e_other_factor_stable
"""How the restorer rises (v379). v377 / v378: under the MLP-3 population cut, MLP-7 unit 1779's plural - singular contrast grows 17% (random tens 1.7%). The
class-wise law (section 4.9): the change of a bilinear unit's contrast splits exactly as delta(h_p - h_s) = delta(L)_p R_p - delta(L)_s R_s ... computed here
directly: for native and edited passes, the per-pair L and R factors of 1779 at the noun; the contrast change decomposed into the L-borne term
(L_edit - L_nat) x R_nat and the R-borne term L_edit x (R_edit - R_nat), per pair, summed; the share on each; and each factor's own plural - singular
contrast native vs edited.
PREDICTIONS (scored as written; failures preserved; priors from section 4.9)
    pred_a_baseline_replays          the native margin replays 2.048 within 1e-3
    pred_b_factor_closure            the two terms sum to the contrast change within relative 1e-3 on every pair
    pred_c_rise_is_on_one_factor     one term carries >= 0.70 of the summed change
    pred_d_rise_factor_reads_the_missing_state  the factor carrying the rise has its own plural - singular contrast change by >= 0.10 relative (it reads the removed MLP-3 state and moves)
    pred_e_other_factor_stable       the other factor's own contrast changes by < 0.05 relative. Prior: unsure.
PRICE (registered maximum): 3 census batches + 3 row batches x (1 native + 1 edit) = 9 forwards; 0 backwards; 0 fits. Bar <= 11.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/restorer_factor_split_v379_result.json"
CANDIDATE_ID = "pronoun_number.restorer_factor_split_v379"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, K = 12, 372, 32, 10
NATIVE_M, M_TOL, FACTOR_TOL, ONE_MIN, MOVE_MIN, STABLE = 2.0481, 1e-3, 1e-3, 0.70, 0.10, 0.05
FORWARDS_MAX = 11
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_factor_closure": "<= 1e-3", "pred_c_rise_is_on_one_factor": ">= 0.70", "pred_d_rise_factor_reads_the_missing_state": ">= 0.10 relative", "pred_e_other_factor_stable": "< 0.05 relative"}


def margins_multi(backend, fw, rows, edits, readers):
    """they - he margins with `edits` = {layer: units} zeroed at every position (several layers in one forward)."""
    torch, F, model = backend.torch, backend.F, backend.model; tokens = fw._tokens(rows)
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if edits and l in edits:
                h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return [{name: float(logits[i, row.final, a] - logits[i, row.final, b]) for name, (a, b) in readers.items()} for i, row in enumerate(rows)]


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "targets": {str(k): list(v) for k, v in TARGETS.items()}, "k": K, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "factor_tol": FACTOR_TOL, "one_min": ONE_MIN, "move_min": MOVE_MIN, "stable": STABLE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]
    # census pass: hidden of MLPs 3, 5, 6, 7, 8 at the noun and the block inputs of 5 and 8
    H = {l: [] for l in (3, 5, 6, 7, 8)}; XIN = {5: [], 8: []}
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in XIN: XIN[l].append(x[idx, pn].float().cpu())
                if l in H: H[l].append(dod_units.hidden(model, block.mlp, xin)[idx, pn].float().cpu())
                x = x + block.mlp(xin)
                if l == 8: break
            forwards += 1
    H = {l: torch.cat(v) for l, v in H.items()}; XIN = {l: torch.cat(v) for l, v in XIN.items()}
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    top = {}
    for l, (dst, unit) in TARGETS.items():
        scale = 1.0
        for j in range(l + 1, dst + 1): scale *= lam[j][0]
        Dw = blocks[l].mlp.Down.weight.detach().float().cpu(); Lr = blocks[dst].mlp.Left.weight.detach().float()[unit].cpu(); Rr = blocks[dst].mlp.Right.weight.detach().float()[unit].cpu(); rms = XIN[dst].pow(2).mean(1).sqrt()
        score = torch.zeros(4608)
        for vec in (Lr, Rr):
            term = H[l] * (scale * (Dw.T @ vec)).unsqueeze(0) / rms.unsqueeze(1); score += (term[plural] - term[sing]).sum(0).abs()
        top[l] = tuple(torch.argsort(score, descending=True)[:K].tolist())
    # MLP 8: v168's reader-direction census (9.6's they - he reader on MLP 8's write)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components()); fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    rD = r @ blocks[8].mlp.Down.weight.detach().float().cpu(); T8 = H[8] * rD.unsqueeze(0); top[8] = tuple(torch.argsort((T8[plural] - T8[sing]).sum(0).abs(), descending=True)[:K].tolist())
    EDIT3 = {3: top[3]}; mlp7 = blocks[7].mlp
    def pass_factors(edits):
        marg, Lf, Rf = [], [], []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r_) for r_ in chunk])
            with torch.no_grad():
                x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l == 7: Lf.append(block.mlp.Left(xin)[idx, pn, 1779].float().cpu()); Rf.append(block.mlp.Right(xin)[idx, pn, 1779].float().cpu())
                    if edits and l in edits:
                        h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
                    else: x = x + block.mlp(xin)
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
                marg += [float(logits[i, r_.final, L._single(" they")] - logits[i, r_.final, L._single(" he")]) for i, r_ in enumerate(chunk)]
        return marg, torch.cat(Lf), torch.cat(Rf)
    def oriented(m): return sum((m[i] if row.present else -m[i]) for i, row in enumerate(rows)) / len(rows)
    m0, L0, R0 = pass_factors(None); forwards += 3; m1, L1, R1 = pass_factors(EDIT3); forwards += 3
    base = oriented(m0); h0, h1 = L0 * R0, L1 * R1
    dh = (h1[plural] - h1[sing]) - (h0[plural] - h0[sing])
    tL = ((L1 - L0) * R0)[plural] - ((L1 - L0) * R0)[sing]; tR = (L1 * (R1 - R0))[plural] - (L1 * (R1 - R0))[sing]
    fclos = float(((tL + tR - dh).abs() / dh.abs().clamp_min(1e-6)).max()); tot = float(dh.sum()); shareL, shareR = float(tL.sum()) / tot, float(tR.sum()) / tot
    cL0, cL1 = float((L0[plural] - L0[sing]).sum()), float((L1[plural] - L1[sing]).sum()); cR0, cR1 = float((R0[plural] - R0[sing]).sum()), float((R1[plural] - R1[sing]).sum())
    relL, relR = (cL1 - cL0) / abs(cL0), (cR1 - cR0) / abs(cR0); rise_factor = "L" if abs(shareL) >= abs(shareR) else "R"
    report = {"native_margin": base, "replay_gap": abs(base - NATIVE_M), "contrast_change_1779": tot, "share_L_term": shareL, "share_R_term": shareR, "L_contrast_native_edit": [cL0, cL1], "R_contrast_native_edit": [cR0, cR1], "rel_change_L": relL, "rel_change_R": relR, "rise_factor": rise_factor, "factor_closure": fclos}
    print(json.dumps(report, indent=1))
    rise_rel, other_rel = (relL, relR) if rise_factor == "L" else (relR, relL)
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_factor_closure": fclos <= FACTOR_TOL, "pred_c_rise_is_on_one_factor": max(abs(shareL), abs(shareR)) >= ONE_MIN, "pred_d_rise_factor_reads_the_missing_state": abs(rise_rel) >= MOVE_MIN, "pred_e_other_factor_stable": abs(other_rel) < STABLE}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "restorer_factor_split_result_v379", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
