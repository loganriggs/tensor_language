#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_1779_rises_under_mlp3_cut pred_c_829_cut_smaller_than_1036_cut pred_d_1036_cut_tracks_head45_cut pred_e_restorer_share
"""Does the compensation stage restore what the MLP-3 population removes? (v377). v376: zeroing MLP 3's top ten cuts head 4.5's number write by 27% but the
margin by only 4.7%. Section 4.9 named MLP-7 unit 1779 as a relay-and-restorer that compensates upstream number edits through a class-dependent detector
gradient. Under the MLP-3-ten edit on the v76 rows, read at the noun the plural - singular activation contrast of 1036 (MLP 5), 1779 (MLP 7) and 829 (MLP 8),
native vs edited, and the contrast of the whole MLP-7 write projected on 829's factor rows (the stage-level restoration).
PREDICTIONS (scored as written; failures preserved; priors from section 4.9 / v376)
    pred_a_baseline_replays          the native margin replays 2.048 within 1e-3
    pred_b_1779_rises_under_mlp3_cut  1779's plural - singular contrast at the noun grows in magnitude under the edit (restorer behaviour)
    pred_c_829_cut_smaller_than_1036_cut  829's relative contrast cut is smaller than 1036's (the cut shrinks between MLP 5 and MLP 8)
    pred_d_1036_cut_tracks_head45_cut  1036's relative cut is within 0.15 of the 0.27 cut of head 4.5's write (both read the MLP-3 state)
    pred_e_restorer_share            MLP 7's write contrast onto 829's L factor is cut by less than 0.10 relative (MLP 7 restores its feed to 829). Prior: unsure.
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
OUT = ROOT / "circuits/followups/compensation_under_population_cut_v377_result.json"
CANDIDATE_ID = "pronoun_number.compensation_under_population_cut_v377"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, K = 12, 372, 32, 10
NATIVE_M, M_TOL, CUT45_V376, TRACK, REST = 2.0481, 1e-3, 0.2667, 0.15, 0.10
FORWARDS_MAX = 11
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_1779_rises_under_mlp3_cut": "|contrast| grows", "pred_c_829_cut_smaller_than_1036_cut": "829 < 1036", "pred_d_1036_cut_tracks_head45_cut": "within 0.15 of 0.27", "pred_e_restorer_share": "< 0.10 relative cut"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "cut45_v376": CUT45_V376, "track": TRACK, "rest": REST}}
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
    EDIT3 = {3: top[3]}; Lr8, Rr8 = blocks[8].mlp.Left.weight.detach().float()[829].cpu(), blocks[8].mlp.Right.weight.detach().float()[829].cpu()
    def pass_with(edits):
        marg, acts, m7 = [], {1036: [], 1779: [], 829: []}, []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r_) for r_ in chunk])
            with torch.no_grad():
                x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    h = dod_units.hidden(model, block.mlp, xin)
                    if edits and l in edits: h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0
                    for u, lay in ((1036, 5), (1779, 7), (829, 8)):
                        if l == lay: acts[u].append(h[idx, pn, u].float().cpu())
                    m = block.mlp.Down(h) + block.mlp.Down_bias
                    if l == 7: m7.append(m[idx, pn].float().cpu())
                    if l == 8: rms8 = x[idx, pn].float().pow(2).mean(1).sqrt().cpu()
                    x = x + m
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
                marg += [float(logits[i, r_.final, L._single(" they")] - logits[i, r_.final, L._single(" he")]) for i, r_ in enumerate(chunk)]
        return marg, {u: torch.cat(v) for u, v in acts.items()}, torch.cat(m7)
    def contrast(v): return float((v[plural] - v[sing]).sum())
    def oriented(m): return sum((m[i] if row.present else -m[i]) for i, row in enumerate(rows)) / len(rows)
    m0, a0, w0 = pass_with(None); forwards += 3; m3, a3, w3 = pass_with(EDIT3); forwards += 3
    base = oriented(m0); cuts = {str(u): 1 - contrast(a3[u]) / contrast(a0[u]) for u in a0}
    m7_native = contrast(float(lam[8][0]) * (w0 @ Lr8)); m7_edit = contrast(float(lam[8][0]) * (w3 @ Lr8)); rest_cut = 1 - m7_edit / m7_native
    report = {"native_margin": base, "replay_gap": abs(base - NATIVE_M), "margin_change_rel": (oriented(m3) - base) / base, "relative_cut_of_contrast": cuts, "contrast_1779_native": contrast(a0[1779]), "contrast_1779_edit": contrast(a3[1779]), "mlp7_write_on_829_L_relative_cut": rest_cut}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_1779_rises_under_mlp3_cut": abs(contrast(a3[1779])) > abs(contrast(a0[1779])), "pred_c_829_cut_smaller_than_1036_cut": cuts["829"] < cuts["1036"],
                   "pred_d_1036_cut_tracks_head45_cut": abs(cuts["1036"] - CUT45_V376) <= TRACK, "pred_e_restorer_share": abs(rest_cut) < REST}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "compensation_under_population_cut_result_v377", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
