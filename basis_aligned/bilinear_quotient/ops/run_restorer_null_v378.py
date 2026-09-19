#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_named_rise_replays pred_c_rise_beats_random_tens pred_d_random_tens_leave_829_intact pred_e_829_protection_specific
"""Null for the restorer (v378). v377: zeroing MLP 3's top ten makes MLP-7 unit 1779's plural - singular contrast grow 17% and leaves 829 with 91% of its
contrast. Is the rise a response to losing the number population, or does 1779 rise under any MLP-3 ten-unit removal? 12 seeded random ten-sets of MLP 3
(disjoint from the named ten) vs the named ten; readouts 1779's and 829's relative contrast changes at the noun and the margin.
PREDICTIONS (scored as written; failures preserved; priors from v377)
    pred_a_baseline_replays        the native margin replays 2.048 within 1e-3
    pred_b_named_rise_replays      1779's relative rise under the named ten replays v377's +0.17 within 0.03
    pred_c_rise_beats_random_tens  the named rise exceeds 3x the largest |relative change of 1779| among the 12 random tens
    pred_d_random_tens_leave_829_intact  every random ten changes 829's contrast by < 0.03 relative
    pred_e_829_protection_specific the named ten's 829 cut (0.09) is at least 3x any random ten's |829 change| (the detector responds to the population, damped by the restorer)
PRICE (registered maximum): 3 census batches + 3 row batches x (1 native + 1 named + 12 null) = 45 forwards; 0 backwards; 0 fits. Bar <= 48.
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
OUT = ROOT / "circuits/followups/restorer_null_v378_result.json"
CANDIDATE_ID = "pronoun_number.restorer_null_v378"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, K = 12, 372, 32, 10
NATIVE_M, M_TOL, V377_RISE, RISE_TOL, NULL_FACTOR, INTACT = 2.0481, 1e-3, 0.171, 0.03, 3.0, 0.03
FORWARDS_MAX = 48
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_named_rise_replays": "0.17 +- 0.03", "pred_c_rise_beats_random_tens": "> 3x null", "pred_d_random_tens_leave_829_intact": "< 0.03 x 12", "pred_e_829_protection_specific": ">= 3x null"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "v377_rise": V377_RISE, "rise_tol": RISE_TOL, "null_factor": NULL_FACTOR, "intact": INTACT}}
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
    rng = random.Random(378); pool = [j for j in range(4608) if j not in top[3]]; nulls = []
    for _ in range(12):
        rs = tuple(rng.sample(pool, 10)); mn, an, wn = pass_with({3: rs}); forwards += 3
        nulls.append({"rel_1779": contrast(an[1779]) / contrast(a0[1779]) - 1, "rel_829": contrast(an[829]) / contrast(a0[829]) - 1})
    base = oriented(m0); rise = contrast(a3[1779]) / contrast(a0[1779]) - 1; cut829 = 1 - contrast(a3[829]) / contrast(a0[829])
    report = {"native_margin": base, "replay_gap": abs(base - NATIVE_M), "named_rise_1779": rise, "named_cut_829": cut829, "null_1779_max_abs": max(abs(n["rel_1779"]) for n in nulls), "null_829_max_abs": max(abs(n["rel_829"]) for n in nulls), "nulls": nulls}
    print(json.dumps({k: v for k, v in report.items() if k != "nulls"}, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_named_rise_replays": abs(rise - V377_RISE) <= RISE_TOL, "pred_c_rise_beats_random_tens": rise > NULL_FACTOR * report["null_1779_max_abs"],
                   "pred_d_random_tens_leave_829_intact": report["null_829_max_abs"] < INTACT, "pred_e_829_protection_specific": cut829 >= NULL_FACTOR * report["null_829_max_abs"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "restorer_null_result_v378", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
