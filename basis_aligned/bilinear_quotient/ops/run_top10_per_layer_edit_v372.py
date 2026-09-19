#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_top10_per_layer_exceeds_the_six pred_c_beats_random_sets pred_d_top10_at_least_double pred_e_top10_below_half_of_margin
"""Beyond the six: the top ten units per layer (v372). v370: the six named chain units cost 14% of the margin; each layer's census (v168 / v194 / v368) showed
spread tails. Here each of MLPs 3, 5, 6, 7, 8 contributes its top ten units by the plural - singular contrast of its per-unit write into the next named unit's
factors (MLP 3 -> 1036, MLPs 5-7 -> 829, from v368's census recomputed in-script by the same exact fold; MLP 8 by v168's reader-direction census recomputed
here), 50 units in all, zeroed at every position on the v76 rows; null: 12 random 50-sets, ten per layer; readout the oriented they - he margin.
PREDICTIONS (scored as written; failures preserved; priors from v370)
    pred_a_baseline_replays            the unedited margin replays 2.048 within 1e-3
    pred_b_top10_per_layer_exceeds_the_six  |change with the 50| > |change with the six| (0.286)
    pred_c_beats_random_sets           |change| exceeds 3x the largest |change| among the 12 random 50-sets
    pred_d_top10_at_least_double       |change with the 50| >= 2 x 0.286 (the tails matter as much again). Prior: unsure.
    pred_e_top10_below_half_of_margin  |change with the 50| < 0.5 x native (the attention readers and the rest still carry most)
PRICE (registered maximum): 3 census batches + 3 row batches x (1 baseline + 1 edit + 12 null) = 45 forwards; 0 backwards; 0 fits. Bar <= 48.
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
OUT = ROOT / "circuits/followups/top10_per_layer_edit_v372_result.json"
CANDIDATE_ID = "pronoun_number.top10_per_layer_edit_v372"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, K = 12, 372, 32, 10
NATIVE_M, M_TOL, SIX, NULL_FACTOR = 2.0481, 1e-3, 0.286, 3.0
FORWARDS_MAX = 48
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_top10_per_layer_exceeds_the_six": "> 0.286", "pred_c_beats_random_sets": "> 3x null", "pred_d_top10_at_least_double": ">= 0.572", "pred_e_top10_below_half_of_margin": "< 1.024"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "six": SIX, "null_factor": NULL_FACTOR}}
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
    EDIT = {l: top[l] for l in top}; named = {u for v in EDIT.values() for u in v}
    rng = random.Random(SEED); null_sets = [{l: tuple(rng.sample([j for j in range(4608) if j not in EDIT[l]], K)) for l in EDIT} for _ in range(N_NULL)]
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("top50", EDIT)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, edits in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            out += margins_multi(backend, fw, rows[start:start + BATCH], edits, readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum((margins[name][i]["they_he"] if row.present else -margins[name][i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
    base = oriented("baseline"); d = oriented("top50") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_top50": d, "change_rel": d / base, "null_changes": nulls, "null_max_abs": max(abs(v) for v in nulls), "replay_gap": abs(base - NATIVE_M), "top10_per_layer": {str(l): list(v) for l, v in EDIT.items()},
              "named_six_present": {str(u): any(u in v for v in EDIT.values()) for u in (3465, 493, 1036, 2483, 1779, 829)}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_top10_per_layer_exceeds_the_six": abs(d) > SIX, "pred_c_beats_random_sets": abs(d) > NULL_FACTOR * report["null_max_abs"], "pred_d_top10_at_least_double": abs(d) >= 2 * SIX, "pred_e_top10_below_half_of_margin": abs(d) < 0.5 * abs(base)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "top10_per_layer_edit_result_v372", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
