#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_monotone_in_k pred_c_every_k_beats_null pred_d_k10_replays_v372 pred_e_diminishing_returns
"""Dose-response of the per-layer populations (v373). v372: the top ten units per MLP (3, 5, 6, 7, 8) cost 34% of the margin, the six leaders 14%. Same census,
k = 1 / 5 / 10 / 20 units per layer zeroed at every position on the v76 rows; 3 random k-sets per k (k per layer) as null; oriented they - he margin.
PREDICTIONS (scored as written; failures preserved; priors from v370 / v372)
    pred_a_baseline_replays   the unedited margin replays 2.048 within 1e-3
    pred_b_monotone_in_k      |change| increases monotonically over k = 1, 5, 10, 20
    pred_c_every_k_beats_null |change(k)| exceeds 3x the largest |change| among the random k-sets, every k
    pred_d_k10_replays_v372   |change(10)| replays v372's 0.701 within 0.02
    pred_e_diminishing_returns  change(20) - change(10) < change(10) - change(5) in magnitude (the head carries more per unit than the tail). Prior: unsure.
PRICE (registered maximum): 3 census batches + 3 row batches x (1 baseline + 4 edits + 12 null) = 54 forwards; 0 backwards; 0 fits. Bar <= 58.
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
OUT = ROOT / "circuits/followups/per_layer_dose_response_v373_result.json"
CANDIDATE_ID = "pronoun_number.per_layer_dose_response_v373"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, KS = 3, 373, 32, (1, 5, 10, 20)
NATIVE_M, M_TOL, V372, NULL_FACTOR = 2.0481, 1e-3, 0.7012, 3.0
FORWARDS_MAX = 58
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_monotone_in_k": "monotone", "pred_c_every_k_beats_null": "> 3x null x 4", "pred_d_k10_replays_v372": "0.701 +- 0.02", "pred_e_diminishing_returns": "concave"}


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
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "targets": {str(k): list(v) for k, v in TARGETS.items()}, "ks": list(KS), "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "v372": V372, "null_factor": NULL_FACTOR}}
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
        top[l] = tuple(torch.argsort(score, descending=True)[:max(KS)].tolist())
    # MLP 8: v168's reader-direction census (9.6's they - he reader on MLP 8's write)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components()); fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    rD = r @ blocks[8].mlp.Down.weight.detach().float().cpu(); T8 = H[8] * rD.unsqueeze(0); top[8] = tuple(torch.argsort((T8[plural] - T8[sing]).sum(0).abs(), descending=True)[:max(KS)].tolist())
    rng = random.Random(SEED); readers = {"they_he": (L._single(" they"), L._single(" he"))}
    def run_margin(edits):
        out = []
        for start in range(0, len(rows), BATCH): out += margins_multi(backend, fw, rows[start:start + BATCH], edits, readers)
        return sum((out[i]["they_he"] if row.present else -out[i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
    base = run_margin(None); forwards += 3; per = {}
    for k in KS:
        EDIT = {l: top[l][:k] for l in top}; d = run_margin(EDIT) - base; forwards += 3; nulls = []
        for _ in range(N_NULL):
            rs = {l: tuple(rng.sample([j for j in range(4608) if j not in top[l]], k)) for l in top}; nulls.append(run_margin(rs) - base); forwards += 3
        per[str(k)] = {"change": d, "change_rel": d / base, "null_max_abs": max(abs(v) for v in nulls)}
    ch = [abs(per[str(k)]["change"]) for k in KS]
    report = {"native_margin": base, "replay_gap": abs(base - NATIVE_M), "per_k": per, "top_units": {str(l): list(v[:5]) for l, v in top.items()}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_monotone_in_k": all(ch[i + 1] > ch[i] for i in range(len(ch) - 1)), "pred_c_every_k_beats_null": all(abs(per[str(k)]["change"]) > NULL_FACTOR * per[str(k)]["null_max_abs"] for k in KS),
                   "pred_d_k10_replays_v372": abs(ch[2] - V372) <= 0.02, "pred_e_diminishing_returns": (ch[3] - ch[2]) < (ch[2] - ch[1])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "per_layer_dose_response_result_v373", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
