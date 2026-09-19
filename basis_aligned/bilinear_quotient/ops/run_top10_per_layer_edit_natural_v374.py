#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_margin_positive pred_b_top50_lowers_margin_on_text pred_c_beats_random_sets_on_text pred_d_exceeds_six_on_text pred_e_within_2x_of_panel
"""The top ten units per layer, edited on natural text (v374). v372 (panel): the 50 units (top ten per MLP 3 / 5 / 6 / 7 / 8, census on the panel rows) cost 34%
of the margin; v371: the six leaders cost 8.8% on the 128 natural rows. Here the same 50 (census recomputed on the panel rows in-script, as in v372) zeroed on
the natural rows; the they - he logit difference at the final token oriented by the label; 12 random 50-sets as null.
PREDICTIONS (scored as written; failures preserved; priors from v371 / v372)
    pred_a_native_margin_positive     the native oriented margin on the natural rows is positive
    pred_b_top50_lowers_margin_on_text  zeroing the 50 lowers the oriented margin by >= 0.10 of its native value
    pred_c_beats_random_sets_on_text  |change| exceeds 3x the largest |change| among the 12 random 50-sets
    pred_d_exceeds_six_on_text        |change| > the six leaders' 0.210 (v371)
    pred_e_within_2x_of_panel         the relative change on text is within a factor 2 of the panel's -0.34 (-0.17 to -0.68). Prior: unsure.
PRICE (registered maximum): 3 census batches + 2 natural batches x (1 native + 1 edit + 12 null) = 31 forwards; 0 backwards; 0 fits. Bar <= 34.
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
OUT = ROOT / "circuits/followups/top10_per_layer_edit_natural_v374_result.json"
CANDIDATE_ID = "pronoun_number.top10_per_layer_edit_natural_v374"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, K = 12, 372, 32, 10
MOVE_MIN, SIX_TEXT, NULL_FACTOR, PANEL_REL = 0.10, 0.2103, 3.0, -0.3423
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 34
PREDICTIONS = {"pred_a_native_margin_positive": "> 0", "pred_b_top50_lowers_margin_on_text": "<= -0.10 of native", "pred_c_beats_random_sets_on_text": "> 3x null", "pred_d_exceeds_six_on_text": "> 0.210", "pred_e_within_2x_of_panel": "-0.17 to -0.68"}


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


def margins_tokens(backend, tokens, finals, edits, readers):
    torch, F, model = backend.torch, backend.F, backend.model
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if edits and l in edits:
                h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0; x = x + block.mlp.Down(h) + block.mlp.Down_bias
            else: x = x + block.mlp(xin)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
    return [{name: float(logits[i, finals[i], a] - logits[i, finals[i], b]) for name, (a, b) in readers.items()} for i in range(tokens.shape[0])]


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "targets": {str(k): list(v) for k, v in TARGETS.items()}, "k": K, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"move_min": MOVE_MIN, "six_text": SIX_TEXT, "null_factor": NULL_FACTOR, "panel_rel": PANEL_REL}, "natural": [p_.name for p_ in NATURAL]}
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
    EDIT = {l: top[l] for l in top}
    recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]; nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); finals = [len(r_["ids"]) - 1 for r_ in recs]; sign = [1.0 if r_["label"] == "they" else -1.0 for r_ in recs]
    rng = random.Random(SEED); null_sets = [{l: tuple(rng.sample([j for j in range(4608) if j not in EDIT[l]], K)) for l in EDIT} for _ in range(N_NULL)]
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("top50", EDIT)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, edits in conditions:
        out = []
        for start in range(0, len(recs), 64):
            out += margins_tokens(backend, nat[start:start + 64], finals[start:start + 64], edits, readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum(m["they_he"] * sg for m, sg in zip(margins[name], sign)) / len(recs)
    base = oriented("baseline"); d = oriented("top50") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_top50": d, "change_rel": d / base, "null_max_abs": max(abs(v) for v in nulls), "top10_per_layer": {str(l): list(v) for l, v in EDIT.items()}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_native_margin_positive": base > 0, "pred_b_top50_lowers_margin_on_text": d <= -MOVE_MIN * abs(base), "pred_c_beats_random_sets_on_text": abs(d) > NULL_FACTOR * report["null_max_abs"], "pred_d_exceeds_six_on_text": abs(d) > SIX_TEXT, "pred_e_within_2x_of_panel": PANEL_REL * 2 <= d / base <= PANEL_REL / 2}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "top10_per_layer_edit_natural_result_v374", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
