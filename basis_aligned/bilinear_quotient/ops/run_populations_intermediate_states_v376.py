#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_mlp3_population_cuts_head45_write pred_c_head45_cut_smaller_than_margin_cut pred_d_829_contrast_cut_by_lower_populations pred_e_lower_populations_alone_cut_margin
"""What the per-layer populations do to the intermediate states (v376). v372 / v373: zeroing the top ten units per MLP (3, 5, 6, 7, 8) costs 34% of the margin.
Where along the chain does that cut act? Under the edit restricted to MLP 3's ten (the stage below the copier), read head 4.5's write at the noun on 9.6's
reader (plural - singular contrast vs native); under the edit restricted to MLPs 3 / 5 / 6 / 7's forty (below MLP 8), read MLP-8 unit 829's activation
contrast at the noun and the margin. Native and edited passes on the v76 rows.
PREDICTIONS (scored as written; failures preserved; priors from v360-v363 / v368)
    pred_a_baseline_replays                 the native margin replays 2.048 within 1e-3
    pred_b_mlp3_population_cuts_head45_write  MLP 3's ten zeroed cut head 4.5's plural - singular write contrast at the noun by >= 0.30 (MLP 3 supplied 0.52 of that state)
    pred_c_head45_cut_smaller_than_margin_cut  that relative cut of 4.5's write is larger than the relative margin change from the same edit (the copier stage amplifies less than it receives). Prior: unsure.
    pred_d_829_contrast_cut_by_lower_populations  MLPs 3 / 5 / 6 / 7's forty zeroed cut 829's plural - singular activation contrast at the noun by >= 0.40
    pred_e_lower_populations_alone_cut_margin  the forty (without MLP 8's ten) cut the margin by >= 0.20 of its native value (the lower populations act mostly through MLP 8, not only directly)
PRICE (registered maximum): 3 census batches + 3 row batches x (1 native + 2 edits) = 12 forwards; 0 backwards; 0 fits. Bar <= 14.
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
OUT = ROOT / "circuits/followups/populations_intermediate_states_v376_result.json"
CANDIDATE_ID = "pronoun_number.populations_intermediate_states_v376"
TARGETS = {3: (5, 1036), 5: (8, 829), 6: (8, 829), 7: (8, 829)}; N_NULL, SEED, BATCH, K = 12, 372, 32, 10
NATIVE_M, M_TOL, CUT45, CUT829, MARGIN_CUT = 2.0481, 1e-3, 0.30, 0.40, 0.20
FORWARDS_MAX = 14
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_mlp3_population_cuts_head45_write": ">= 0.30 cut", "pred_c_head45_cut_smaller_than_margin_cut": "|4.5 cut| > |margin cut|", "pred_d_829_contrast_cut_by_lower_populations": ">= 0.40 cut", "pred_e_lower_populations_alone_cut_margin": ">= 0.20 of native"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "cut45": CUT45, "cut829": CUT829, "margin_cut": MARGIN_CUT}}
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
    EDIT3 = {3: top[3]}; EDIT_LOW = {l: top[l] for l in (3, 5, 6, 7)}
    Wp = blocks[4].attn.c_proj.weight.detach().float(); hd = model.config.n_embd // 9
    def pass_with(edits):
        """one forward per batch: returns oriented margin, head 4.5's noun write on r (per row), 829's noun activation (per row)."""
        marg, w45, a829 = [], [], []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r_) for r_ in chunk])
            with torch.no_grad():
                x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (model.config.n_embd,))
                    if l == 4:
                        captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                        try: attention, v1_ = block.attn(xin_a, v1_)
                        finally: hook.remove()
                        y = captured["y"][idx, pn].float(); w45.append(((y[:, 5 * hd:6 * hd] @ Wp[:, 5 * hd:6 * hd].T) @ r.to(y.device)).cpu())
                    else: attention, v1_ = block.attn(xin_a, v1_)
                    x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if edits and l in edits:
                        h = dod_units.hidden(model, block.mlp, xin); h[:, :, torch.tensor(list(edits[l]), device=h.device)] = 0
                        if l == 8: a829.append(h[idx, pn, 829].float().cpu())
                        x = x + block.mlp.Down(h) + block.mlp.Down_bias
                    else:
                        if l == 8: a829.append(dod_units.hidden(model, block.mlp, xin)[idx, pn, 829].float().cpu())
                        x = x + block.mlp(xin)
                logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
                marg += [float(logits[i, r_.final, L._single(" they")] - logits[i, r_.final, L._single(" he")]) for i, r_ in enumerate(chunk)]
        return marg, torch.cat(w45), torch.cat(a829)
    def contrast(v): return float((v[plural] - v[sing]).sum())
    def oriented(m): return sum((m[i] if row.present else -m[i]) for i, row in enumerate(rows)) / len(rows)
    m0, w0, a0 = pass_with(None); forwards += 3; m3, w3, a3 = pass_with(EDIT3); forwards += 3; mL, wL, aL = pass_with(EDIT_LOW); forwards += 3
    base = oriented(m0); cut45 = 1 - contrast(w3) / contrast(w0); marg3 = (oriented(m3) - base) / base; cut829 = 1 - contrast(aL) / contrast(a0); margL = (oriented(mL) - base) / base
    report = {"native_margin": base, "replay_gap": abs(base - NATIVE_M), "head45_write_contrast_native": contrast(w0), "head45_write_cut_by_mlp3_ten": cut45, "margin_change_rel_mlp3_ten": marg3, "unit829_contrast_native": contrast(a0), "unit829_cut_by_lower_forty": cut829, "margin_change_rel_lower_forty": margL,
              "top10_per_layer": {str(l): list(v) for l, v in top.items()}}
    print(json.dumps({k: v for k, v in report.items() if k != "top10_per_layer"}, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_mlp3_population_cuts_head45_write": cut45 >= CUT45, "pred_c_head45_cut_smaller_than_margin_cut": abs(cut45) > abs(marg3), "pred_d_829_contrast_cut_by_lower_populations": cut829 >= CUT829, "pred_e_lower_populations_alone_cut_margin": margL <= -MARGIN_CUT}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "populations_intermediate_states_result_v376", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
