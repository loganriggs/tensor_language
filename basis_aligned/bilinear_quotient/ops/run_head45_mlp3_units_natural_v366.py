#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_3465_in_top3_on_text pred_c_top10_share_on_text pred_d_mlp3_share_of_head_on_text pred_e_114_in_top10_on_text
"""MLP-3 units writing head 4.5's copied state at natural cue nouns (v366). v361 / v363: on frames and panel rows MLP 3's half of the state head 4.5 copies is
led by 3465, with 114 and 493 near the top and a long tail (top-10 ~ 21%). Same exact per-unit fold at the cue noun of the 128 natural sentences (plural-cue
minus singular-cue means on 9.6's reader).
PREDICTIONS (scored as written; failures preserved; priors from v361 / v363 / v365)
    pred_a_unit_closure           the per-unit terms sum to MLP 3's own-key term within relative 1e-3 on every row
    pred_b_3465_in_top3_on_text   3465 is among the top three MLP-3 units by |contrast| on text
    pred_c_top10_share_on_text    the top ten carry <= 0.40 of the summed |contrast| (long tail, as on the panel)
    pred_d_mlp3_share_of_head_on_text  MLP 3's own-key term is 0.30-0.70 of head 4.5's write contrast (v365: 0.53)
    pred_e_114_in_top10_on_text   114 is in the top ten. Prior: unsure -- on text the verb-site unit may not co-fire at the noun.
PRICE (registered maximum): 2 natural batches (blocks 0-4) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/head45_mlp3_units_natural_v366_result.json"
CANDIDATE_ID = "pronoun_number.head45_mlp3_units_natural_v366"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, TOP10_MAX, LO, HI, HEAD, BATCH = 1e-3, 0.40, 0.30, 0.70, 5, 64
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_3465_in_top3_on_text": "top 3", "pred_c_top10_share_on_text": "<= 0.40", "pred_d_mlp3_share_of_head_on_text": "0.30-0.70", "pred_e_114_in_top10_on_text": "top 10"}


def main() -> None:
    rows, he, she, agents, objects = g.build(); recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "natural_rows": len(recs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_max": TOP10_MAX, "lo": LO, "hi": HI}, "frame": "The X", "head": HEAD}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    the = L._single("The"); Wo_h = Wp[:, HEAD * hd:(HEAD + 1) * hd]; Wv_h = attn.c_v.weight.detach().float()[HEAD * hd:(HEAD + 1) * hd]; lamb = float(attn.lamb) if hasattr(attn, "lamb") else None
    Wv0_h = blocks[0].attn.c_v.weight.detach().float()[HEAD * hd:(HEAD + 1) * hd]; store = {}
    def wrap(orig):
        def f(q, k, v, q2, k2):
            B, T, H, Dh = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dh) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dh)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0); store["pat"] = pat[:, HEAD].float(); store["v"] = v[:, :, HEAD].float(); return orig(q, k, v, q2, k2)
        return f
    nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); cue = [int(r_["cue_offset"]) for r_ in recs]
    mlp3 = blocks[3].mlp; Dw3 = mlp3.Down.weight.detach().float(); b3 = mlp3.Down_bias.detach().float(); unit_terms, head_writes, mlp3_terms = [], [], []; closure = 0.0
    with torch.no_grad():
        for start in range(0, len(recs), BATCH):
            ids = nat[start:start + BATCH]; idx = torch.arange(ids.shape[0], device=ids.device); pn = torch.tensor(cue[start:start + BATCH], device=ids.device)
            x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None; h3 = None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    orig = block.attn.squared_attention; block.attn.squared_attention = wrap(orig)
                    captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: hook.remove(); block.attn.squared_attention = orig
                    y = captured["y"][idx, pn].float(); head_write = y[:, HEAD * hd:(HEAD + 1) * hd] @ Wo_h.T
                    pat = store["pat"]; pXX = pat[idx, pn, pn]; rms = live[idx, pn].float().pow(2).mean(-1, keepdim=True).sqrt(); lam = float(block.attn.lamb); scale = float(block.lambdas[0])
                    hn = h3[idx, pn]; per_unit = hn.unsqueeze(2) * Dw3.T.unsqueeze(0)
                    per_unit_r = ((pXX.unsqueeze(1) * (1 - lam) * scale / rms) * ((per_unit @ Wv_h.T) @ Wo_h.T @ r.to(per_unit.device)))
                    m3 = scale * (hn @ Dw3.T + b3); term_total = ((pXX.unsqueeze(1) * ((1 - lam) * (m3 / rms) @ Wv_h.T)) @ Wo_h.T) @ r.to(m3.device)
                    recon = per_unit_r.sum(1) + (pXX * (1 - lam) * scale / rms.squeeze(1)) * float(((b3 @ Wv_h.T) @ Wo_h.T) @ r.to(m3.device))
                    closure = max(closure, float(((recon - term_total).abs() / term_total.abs().clamp_min(1e-6)).max()))
                    unit_terms.append(per_unit_r.cpu()); head_writes.append((head_write @ r.to(head_write.device)).cpu()); mlp3_terms.append(term_total.cpu()); break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; xin = F.rms_norm(x, (D,))
                if l == 3: h3 = dod_units.hidden(model, block.mlp, xin).float()
                x = x + block.mlp(xin)
            forwards += 1
    U, HW, M3 = torch.cat(unit_terms), torch.cat(head_writes), torch.cat(mlp3_terms)
    plural = [i for i, r_ in enumerate(recs) if r_["cue"] == "plural"]; sing = [i for i, r_ in enumerate(recs) if r_["cue"] != "plural"]
    pooled = U[plural].mean(0) - U[sing].mean(0); order = torch.argsort(pooled.abs(), descending=True); total = float(pooled.abs().sum())
    rank = {str(u): int((pooled.abs() > pooled.abs()[u]).sum()) + 1 for u in (3465, 114, 493)}; top10 = float(pooled.abs()[order[:10]].sum() / total)
    head_c = float(HW[plural].mean() - HW[sing].mean()); m3_c = float(M3[plural].mean() - M3[sing].mean())
    report = {"closure_max": closure, "top12": [(int(j), float(pooled[j])) for j in order[:12]], "ranks": rank, "top10_share": top10, "head_write_contrast": head_c, "mlp3_term_contrast": m3_c, "mlp3_share_of_head": m3_c / head_c}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_3465_in_top3_on_text": rank["3465"] <= 3, "pred_c_top10_share_on_text": top10 <= TOP10_MAX, "pred_d_mlp3_share_of_head_on_text": LO <= report["mlp3_share_of_head"] <= HI, "pred_e_114_in_top10_on_text": rank["114"] <= 10}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_mlp3_units_natural_result_v366", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
