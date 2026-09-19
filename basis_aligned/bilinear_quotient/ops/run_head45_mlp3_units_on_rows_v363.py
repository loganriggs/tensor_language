#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_trio_leads_on_rows pred_c_trio_share_of_mlp3_term pred_d_3465_first_on_rows pred_e_mlp3_term_share_of_head_write
"""The MLP-3 trio's share of head 4.5's noun write on the pronoun rows (v363). v361 (frames): MLP 3's contribution to the state 4.5 copies is led by 3465, 114,
493. v362 (rows, edit): zeroing them costs 3.6% of the margin. Here the same exact per-unit fold on the v76 rows at the NOUN: MLP 3's term in 4.5's own-key
value (all keys' own term; the head's other keys reported as the rest), per unit, plural - singular over aligned pairs on 9.6's reader; the trio's share of
MLP 3's term, MLP 3's term's share of the head's whole write contrast.
PREDICTIONS (scored as written; failures preserved; priors from v360 / v361)
    pred_a_unit_closure               the per-unit terms sum to MLP 3's own-key term within relative 1e-3 on every row
    pred_b_trio_leads_on_rows         3465, 114 and 493 are the top three MLP-3 units by |pooled contrast| on the rows
    pred_c_trio_share_of_mlp3_term    the trio carries >= 0.30 of the summed |contrast| over MLP 3's units
    pred_d_3465_first_on_rows         3465 ranks first
    pred_e_mlp3_term_share_of_head_write  MLP 3's own-key term carries between 0.30 and 0.70 of head 4.5's total write contrast at the noun (v360 gave 0.52 on frames)
PRICE (registered maximum): 3 row batches x 1 forward (blocks 0-4) = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
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
OUT = ROOT / "circuits/followups/head45_mlp3_units_on_rows_v363_result.json"
CANDIDATE_ID = "pronoun_number.head45_mlp3_units_on_rows_v363"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, TRIO_MIN, LO, HI, HEAD, BATCH = 1e-3, 0.30, 0.30, 0.70, 5, 32
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_trio_leads_on_rows": "top 3", "pred_c_trio_share_of_mlp3_term": ">= 0.30", "pred_d_3465_first_on_rows": "rank 1", "pred_e_mlp3_term_share_of_head_write": "0.30-0.70"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "trio_min": TRIO_MIN, "lo": LO, "hi": HI}, "frame": "The X", "head": HEAD}
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
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    mlp3 = blocks[3].mlp; Dw3 = mlp3.Down.weight.detach().float(); b3 = mlp3.Down_bias.detach().float(); unit_terms, head_writes, mlp3_terms = [], [], []; closure = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; ids = fw._tokens(chunk); idx = torch.arange(len(chunk), device=ids.device); pn = torch.tensor([noun_of(r_) for r_ in chunk], device=ids.device)
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
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    pooled = (U[plural] - U[sing]).sum(0); order = torch.argsort(pooled.abs(), descending=True); total = float(pooled.abs().sum())
    rank = {str(u): int((pooled.abs() > pooled.abs()[u]).sum()) + 1 for u in (3465, 114, 493)}; trio_share = float(sum(pooled.abs()[u] for u in (3465, 114, 493)) / total)
    head_c = float((HW[plural] - HW[sing]).sum()); m3_c = float((M3[plural] - M3[sing]).sum())
    report = {"closure_max": closure, "top12": [(int(j), float(pooled[j])) for j in order[:12]], "ranks": rank, "trio_share_of_mlp3": trio_share, "head_write_contrast": head_c, "mlp3_term_contrast": m3_c, "mlp3_share_of_head": m3_c / head_c}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_trio_leads_on_rows": set(order[:3].tolist()) == {3465, 114, 493}, "pred_c_trio_share_of_mlp3_term": trio_share >= TRIO_MIN, "pred_d_3465_first_on_rows": rank["3465"] == 1, "pred_e_mlp3_term_share_of_head_write": LO <= report["mlp3_share_of_head"] <= HI}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_mlp3_units_on_rows_result_v363", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
