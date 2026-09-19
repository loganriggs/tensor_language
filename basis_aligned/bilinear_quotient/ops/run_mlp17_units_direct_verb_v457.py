#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_top_unit_carries_a_tenth pred_c_top_10_carry_half pred_d_top_50_carry_most pred_e_top_unit_column_aligns_with_u_v
"""MLP 17 at unit grain on the direct path to an adjacent verb (v457). v455: on the direct path (the noun's own final residual, adjacent verbs) MLP 17 writes
a tenth of the agreement change, as much as MLP 8; v456: not along the panel axis VP0. Here MLP 17's write at the noun split exactly by unit on u_v (no lambda
chain: it is the last block), on the 43 adjacent natural pairs: is the late write a few units or a spread?
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_unit_closure               per-unit terms sum to MLP 17's term within relative 1e-3 on every row
    pred_b_top_unit_carries_a_tenth   the largest |pooled contrast| unit carries >= 0.10 of MLP 17's pooled contrast
    pred_c_top_10_carry_half          the top 10 units carry >= 0.50
    pred_d_top_50_carry_most          the top 50 units carry >= 0.80
    pred_e_top_unit_column_aligns_with_u_v  |cos(Down[:, top unit], u_v)| >= 0.30 (the unit writes the agreement direction itself)
PRICE (registered maximum): 2 text batches = 2 forwards; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp17_units_direct_verb_v457_result.json"
CANDIDATE_ID = "chain.mlp17_units_direct_verb_v457"
N_HEAD, LAYER = 9, 17
CLOSURE_TOL, TOP1_MIN, TOP10_MIN, TOP50_MIN, COL_COS_MIN = 1e-3, 0.10, 0.50, 0.80, 0.30
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_top_unit_carries_a_tenth": ">= 0.10", "pred_c_top_10_carry_half": ">= 0.50", "pred_d_top_50_carry_most": ">= 0.80", "pred_e_top_unit_column_aligns_with_u_v": "|cos| >= 0.30"}


def main() -> None:
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]]; E = L.ENCODING
    def partner(tok):
        t = E.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E.encode(c)) == 1: return E.encode(c)[0]
        return None
    items = []
    for r_ in recs:
        c = r_["cue_offset"]; alt = partner(r_["ids"][c])
        if alt is None or r_["second_offset"] - c != 1: continue          # adjacent verbs only
        sw = list(r_["ids"]); sw[c] = alt; plural, singular = (r_["ids"], sw) if r_["cue"] == "plural" else (sw, r_["ids"])
        items.append((plural[:c + 1], singular[:c + 1], c))                # truncated at the verb slot: the answer position is the noun
    plan = {"candidate_id": CANDIDATE_ID, "adjacent_pairs": len(items), "layer": LAYER, "reader": "direct (W_U)", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top1_min": TOP1_MIN, "top10_min": TOP10_MIN, "top50_min": TOP50_MIN, "col_cos_min": COL_COS_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    m = (WU[L._single(" are")] - WU[L._single(" is")] + WU[L._single(" were")] - WU[L._single(" was")]).cuda()   # the direct path: the agreement direction on the final residual
    mlp = blocks[LAYER].mlp; Dw = mlp.Down.weight.detach().float(); mD = m @ Dw                                     # (4608,)
    seqs = [p for p, _, _ in items] + [s for _, s, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
    per_row_units, per_row_mlp, forwards = [], [], 0
    with torch.no_grad():
        for s0 in range(0, len(seqs), 64):
            tokens = torch.tensor([q + [0] * (max(len(t) for t in seqs[s0:s0 + 64]) - len(q)) for q in seqs[s0:s0 + 64]], device="cuda"); pp = torch.tensor(pos[s0:s0 + 64]); idx = torch.arange(tokens.shape[0])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; lam_after = 1.0
            for l, block in enumerate(list(blocks) + [None]):
                if block is None: rms = x[idx, pp].float().pow(2).mean(1).sqrt(); break
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                if l == 18: break
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; xin = F.rms_norm(x, (D,))
                if l == LAYER:
                    Lx, Rx = mlp.Left(xin), mlp.Right(xin); h = (F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx); h8 = h[idx, pp].float(); m8 = mlp(xin)[idx, pp].float()
                x = x + block.mlp(xin)
            # the lambda chain from block 8's output to block 9's live: block 9's lambdas[0] (blocks 9 only, since MLP 8 is added after block 8's live)
            lam = 1.0
            for l_ in range(LAYER + 1, 18): lam *= float(blocks[l_].lambdas[0])          # the lambda chain from block 8 output to the final residual
            per_row_units.append((lam * mD.cpu() * h8.cpu()) / rms.cpu()[:, None]); per_row_mlp.append((lam * (m8 @ m).cpu() - lam * float(m @ mlp.Down_bias.detach().float())) / rms.cpu())   # module output minus its bias = sum of unit terms
            forwards += 1
    U = torch.cat(per_row_units); M = torch.cat(per_row_mlp)
    closure = float(((U.sum(1) - M).abs() / M.abs().clamp_min(1e-6)).max())
    total = (U[:n] - U[n:]).sum(0); contrast = float(total.sum()); order = torch.argsort(total.abs(), descending=True)
    rank = {j: int((total.abs() > abs(total[j])).sum()) + 1 for j in (829, 953, 1030, 3152, 3943)}
    prev = json.loads((ROOT / "circuits/followups/value_copy_mlp8_units_text_v408_result.json").read_text())["report"]["top_units"][:20]; overlap = len({int(j) for j, _ in prev} & set(order[:20].tolist()))
    share = lambda k: float(total[order[:k]].sum()) / contrast
    report = {"closure_max": closure, "mlp8_contrast": contrast, "shares": {str(k): share(k) for k in (10, 20, 50, 100, 200)}, "top_units": [(int(j), float(total[j])) for j in order[:30]], "ranks": rank, "top20_overlap_with_v408": overlap}
    print(json.dumps(report, indent=1))
    top = int(order[0]); col = Dw[:, top].cpu(); col_cos = float(abs(col @ m.cpu()) / (col.norm() * m.cpu().norm())); report["top_unit_column_cos_u_v"] = col_cos
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_top_unit_carries_a_tenth": share(1) >= TOP1_MIN, "pred_c_top_10_carry_half": share(10) >= TOP10_MIN, "pred_d_top_50_carry_most": share(50) >= TOP50_MIN, "pred_e_top_unit_column_aligns_with_u_v": col_cos >= COL_COS_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp17_units_direct_verb_result_v457", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
