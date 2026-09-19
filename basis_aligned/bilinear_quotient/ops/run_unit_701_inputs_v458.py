#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_701_is_a_product_of_two_number_inputs pred_c_mlp8_feeds_701 pred_d_mlps_feed_701 pred_e_829_column_reads_into_701
"""What feeds unit 701 of MLP 17 (v458). v457: on the direct path to an adjacent verb, MLP 17's agreement write is half one unit, 701. A bilinear unit's
activation is h = (L x)(R x); its plural - singular change splits exactly as dL R_s + L_s dR + dL dR, and dL, dR split exactly by writer of the residual
entering block 17 (lambda chain; the input norm's scale shared per row). On the 43 adjacent pairs: which side carries the change, who writes it, and does
MLP 8's unit 829 (the number hub) write into 701's input directions?
PREDICTIONS (scored as written; failures preserved; priors from v379 / v406 / v455)
    pred_a_closure                            the three product terms sum to dh, and the writer terms to dL and dR, within relative 1e-3 per pair
    pred_b_701_is_a_product_of_two_number_inputs  both sides carry number: |dL| >= 0.30 |L_s| and |dR| >= 0.30 |R_s| pooled (a product of two number-bearing inputs, as 3465 / 493 were). Prior: unsure
    pred_c_mlp8_feeds_701                     mlp8 is among the three largest |writer terms| of dL or of dR
    pred_d_mlps_feed_701                      MLP writers carry >= 0.60 of dL and of dR
    pred_e_829_column_reads_into_701          |cos(Down_8[:, 829], L_701)| >= 0.15 or |cos(Down_8[:, 829], R_701)| >= 0.15. Prior: unsure
PRICE (registered maximum): 2 text batches = 2 forwards; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/unit_701_inputs_v458_result.json"
CANDIDATE_ID = "chain.unit_701_inputs_v458"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, SIDE_MIN, MLP_MIN, COL_MIN, UNIT = 1e-3, 0.30, 0.60, 0.15, 701
READERS = ((17, 0),)   # the residual entering block 17's MLP
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-3", "pred_b_701_is_a_product_of_two_number_inputs": ">= 0.30 x 2", "pred_c_mlp8_feeds_701": "top 3", "pred_d_mlps_feed_701": ">= 0.60 x 2", "pred_e_829_column_reads_into_701": "|cos| >= 0.15"}


def natural_pairs():
    E = L.ENCODING
    def partner(tok):
        t = E.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E.encode(c)) == 1: return E.encode(c)[0]
        return None
    recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]; items = []
    for r_ in recs:
        c = r_["cue_offset"]; alt = partner(r_["ids"][c])
        if alt is None: continue
        swapped = list(r_["ids"]); swapped[c] = alt
        plural, singular = (r_["ids"], swapped) if r_["cue"] == "plural" else (swapped, r_["ids"])
        if r_["second_offset"] - c != 1: continue                     # adjacent verbs: the answer position is the noun
        items.append((plural[:c + 1], singular[:c + 1], c))
    return items


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, noun_of(rows[i])) for i, r_ in enumerate(rows) if r_.present]
    text_items = natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "side_min": SIDE_MIN, "mlp_min": MLP_MIN, "col_min": COL_MIN, "unit": UNIT}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    # per reader: the linear map from the normalised noun residual to w: m_h = c_v^T (u O_h) restricted to the head's slice
    u = (WU[L._single(" are")] - WU[L._single(" is")] + WU[L._single(" were")] - WU[L._single(" was")]).cuda(); maps = {READERS[0]: u}
    layers = {l for l, _ in READERS}; forwards = 0
    def capture(seqs, positions, batch):
        """per-writer parts of `live` entering each reader block at the noun position, plus live itself"""
        out = {r: {"parts": [], "live": []} for r in READERS}
        with torch.no_grad():
            for s0 in range(0, len(seqs), batch):
                tokens = torch.tensor([list(q) + [0] * (max(len(t) for t in seqs[s0:s0 + batch]) - len(q)) for q in seqs[s0:s0 + batch]], device="cuda"); pos = torch.tensor(positions[s0:s0 + batch]); idx = torch.arange(tokens.shape[0])   # padded: truncated rows vary in length; pos = torch.tensor(positions[s0:s0 + batch]); idx = torch.arange(tokens.shape[0])
                x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {"embedding": x.clone()}
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0
                    for k in parts: parts[k] = block.lambdas[0] * parts[k]
                    parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                    attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; parts[f"attn{l}"] = attention.clone()
                    if l == 17: break
                    m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = m.clone()
                r = READERS[0]; out[r]["parts"].append({k: v[idx, pos].float().cpu() for k, v in parts.items()}); out[r]["live"].append(x[idx, pos].float().cpu())   # block 17's MLP input at the noun
        res = {}
        for r in READERS:
            keys = out[r]["parts"][0].keys(); res[r] = ({k: torch.cat([p[k] for p in out[r]["parts"]]) for k in keys}, torch.cat(out[r]["live"]))
        return res, -(-len(seqs) // batch)
    def fold(items, batch):
        seqs = [p for p, _, _ in items] + [s_ for _, s_, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
        res, fw_ = capture(seqs, pos, batch); parts, live = res[READERS[0]]
        mlp17 = blocks[17].mlp; Lw = mlp17.Left.weight.detach().float().cpu()[UNIT]; Rw = mlp17.Right.weight.detach().float().cpu()[UNIT]
        rms = live.pow(2).mean(1, keepdim=True).sqrt(); xn = live / rms
        Lv, Rv = xn @ Lw, xn @ Rw; h = (torch.nn.functional.silu(Lv) * Rv) if model.config.gated else (Lv * Rv)
        if model.config.gated: raise SystemExit("gated MLP: the product split below assumes h = L R")
        dL, dR, Ls, Rs = Lv[:n] - Lv[n:], Rv[:n] - Rv[n:], Lv[n:], Rv[n:]; dh = h[:n] - h[n:]
        recon = dL * Rs + Ls * dR + dL * dR; closure = float(((recon - dh).abs() / dh.abs().clamp_min(1e-6)).max())
        wL = {k: ((v[:n] / rms[:n]) @ Lw - (v[n:] / rms[n:]) @ Lw) for k, v in parts.items()}; wR = {k: ((v[:n] / rms[:n]) @ Rw - (v[n:] / rms[n:]) @ Rw) for k, v in parts.items()}
        closure = max(closure, float(((sum(wL.values()) - dL).abs() / dL.abs().clamp_min(1e-6)).max()), float(((sum(wR.values()) - dR).abs() / dR.abs().clamp_min(1e-6)).max()))
        shares = lambda w, d: {k: float(v.sum() / d.sum()) for k, v in w.items()}
        sL, sR = shares(wL, dL), shares(wR, dR); topL, topR = sorted(sL, key=lambda k: -abs(sL[k]))[:5], sorted(sR, key=lambda k: -abs(sR[k]))[:5]
        Dw8 = blocks[8].mlp.Down.weight.detach().float().cpu()[:, 829]; cosL, cosR = float(abs(Dw8 @ Lw) / (Dw8.norm() * Lw.norm())), float(abs(Dw8 @ Rw) / (Dw8.norm() * Rw.norm()))
        rep = {"pairs": n, "closure": closure, "dh_mean": float(dh.mean()), "terms": {"dL_R": float((dL * Rs).sum() / dh.sum()), "L_dR": float((Ls * dR).sum() / dh.sum()), "dL_dR": float((dL * dR).sum() / dh.sum())},
               "rel_change": {"L": float(dL.abs().mean() / Ls.abs().mean()), "R": float(dR.abs().mean() / Rs.abs().mean())}, "L_writers_top5": {k: sL[k] for k in topL}, "R_writers_top5": {k: sR[k] for k in topR},
               "L_mlp_share": sum(v for k, v in sL.items() if k.startswith("mlp")), "R_mlp_share": sum(v for k, v in sR.items() if k.startswith("mlp")), "cos_829_column_with_L": cosL, "cos_829_column_with_R": cosR}
        return rep, closure, fw_
    rep, closure, forwards = fold(text_items, 64); report = {"closure_max": closure, "unit": UNIT, "text": rep}
    print(json.dumps(rep, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_701_is_a_product_of_two_number_inputs": rep["rel_change"]["L"] >= SIDE_MIN and rep["rel_change"]["R"] >= SIDE_MIN,
                   "pred_c_mlp8_feeds_701": "mlp8" in list(rep["L_writers_top5"])[:3] or "mlp8" in list(rep["R_writers_top5"])[:3], "pred_d_mlps_feed_701": rep["L_mlp_share"] >= MLP_MIN and rep["R_mlp_share"] >= MLP_MIN,
                   "pred_e_829_column_reads_into_701": max(rep["cos_829_column_with_L"], rep["cos_829_column_with_R"]) >= COL_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "unit_701_inputs_result_v458", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
