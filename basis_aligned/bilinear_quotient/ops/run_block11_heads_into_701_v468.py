#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_11_3_is_most_of_block_11_into_L pred_c_11_3_is_most_of_block_11_into_R pred_d_11_3_top_three_overall pred_e_7_8_contributes
"""Which head of block 11 feeds the detector at the verb (v468). v467: at a distant verb attention block 11 is the largest writer of both of 701's inputs
(0.15 / 0.18). Blocks 7, 9, 11 and 13 split by head (each head's write through its c_proj slice, lambda chain applied) in the same exact writer split of
dL and dR at the answer position, 79 distant rows.
PREDICTIONS (scored as written; failures preserved; priors from v448 / v467)
    pred_a_closure                        the writer terms (heads listed singly for blocks 7 / 9 / 11 / 13) sum to dL and dR within relative 1e-3 per pair
    pred_b_11_3_is_most_of_block_11_into_L  head 11.3 carries >= 0.70 of block 11's term into dL
    pred_c_11_3_is_most_of_block_11_into_R  head 11.3 carries >= 0.70 of block 11's term into dR
    pred_d_11_3_top_three_overall         11.3 is among the three largest |writer terms| of dL or of dR
    pred_e_7_8_contributes                head 7.8's term into dL or dR is >= 0.03 of it (the second verb reader also feeds the detector). Prior: unsure
PRICE (registered maximum): 3 text batches = 3 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/block11_heads_into_701_v468_result.json"
CANDIDATE_ID = "chain.block11_heads_into_701_v468"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, HEAD_SHARE_MIN, H78_MIN, UNIT = 1e-3, 0.70, 0.03, 701
SPLIT_BLOCKS = (7, 9, 11, 13)
READERS = ((17, 0),)   # the residual entering block 17's MLP
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_closure": "<= 1e-3", "pred_b_11_3_is_most_of_block_11_into_L": ">= 0.70", "pred_c_11_3_is_most_of_block_11_into_R": ">= 0.70", "pred_d_11_3_top_three_overall": "top 3", "pred_e_7_8_contributes": ">= 0.03"}


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
        vs = r_["second_offset"]
        if vs - c < 2: continue                                       # distant verbs: the answer position is the verb slot
        items.append((plural[:vs], singular[:vs], vs - 1))            # position = the answer (last token)
    return items


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, noun_of(rows[i])) for i, r_ in enumerate(rows) if r_.present]
    text_items = natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "head_share_min": HEAD_SHARE_MIN, "h78_min": H78_MIN, "unit": UNIT}}
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
                    if l in SPLIT_BLOCKS:
                        captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m_, args: captured.setdefault("y", args[0]))
                        try: attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_)
                        finally: hook.remove()
                        Wp = block.attn.c_proj.weight.detach().float(); y = captured["y"].float()
                        for h in range(N_HEAD): parts[f"{l}.{h}"] = (y[:, :, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T).to(attention.dtype)
                    else:
                        attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); parts[f"attn{l}"] = attention.clone()
                    x = live + attention
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
        b11L = {k: sL[k] for k in sL if k.startswith("11.")}; b11R = {k: sR[k] for k in sR if k.startswith("11.")}
        rep = {"pairs": n, "closure": closure, "dh_mean": float(dh.mean()), "block11_L": b11L, "block11_R": b11R, "share_11_3_of_block11_L": b11L["11.3"] / sum(b11L.values()), "share_11_3_of_block11_R": b11R["11.3"] / sum(b11R.values()), "h78_L": sL.get("7.8"), "h78_R": sR.get("7.8"), "terms": {"dL_R": float((dL * Rs).sum() / dh.sum()), "L_dR": float((Ls * dR).sum() / dh.sum()), "dL_dR": float((dL * dR).sum() / dh.sum())},
               "rel_change": {"L": float(dL.abs().mean() / Ls.abs().mean()), "R": float(dR.abs().mean() / Rs.abs().mean())}, "L_writers_top5": {k: sL[k] for k in topL}, "R_writers_top5": {k: sR[k] for k in topR},
               "L_mlp_share": sum(v for k, v in sL.items() if k.startswith("mlp")), "R_mlp_share": sum(v for k, v in sR.items() if k.startswith("mlp")), "L_attn_share": sum(v for k, v in sL.items() if k.startswith("attn")), "R_attn_share": sum(v for k, v in sR.items() if k.startswith("attn")),
               "prod_abs_share": float((dL * dR).abs().sum() / dh.abs().sum()), "cos_829_column_with_L": cosL, "cos_829_column_with_R": cosR}
        return rep, closure, fw_
    rep, closure, forwards = fold(text_items, 64); report = {"closure_max": closure, "unit": UNIT, "text": rep}
    print(json.dumps(rep, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_11_3_is_most_of_block_11_into_L": rep["share_11_3_of_block11_L"] >= HEAD_SHARE_MIN, "pred_c_11_3_is_most_of_block_11_into_R": rep["share_11_3_of_block11_R"] >= HEAD_SHARE_MIN,
                   "pred_d_11_3_top_three_overall": "11.3" in list(rep["L_writers_top5"])[:3] or "11.3" in list(rep["R_writers_top5"])[:3], "pred_e_7_8_contributes": max(abs(rep["h78_L"] or 0), abs(rep["h78_R"] or 0)) >= H78_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "block11_heads_into_701_result_v468", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
