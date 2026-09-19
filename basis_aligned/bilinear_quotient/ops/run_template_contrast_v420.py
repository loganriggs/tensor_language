#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture_closure pred_b_readers_score_the_contrast_template pred_c_identity_template_small pred_d_rank_one_map_predicts_per_pair_transport pred_e_template_is_a_small_part_of_the_map
"""Template contraction on the readers' OV maps (v420; Logan's template direction, 19 Sep). The readers are value copiers (v401-v419), so their
OV map M_h = O_h V_h (D x D) is the natural object for a fixed-map template. Templates: identity I (v389: small for every head) and the CONTRAST template
T = u d^T, where u = W_U[they] - W_U[he] (output end) and d = the mean plural - singular difference of the normalised noun residual entering the head's
block over the 122 natural pairs (input end; the same quantity per pair is delta_i). Scores: cos_F(M_h, T) for every head of the block (is the reader the
head that looks most like 'copy number'?); cos_F(M_h, I); and the fixed-map test: does the rank-one template predict the head's per-pair number transport
u^T M_h delta_i from the projection d^T delta_i alone (corr over pairs)? Also the share of M_h's Frobenius energy along T (how much of the head is this).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_capture_closure                       per-writer parts sum to the residual within relative 1e-3 (instrument, as v406)
    pred_b_readers_score_the_contrast_template   each reader's cos_F(M_h, T) is the largest among its block's 9 heads. Prior: unsure
    pred_c_identity_template_small               |cos_F(M_h, I)| <= 0.05 for each reader
    pred_d_rank_one_map_predicts_per_pair_transport  corr over the 122 pairs of (u^T M_h delta_i) with (d^T delta_i) >= 0.80 for 9.6. Prior: unsure
    pred_e_template_is_a_small_part_of_the_map   the Frobenius energy share of M_h along T is <= 0.05 for each reader (the head does much more than number)
PRICE (registered maximum): 4 text batches = 4 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/template_contrast_v420_result.json"
CANDIDATE_ID = "template.contrast_readers_v420"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, IDENT_MAX, CORR_MIN, ENERGY_MAX = 1e-3, 0.05, 0.80, 0.05
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_capture_closure": "<= 1e-3", "pred_b_readers_score_the_contrast_template": "rank 1 of 9 x 3", "pred_c_identity_template_small": "<= 0.05 x 3", "pred_d_rank_one_map_predicts_per_pair_transport": "corr >= 0.80", "pred_e_template_is_a_small_part_of_the_map": "<= 0.05 x 3"}


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
        items.append((plural, singular, c))
    return items


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, noun_of(rows[i])) for i, r_ in enumerate(rows) if r_.present]
    text_items = natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "ident_max": IDENT_MAX, "corr_min": CORR_MIN, "energy_max": ENERGY_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    # per reader: the linear map from the normalised noun residual to w: m_h = c_v^T (u O_h) restricted to the head's slice
    maps = {}
    for l, h in READERS:
        attn = blocks[l].attn; Wp = attn.c_proj.weight.detach().float(); uO = u @ Wp[:, h * hd:(h + 1) * hd]
        Wv = attn.c_v.weight.detach().float()[h * hd:(h + 1) * hd]          # (hd, D): v_h = Wv x
        maps[(l, h)] = float(1 - attn.lamb) * (Wv.T @ uO)                  # (D,): w_cur = m . n(live)
    layers = {l for l, _ in READERS}; forwards = 0
    def capture(seqs, positions, batch):
        """per-writer parts of `live` entering each reader block at the noun position, plus live itself"""
        out = {r: {"parts": [], "live": []} for r in READERS}
        with torch.no_grad():
            for s0 in range(0, len(seqs), batch):
                tokens = torch.tensor(seqs[s0:s0 + batch], device="cuda"); pos = torch.tensor(positions[s0:s0 + batch]); idx = torch.arange(tokens.shape[0])
                x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {"embedding": x.clone()}
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0
                    for k in parts: parts[k] = block.lambdas[0] * parts[k]
                    parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                    if l in layers:
                        r = next(r_ for r_ in READERS if r_[0] == l)
                        out[r]["parts"].append({k: v[idx, pos].float().cpu() for k, v in parts.items()}); out[r]["live"].append(live[idx, pos].float().cpu())
                    if l == max(layers): break
                    attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; parts[f"attn{l}"] = attention.clone()
                    m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = m.clone()
        res = {}
        for r in READERS:
            keys = out[r]["parts"][0].keys(); res[r] = ({k: torch.cat([p[k] for p in out[r]["parts"]]) for k in keys}, torch.cat(out[r]["live"]))
        return res, -(-len(seqs) // batch)
    def fold(items, batch):
        seqs = [p for p, _, _ in items] + [s for _, s, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
        res, fw_ = capture(seqs, pos, batch); report = {}; closure = 0.0
        for r in READERS:
            parts, live = res[r]; m = maps[r].cpu(); rms = live.pow(2).mean(1, keepdim=True).sqrt()
            wP = (live[:n] / rms[:n]) @ m; wS = (live[n:] / rms[n:]) @ m; dw = wP - wS
            terms = {k: ((v[:n] / rms[:n]) @ m - (v[n:] / rms[n:]) @ m) for k, v in parts.items()}
            recon = sum(terms.values()); closure = max(closure, float(((recon - dw).abs() / dw.abs().clamp_min(1e-6)).max()))
            pooled = {k: float(v.sum()) for k, v in terms.items()}; total = float(dw.sum())
            shares = {k: v / total for k, v in pooled.items()}
            report[f"{r[0]}.{r[1]}"] = {"delta_w_total": total, "writer_shares": dict(sorted(shares.items(), key=lambda kv: -abs(kv[1]))), "mlp_share": sum(v for k, v in shares.items() if k.startswith("mlp")),
                                        "attn_abs_share": sum(abs(v) for k, v in shares.items() if k.startswith("attn")), "embedding_share": shares["embedding"], "top3": sorted(shares, key=lambda k: -abs(shares[k]))[:3]}
        return report, closure, fw_
    seqs = [p for p, _, _ in text_items] + [s_ for _, s_, _ in text_items]; pos = [c for _, _, c in text_items] * 2; n = len(text_items)
    res, forwards = capture(seqs, pos, 64); closure = 0.0; report = {"readers": {}}
    import statistics
    for l, h in READERS:
        parts, live = res[(l, h)]; closure = max(closure, float(((sum(parts.values()) - live).norm(dim=1) / live.norm(dim=1)).max()))
        xn = live / live.pow(2).mean(1, keepdim=True).sqrt(); delta = xn[:n] - xn[n:]; d = delta.mean(0)
        attn = blocks[l].attn; Wp = attn.c_proj.weight.detach().float().cpu(); Wv = attn.c_v.weight.detach().float().cpu(); uc = u.cpu()
        Ms = {hh: Wp[:, hh * hd:(hh + 1) * hd] @ Wv[hh * hd:(hh + 1) * hd] for hh in range(N_HEAD)}      # O_hh V_hh, (D, D)
        T = torch.outer(uc, d); Tn = T / T.norm(); I = torch.eye(D)
        cosT = {hh: float((M * Tn).sum() / M.norm()) for hh, M in Ms.items()}; cosI = float((Ms[h] * I).sum() / (Ms[h].norm() * I.norm()))
        transport = delta @ Ms[h].T @ uc; proj = delta @ d
        mt, mp = transport.mean(), proj.mean(); corr = float(((transport - mt) * (proj - mp)).mean() / (transport.std(unbiased=False) * proj.std(unbiased=False)))
        report["readers"][f"{l}.{h}"] = {"cos_contrast_by_head": cosT, "reader_rank": 1 + sum(1 for hh in cosT if cosT[hh] > cosT[h]), "cos_identity": cosI, "energy_share_along_T": cosT[h] ** 2, "corr_transport_vs_projection": corr,
                                          "mean_transport": float(mt), "d_norm": float(d.norm())}
    report["closure_max"] = closure
    print(json.dumps(report, indent=1))
    R = report["readers"]
    predictions = {"pred_a_capture_closure": closure <= CLOSURE_TOL, "pred_b_readers_score_the_contrast_template": all(R[k]["reader_rank"] == 1 for k in R), "pred_c_identity_template_small": all(abs(R[k]["cos_identity"]) <= IDENT_MAX for k in R),
                   "pred_d_rank_one_map_predicts_per_pair_transport": R["9.6"]["corr_transport_vs_projection"] >= CORR_MIN, "pred_e_template_is_a_small_part_of_the_map": all(R[k]["energy_share_along_T"] <= ENERGY_MAX for k in R)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "template_contrast_result_v420", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
