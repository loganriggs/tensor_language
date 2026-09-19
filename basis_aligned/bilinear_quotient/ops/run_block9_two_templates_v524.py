#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture_closure pred_b_pronoun_template_prefers_9_6 pred_c_verb_template_prefers_9_7 pred_d_block_9_transport_vectors_distinct pred_e_cross_ranks_low
"""Block 9's two readers, two templates (v524). v421 / v523: in block 9 the pronoun template (they - he) ranks 9.6 first and the verb template (agreement)
ranks 9.7 first. Here both templates on every head of block 9, natural pairs: the 2 x 2 of 9.6 / 9.7 against u / u_v, the cross ranks (9.7 on the pronoun
template, 9.6 on the verb template), and the cosine of their transport vectors (v523: -0.11).
PREDICTIONS (scored as written; failures preserved; priors from v421 / v523)
    pred_a_capture_closure                per-writer parts sum to the residual within relative 1e-3 (instrument)
    pred_b_pronoun_template_prefers_9_6   on the they - he template 9.6's cos_F is >= 3 x 9.7's
    pred_c_verb_template_prefers_9_7      on the agreement template 9.7's cos_F is >= 3 x 9.6's. Prior: unsure (9.7's own cos is small, 0.007)
    pred_d_block_9_transport_vectors_distinct  |cos(M_9.6^T u, M_9.7^T u_v)| <= 0.30
    pred_e_cross_ranks_low                9.7 ranks outside the top 2 on the pronoun template and 9.6 outside the top 2 on the verb template. Prior: unsure
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
OUT = ROOT / "circuits/followups/block9_two_templates_v524_result.json"
CANDIDATE_ID = "template.block9_two_templates_v524"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, RATIO_MIN, DISTINCT_MAX = 1e-3, 3.0, 0.30
READERS = ((9, 6), (9, 7))   # block 9: the pronoun reader (on u) and the verb reader (on u_v)
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_capture_closure": "<= 1e-3", "pred_b_pronoun_template_prefers_9_6": ">= 3 x", "pred_c_verb_template_prefers_9_7": ">= 3 x", "pred_d_block_9_transport_vectors_distinct": "|cos| <= 0.30", "pred_e_cross_ranks_low": "rank > 2 x 2"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "ratio_min": RATIO_MIN, "distinct_max": DISTINCT_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda(); u_v = (WU[L._single(" are")] - WU[L._single(" is")] + WU[L._single(" were")] - WU[L._single(" was")]).cuda()
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
                        for r in [r_ for r_ in READERS if r_[0] == l]:   # two readers may share a block (9.7 and 9.6)
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
    tvec = {}
    for l, h in READERS:
        parts, live = res[(l, h)]; closure = max(closure, float(((sum(parts.values()) - live).norm(dim=1) / live.norm(dim=1)).max()))
        xn = live / live.pow(2).mean(1, keepdim=True).sqrt(); delta = xn[:n] - xn[n:]; d = delta.mean(0)
        attn = blocks[l].attn; Wp = attn.c_proj.weight.detach().float().cpu(); Wv = attn.c_v.weight.detach().float().cpu(); uc = (u if (l, h) == (9, 6) else u_v).cpu()
        Ms = {hh: Wp[:, hh * hd:(hh + 1) * hd] @ Wv[hh * hd:(hh + 1) * hd] for hh in range(N_HEAD)}      # O_hh V_hh, (D, D)
        T = torch.outer(uc, d); Tn = T / T.norm(); I = torch.eye(D)
        sh = 1.0 if float(uc @ (Ms[h] @ d)) >= 0 else -1.0; cosT = {hh: float(sh * (M * Tn).sum() / M.norm()) for hh, M in Ms.items()}
        transport = delta @ Ms[h].T @ uc; tvec[f"{l}.{h}"] = Ms[h].T @ uc
        sv = torch.linalg.svd(delta - delta.mean(0), full_matrices=False); V = sv.Vh
        var_tot = float(transport.var(unbiased=False)); shares = {}
        for k in range(1, 9):
            coords = (delta - delta.mean(0)) @ V[:k].T; recon = coords @ (V[:k] @ tvec[f"{l}.{h}"]) + float(transport.mean()); shares[k] = 1 - float((transport - recon).var(unbiased=False)) / var_tot
        cos_d = float(abs(V[0] @ d) / d.norm())
        report["readers"][f"{l}.{h}"] = {"sign": sh, "cos_contrast_by_head": cosT, "reader_rank": 1 + sum(1 for hh in cosT if cosT[hh] > cosT[h]), "transport_variance_explained_by_topk": shares, "cos_top_direction_vs_d": cos_d, "mean_transport": float(transport.mean())}
    keys = list(tvec); report["transport_vector_cos"] = {f"{a_}|{b_}": float(tvec[a_] @ tvec[b_] / (tvec[a_].norm() * tvec[b_].norm())) for i, a_ in enumerate(keys) for b_ in keys[i + 1:]}
    report["closure_max"] = closure
    print(json.dumps(report, indent=1))
    R = report["readers"]
    tc = report["transport_vector_cos"]; pron = R["9.6"]["cos_contrast_by_head"]; verbT = R["9.7"]["cos_contrast_by_head"]   # 9.6's entry holds the they - he template over block 9; 9.7's the agreement template
    rank = lambda tbl, h: 1 + sum(1 for hh in tbl if tbl[hh] > tbl[h])
    report["cross"] = {"pronoun_template": {"9.6": pron[6], "9.7": pron[7], "rank_9_7": rank(pron, 7)}, "verb_template": {"9.6": verbT[6], "9.7": verbT[7], "rank_9_6": rank(verbT, 6)}}
    predictions = {"pred_a_capture_closure": closure <= CLOSURE_TOL, "pred_b_pronoun_template_prefers_9_6": pron[6] >= RATIO_MIN * max(pron[7], 1e-9), "pred_c_verb_template_prefers_9_7": verbT[7] >= RATIO_MIN * max(verbT[6], 1e-9),
                   "pred_d_block_9_transport_vectors_distinct": abs(tc["9.6|9.7"]) <= DISTINCT_MAX, "pred_e_cross_ranks_low": rank(pron, 7) > 2 and rank(verbT, 6) > 2}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "block9_two_templates_result_v524", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
