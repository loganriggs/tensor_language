#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture_closure pred_b_sign_aware_template_picks_all_three pred_c_transport_needs_few_directions pred_d_top_direction_is_d pred_e_directions_shared_across_readers
"""Template contraction, second pass (v421). v420: the contrast template u d^T finds 9.6 and 12.4 among their blocks' heads by weights alone but
misses 15.1, whose number transport enters u with a negative sign (v403: its w moves -237% while its pattern weight is negative), and the rank-one
template predicts the per-pair transport only at corr 0.71. Here (a) the SIGN-AWARE template s_h u d^T with s_h = sign of the head's mean pattern weight at
the noun (from the same capture), scored across all 27 heads of blocks 9 / 12 / 15; (b) how many noun-difference directions the transport uses: the
principal directions of the 122 per-pair differences delta_i (an SVD summary, no fit), and the share of the transport's variance explained by the
projection of delta_i onto the top-k directions for k = 1..8, exactly (the transport is linear in delta_i); (c) whether the top direction is d and
whether the readers use the same directions (cos of their transport vectors M_h^T u).
PREDICTIONS (scored as written; failures preserved; priors from v420)
    pred_a_capture_closure                    per-writer parts sum to the residual within relative 1e-3 (instrument)
    pred_b_sign_aware_template_picks_all_three  with s_h applied, each reader is the top head of its block on cos_F(M_h, s_h u d^T). Prior: likely for 15.1 now
    pred_c_transport_needs_few_directions     for 9.6 the top-4 principal directions of delta explain >= 0.80 of the transport's variance over pairs. Prior: unsure
    pred_d_top_direction_is_d                 the first principal direction of delta has |cos| >= 0.90 with d (the mean difference dominates the pair differences). Prior: unsure
    pred_e_directions_shared_across_readers   |cos(M_9.6^T u, M_12.4^T u)| >= 0.50 (the two positive readers read the same noun directions). Prior: unsure
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
OUT = ROOT / "circuits/followups/template_contrast_v421_result.json"
CANDIDATE_ID = "template.contrast_readers_v421"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, VAR_MIN, COS_D_MIN, SHARED_MIN = 1e-3, 0.80, 0.90, 0.50
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_capture_closure": "<= 1e-3", "pred_b_sign_aware_template_picks_all_three": "rank 1 x 3", "pred_c_transport_needs_few_directions": ">= 0.80 at k = 4", "pred_d_top_direction_is_d": "|cos| >= 0.90", "pred_e_directions_shared_across_readers": "|cos| >= 0.50"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "var_min": VAR_MIN, "cos_d_min": COS_D_MIN, "shared_min": SHARED_MIN}}
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
    v411 = json.loads((ROOT / "circuits/followups/pattern_weight_distribution_v411_result.json").read_text())["report"]["readers"]; SIGN = {k: (1.0 if v411[k]["text"]["mean"] > 0 else -1.0) for k in v411}
    tvec = {}
    for l, h in READERS:
        parts, live = res[(l, h)]; closure = max(closure, float(((sum(parts.values()) - live).norm(dim=1) / live.norm(dim=1)).max()))
        xn = live / live.pow(2).mean(1, keepdim=True).sqrt(); delta = xn[:n] - xn[n:]; d = delta.mean(0)
        attn = blocks[l].attn; Wp = attn.c_proj.weight.detach().float().cpu(); Wv = attn.c_v.weight.detach().float().cpu(); uc = u.cpu()
        Ms = {hh: Wp[:, hh * hd:(hh + 1) * hd] @ Wv[hh * hd:(hh + 1) * hd] for hh in range(N_HEAD)}      # O_hh V_hh, (D, D)
        T = torch.outer(uc, d); Tn = T / T.norm(); I = torch.eye(D)
        sh = SIGN[f"{l}.{h}"]; cosT = {hh: float(sh * (M * Tn).sum() / M.norm()) for hh, M in Ms.items()}
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
    predictions = {"pred_a_capture_closure": closure <= CLOSURE_TOL, "pred_b_sign_aware_template_picks_all_three": all(R[k]["reader_rank"] == 1 for k in R), "pred_c_transport_needs_few_directions": R["9.6"]["transport_variance_explained_by_topk"][4] >= VAR_MIN,
                   "pred_d_top_direction_is_d": R["9.6"]["cos_top_direction_vs_d"] >= COS_D_MIN, "pred_e_directions_shared_across_readers": abs(report["transport_vector_cos"]["9.6|12.4"]) >= SHARED_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "template_contrast_result_v421", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
