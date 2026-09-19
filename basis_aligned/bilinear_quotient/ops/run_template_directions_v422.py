#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture_closure pred_b_nuisance_direction_is_noun_identity pred_c_transport_directions_align_with_d pred_d_transport_directions_align_with_829 pred_e_transport_vector_is_mlp8_written
"""What the readers' transport directions are (v422). v421: the largest principal direction of the 122 noun differences carries none of the readers'
number transport; directions 2-4 carry 0.92 (9.6). Here the directions are named: (a) is direction 1 the noun's IDENTITY (its variance across pairs
comes from which noun, not from number: test = its coordinate's correlation with the token-embedding difference's coordinate); (b) do the transport-carrying
directions 2-4 span d (the mean difference) and (c) the MLP-8 unit 829's Down column (the number writer, v168 / v408); (d) is the transport vector
M_9.6^T u itself written by MLP 8: its cos with the span of the top-3 Down columns of MLP 8 by contrast (829 / 953 / 1030).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_capture_closure                     per-writer parts sum to the residual within relative 1e-3 (instrument)
    pred_b_nuisance_direction_is_noun_identity  |corr| over pairs between direction 1's coordinate of delta_i and the same direction's coordinate of the raw embedding difference >= 0.60 for 9.6
    pred_c_transport_directions_align_with_d   the projection of d onto span(directions 2-4) carries >= 0.60 of |d|^2 for 9.6
    pred_d_transport_directions_align_with_829  the projection of Down[:, 829] onto span(directions 2-4) carries >= 0.30 of its norm^2 for 9.6. Prior: unsure
    pred_e_transport_vector_is_mlp8_written    cos between M_9.6^T u and its projection onto span(Down[:, 829], Down[:, 953], Down[:, 1030]) >= 0.50. Prior: unsure
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
OUT = ROOT / "circuits/followups/template_directions_v422_result.json"
CANDIDATE_ID = "template.directions_v422"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, IDENT_MIN, D_MIN, U829_MIN, MLP8_MIN = 1e-3, 0.60, 0.60, 0.30, 0.50
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_capture_closure": "<= 1e-3", "pred_b_nuisance_direction_is_noun_identity": "|corr| >= 0.60", "pred_c_transport_directions_align_with_d": ">= 0.60", "pred_d_transport_directions_align_with_829": ">= 0.30", "pred_e_transport_vector_is_mlp8_written": "cos >= 0.50"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "ident_min": IDENT_MIN, "d_min": D_MIN, "u829_min": U829_MIN, "mlp8_min": MLP8_MIN}}
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
        transport = delta @ Ms[h].T @ uc; tv = Ms[h].T @ uc; tvec[f"{l}.{h}"] = tv
        emb = parts["embedding"]; embd = emb[:n] - emb[n:]                                   # the token-embedding part of the difference (lambda-chained)
        sv = torch.linalg.svd(delta - delta.mean(0), full_matrices=False); V = sv.Vh; c1 = (delta - delta.mean(0)) @ V[0]; e1 = (embd - embd.mean(0)) @ V[0]
        corr1 = float(((c1 - c1.mean()) * (e1 - e1.mean())).mean() / (c1.std(unbiased=False) * e1.std(unbiased=False)))
        S = V[1:4]; proj = lambda x: float(((S @ x) ** 2).sum() / (x @ x))
        Dw8 = blocks[8].mlp.Down.weight.detach().float().cpu(); u829 = Dw8[:, 829]; B3 = torch.linalg.qr(torch.stack([Dw8[:, j] for j in (829, 953, 1030)], 1)).Q
        cos_mlp8 = float((B3.T @ tv).norm() / tv.norm())
        report["readers"][f"{l}.{h}"] = {"corr_dir1_with_embedding_coordinate": corr1, "d_share_in_dirs_2_4": proj(d), "u829_share_in_dirs_2_4": proj(u829), "cos_transport_vs_mlp8_top3_span": cos_mlp8,
                                          "transport_var_dir1": float(((delta - delta.mean(0)) @ V[0]).var(unbiased=False) * float(V[0] @ tv) ** 2 / transport.var(unbiased=False)), "mean_transport": float(transport.mean())}
    report["closure_max"] = closure
    print(json.dumps(report, indent=1))
    R = report["readers"]
    r96 = R["9.6"]
    predictions = {"pred_a_capture_closure": closure <= CLOSURE_TOL, "pred_b_nuisance_direction_is_noun_identity": abs(r96["corr_dir1_with_embedding_coordinate"]) >= IDENT_MIN, "pred_c_transport_directions_align_with_d": r96["d_share_in_dirs_2_4"] >= D_MIN,
                   "pred_d_transport_directions_align_with_829": r96["u829_share_in_dirs_2_4"] >= U829_MIN, "pred_e_transport_vector_is_mlp8_written": r96["cos_transport_vs_mlp8_top3_span"] >= MLP8_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "template_directions_result_v422", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
