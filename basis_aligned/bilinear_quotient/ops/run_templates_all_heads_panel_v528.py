#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture_closure pred_b_pronoun_top10_hits_on_the_panel pred_c_verb_top10_hits_on_the_panel pred_d_pronoun_top10_replays_across_frames pred_e_verb_top10_replays_across_frames
"""The two templates on the PANEL frame's noun difference (v528). v525 (natural pairs): with d = the natural mean noun difference, the pronoun template's top 10
holds all five pronoun readers and the agreement template's top 10 four of the five verb readers. Here d is taken from the 48 v76 panel pairs instead: does the
weight ranking replay across frames?
PREDICTIONS (scored as written; failures preserved; priors from v525)
    pred_a_capture_closure                 per-writer parts sum to the residual within relative 1e-3 at every block (instrument)
    pred_b_pronoun_top10_hits_on_the_panel >= 4 of the 5 pronoun heads are in the pronoun template's top 10 of 162
    pred_c_verb_top10_hits_on_the_panel    >= 3 of the 5 verb heads are in the agreement template's top 10 of 162
    pred_d_pronoun_top10_replays_across_frames  the pronoun template's top 10 shares >= 6 heads with the natural-frame top 10 (v525)
    pred_e_verb_top10_replays_across_frames     the agreement template's top 10 shares >= 6 heads with the natural-frame top 10 (v525)
PRICE (registered maximum): 3 panel batches = 3 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/templates_all_heads_panel_v528_result.json"
CANDIDATE_ID = "template.all_heads_panel_v528"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, PRON_HITS, VERB_HITS, REPLAY_MIN = 1e-3, 4, 3, 6
V525_TOP = {"pronoun": ["9.6", "12.4", "10.1", "10.5", "3.4", "4.5", "15.1", "13.1", "1.2", "2.6"], "verb": ["11.3", "7.8", "2.6", "13.1", "4.5", "9.6", "5.3", "10.5", "1.5", "3.4"]}
PRON_SET = {"9.6", "12.4", "10.5", "15.1", "10.1"}; VERB_SET = {"11.3", "7.8", "13.1", "9.7", "5.3"}
READERS = tuple((l, 0) for l in range(18))   # one entry per block: the capture stores each block's residual; both templates are scored over all 9 heads
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_capture_closure": "<= 1e-3", "pred_b_pronoun_top10_hits_on_the_panel": ">= 4 of 5", "pred_c_verb_top10_hits_on_the_panel": ">= 3 of 5", "pred_d_pronoun_top10_replays_across_frames": ">= 6 shared", "pred_e_verb_top10_replays_across_frames": ">= 6 shared"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "pron_hits": PRON_HITS, "verb_hits": VERB_HITS, "replay_min": REPLAY_MIN}}
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
                tokens = torch.tensor([list(q) + [0] * (max(len(t) for t in seqs[s0:s0 + batch]) - len(q)) for q in seqs[s0:s0 + batch]], device="cuda"); pos = torch.tensor(positions[s0:s0 + batch]); idx = torch.arange(tokens.shape[0])   # padded: panel rows vary in length
                x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {"embedding": x.clone()}
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0
                    for k in parts: parts[k] = block.lambdas[0] * parts[k]
                    parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                    if l in layers:
                        for r in [r_ for r_ in READERS if r_[0] == l]:   # two readers may share a block (9.7 and 9.6)
                            out[r]["parts"].append({k: v[idx, pos].float().cpu() for k, v in parts.items()}); out[r]["live"].append(live[idx, pos].float().cpu())
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
    seqs = [list(p) for p, _, _ in panel_items] + [list(s_) for _, s_, _ in panel_items]; pos = [c for _, _, c in panel_items] * 2; n = len(panel_items)
    res, forwards = capture(seqs, pos, 32); closure = 0.0; report = {"readers": {}}
    import statistics
    tvec = {}
    scores = {"pronoun": {}, "verb": {}}
    for l, h in READERS:
        parts, live = res[(l, h)]; closure = max(closure, float(((sum(parts.values()) - live).norm(dim=1) / live.norm(dim=1)).max()))
        xn = live / live.pow(2).mean(1, keepdim=True).sqrt(); delta = xn[:n] - xn[n:]; d = delta.mean(0)
        attn = blocks[l].attn; Wp = attn.c_proj.weight.detach().float().cpu(); Wv = attn.c_v.weight.detach().float().cpu()
        Ms = {hh: Wp[:, hh * hd:(hh + 1) * hd] @ Wv[hh * hd:(hh + 1) * hd] for hh in range(N_HEAD)}
        for name, uu in (("pronoun", u.cpu()), ("verb", u_v.cpu())):
            T = torch.outer(uu, d); Tn = T / T.norm()
            for hh, M in Ms.items():
                sh = 1.0 if float(uu @ (M @ d)) >= 0 else -1.0; scores[name][f"{l}.{hh}"] = float(sh * (M * Tn).sum() / M.norm())
    ranked = {name: sorted(sc, key=lambda k: -sc[k]) for name, sc in scores.items()}
    rank_of = {name: {k: i + 1 for i, k in enumerate(ranked[name])} for name in ranked}
    import statistics
    report["top10"] = {name: ranked[name][:10] for name in ranked}
    report["named_ranks"] = {"pronoun": {k: rank_of["pronoun"][k] for k in sorted(PRON_SET)}, "verb": {k: rank_of["verb"][k] for k in sorted(VERB_SET)}}
    report["pronoun_hits"] = len(PRON_SET & set(ranked["pronoun"][:10])); report["verb_hits"] = len(VERB_SET & set(ranked["verb"][:10])); report["shared_top10"] = len(set(ranked["pronoun"][:10]) & set(ranked["verb"][:10]))
    report["frame_replay"] = {name: len(set(ranked[name][:10]) & set(V525_TOP[name])) for name in ranked}
    report["median_rank"] = {"pronoun": statistics.median(rank_of["pronoun"][k] for k in PRON_SET), "verb": statistics.median(rank_of["verb"][k] for k in VERB_SET)}
    report["closure_max"] = closure
    print(json.dumps(report, indent=1))
    R = report["readers"]
    predictions = {"pred_a_capture_closure": closure <= CLOSURE_TOL, "pred_b_pronoun_top10_hits_on_the_panel": report["pronoun_hits"] >= PRON_HITS, "pred_c_verb_top10_hits_on_the_panel": report["verb_hits"] >= VERB_HITS,
                   "pred_d_pronoun_top10_replays_across_frames": report["frame_replay"]["pronoun"] >= REPLAY_MIN, "pred_e_verb_top10_replays_across_frames": report["frame_replay"]["verb"] >= REPLAY_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "templates_all_heads_panel_result_v528", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
