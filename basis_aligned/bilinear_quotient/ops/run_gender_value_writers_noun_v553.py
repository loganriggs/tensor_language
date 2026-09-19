#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_embedding_is_a_quarter_for_10_1 pred_c_attention_carries_for_10_1 pred_d_mlps_below_half_for_10_1 pred_e_readers_agree_on_writers
"""The gender readers' value at the noun, split exactly by writer (v553): the number line's v553 fold on the gender line. v549-v552: gender leaves the noun
through values (0.901), read by 10.1 (0.255) / 12.4 (0.170) / 9.6 (0.107); the self-copies 8.1 / 6.1 plant 0.31 of it, the whole MLP stack ~0.17. Here, for
each of the three gender readers, the male - female change of its value factor w = (uO_h) . v_h at the noun (u = W_U[he] - W_U[she]) splits EXACTLY by writer
of the noun residual entering the reader's block (embedding, each attention block, each MLP; lambda chain applied; shared rms), on the 61 natural gender pairs.
Folds nominate (v552 / v550 decided the MLP and self-copy sizes); this sizes the embedding's own share, which no edit can isolate.
PREDICTIONS (scored as written; failures preserved; priors from the 18 Sep carrier split (embedding 34%, self-copies 39%) and v550-v552)
    pred_a_closure                          per-writer terms sum to the value change within relative 1e-3, every row and reader
    pred_b_embedding_is_a_quarter_for_10_1  the embedding's share of 10.1's value change is >= 0.25
    pred_c_attention_carries_for_10_1       the attention blocks' shares sum (signed) to >= 0.25 for 10.1 (the self-copies at blocks 6 and 8)
    pred_d_mlps_below_half_for_10_1         the MLPs' signed share for 10.1 is <= 0.50
    pred_e_readers_agree_on_writers         10.1 and 12.4 share at least two of their top-3 writers. Prior: unsure
PRICE (registered maximum): 1 text batch x 1 forward (blocks 0-12) = 1 forward; 0 backwards; 0 fits. Bar <= 2.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_gender_route_census_noun_v549 as gv
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_value_writers_noun_v553_result.json"
CANDIDATE_ID = "gender.value_writers_noun_v553"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, EMB_MIN, ATTN_MIN, MLP_MAX, AGREE_MIN = 1e-3, 0.25, 0.25, 0.50, 2
READERS = ((10, 1), (12, 4), (9, 6))
FORWARDS_MAX = 2
PREDICTIONS = {"pred_a_closure": "<= 1e-3", "pred_b_embedding_is_a_quarter_for_10_1": ">= 0.25", "pred_c_attention_carries_for_10_1": ">= 0.25", "pred_d_mlps_below_half_for_10_1": "<= 0.50", "pred_e_readers_agree_on_writers": ">= 2 of top-3"}


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
    text_items = gv.gender_pairs(); panel_items = []
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "emb_min": EMB_MIN, "attn_min": ATTN_MIN, "mlp_max": MLP_MAX, "agree_min": AGREE_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" he")] - WU[L._single(" she")]).cuda()
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
    text, closure, forwards = fold(text_items, 64)
    for r in text.values(): r["attn_signed_share"] = sum(v for k, v in r["writer_shares"].items() if k.startswith("attn"))
    report = {"closure_max": closure, "text": text}
    print(json.dumps({"closure": closure, "text": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "writer_shares"} for k, v in text.items()}}, indent=1))
    print("10.1 writers", {k: round(v, 3) for k, v in list(text["10.1"]["writer_shares"].items())[:8]})
    t = text["10.1"]
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_embedding_is_a_quarter_for_10_1": t["embedding_share"] >= EMB_MIN, "pred_c_attention_carries_for_10_1": t["attn_signed_share"] >= ATTN_MIN,
                   "pred_d_mlps_below_half_for_10_1": t["mlp_share"] <= MLP_MAX, "pred_e_readers_agree_on_writers": len(set(text["10.1"]["top3"]) & set(text["12.4"]["top3"])) >= AGREE_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_value_writers_noun_result_v553", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
