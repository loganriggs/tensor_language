#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_mlp8_is_the_largest_writer_on_text pred_c_mlps_carry_most pred_d_attention_blocks_small pred_e_writer_ranking_replays_panel_to_text
"""Who writes the number the readers copy (v406). v401-v405: the readers' number output is a copy of the noun's CURRENT-STATE value (patterns
number-blind). The value factor w = (u O_h) . c_v(n(live_n)) is linear in the normalised noun residual, so its plural - singular change splits EXACTLY by
writer of the residual entering the reader's block: delta w = sum_k (u O_h) c_v [part_k,P / rms_P - part_k,S / rms_S] (the lambda chain applied to every
part, as in v398). Writers: embedding, each attention block (total), each MLP, up to the reader's block. On the 122 natural swapped pairs (v405) AND the
v76 panel pairs, for 9.6 (block 9), 12.4 (block 12), 15.1 (block 15). v82 (panel, state not contrast): MLP 8 first at the noun for 9.6.
PREDICTIONS (scored as written; failures preserved; priors from v82 / v168)
    pred_a_closure                              per-writer terms sum to delta w within relative 1e-3, every pair and reader, text and panel
    pred_b_mlp8_is_the_largest_writer_on_text   for 9.6 on text the largest |pooled writer term| is mlp8. Prior: likely (v82)
    pred_c_mlps_carry_most                      for 9.6 the MLP writers together carry >= 0.60 of delta w on text and on panel
    pred_d_attention_blocks_small               for 9.6 the attention blocks together carry <= 0.30 of |delta w| on text
    pred_e_writer_ranking_replays_panel_to_text the top-3 writers for 9.6 are the same set on panel and text. Prior: unsure.
PRICE (registered maximum): 4 text batches + 3 panel batches = 7 forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/value_copy_writers_v406_result.json"
CANDIDATE_ID = "both_ends.value_copy_writers_v406"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, MLP_MIN, ATTN_MAX = 1e-3, 0.60, 0.30
READERS = ((9, 6), (12, 4), (15, 1))
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_closure": "<= 1e-3", "pred_b_mlp8_is_the_largest_writer_on_text": "mlp8 first", "pred_c_mlps_carry_most": ">= 0.60 x 2", "pred_d_attention_blocks_small": "<= 0.30", "pred_e_writer_ranking_replays_panel_to_text": "same top-3 set"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mlp_min": MLP_MIN, "attn_max": ATTN_MAX}}
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
    text, c1, f1 = fold(text_items, 64); panel, c2, f2 = fold(panel_items, 32); forwards = f1 + f2; closure = max(c1, c2)
    report = {"closure_max": closure, "text": text, "panel": panel}
    print(json.dumps({"closure": closure, "text_9.6": text["9.6"], "panel_9.6": panel["9.6"]}, indent=1))
    t96, p96 = text["9.6"], panel["9.6"]
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_mlp8_is_the_largest_writer_on_text": t96["top3"][0] == "mlp8", "pred_c_mlps_carry_most": t96["mlp_share"] >= MLP_MIN and p96["mlp_share"] >= MLP_MIN,
                   "pred_d_attention_blocks_small": t96["attn_abs_share"] <= ATTN_MAX, "pred_e_writer_ranking_replays_panel_to_text": set(t96["top3"]) == set(p96["top3"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "value_copy_writers_result_v406", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
