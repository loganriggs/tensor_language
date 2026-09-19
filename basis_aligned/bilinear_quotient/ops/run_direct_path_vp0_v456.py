#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_late_part_aligns_with_vp0 pred_c_late_part_more_vp0_than_early pred_d_vp0_carries_a_share_of_the_direct_change pred_e_u_v_and_vp0_distinct
"""Is the direct path's late-MLP write the panel axis VP0 (v456)? v390-v396: the plural-noun class's top residual axis of W_U (VP0: plural nouns vs
is / has / goes) carried half the PANEL output and was written by MLPs 12-17, but 0.2% on natural text at the pronoun slot — panel-only. v455: on the
direct path to an adjacent verb, MLPs 12-17 write 0.34 of the agreement change. Here VP0 is recomputed (top right-singular vector of the 256 plural-noun
rows of W_U about their mean, v390's construction) and the direct change's parts are projected on it: the late-MLP part (12-17), the early part (<= 8), the
whole; and its cosine with u_v.
PREDICTIONS (scored as written; failures preserved; priors from v390-v396 / v455)
    pred_a_closure                            per-writer terms sum to the change within relative 1e-3
    pred_b_late_part_aligns_with_vp0          |cos(late-MLP part of the mean change, VP0)| >= 0.30. Prior: unsure
    pred_c_late_part_more_vp0_than_early      the late part's |cos| with VP0 exceeds the early part's
    pred_d_vp0_carries_a_share_of_the_direct_change  the whole mean change has >= 0.10 of its squared norm along VP0
    pred_e_u_v_and_vp0_distinct               |cos(u_v, VP0)| <= 0.60 (VP0 is not simply the agreement direction)
PRICE (registered maximum): 2 text batches = 2 forwards; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/direct_path_vp0_v456_result.json"
CANDIDATE_ID = "chain.direct_path_vp0_v456"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, COS_MIN, SHARE_MIN, DISTINCT_MAX = 1e-3, 0.30, 0.10, 0.60
READERS = ((18, 0),)   # the final residual (after block 17), read directly by W_U
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-3", "pred_b_late_part_aligns_with_vp0": "|cos| >= 0.30", "pred_c_late_part_more_vp0_than_early": "late > early", "pred_d_vp0_carries_a_share_of_the_direct_change": ">= 0.10", "pred_e_u_v_and_vp0_distinct": "|cos| <= 0.60"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cos_min": COS_MIN, "share_min": SHARE_MIN, "distinct_max": DISTINCT_MAX}}
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
                    m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = m.clone()
                r = READERS[0]; out[r]["parts"].append({k: v[idx, pos].float().cpu() for k, v in parts.items()}); out[r]["live"].append(x[idx, pos].float().cpu())   # the final residual at the noun
        res = {}
        for r in READERS:
            keys = out[r]["parts"][0].keys(); res[r] = ({k: torch.cat([p[k] for p in out[r]["parts"]]) for k in keys}, torch.cat(out[r]["live"]))
        return res, -(-len(seqs) // batch)
    def fold(items, batch):
        seqs = [p for p, _, _ in items] + [s for _, s, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
        res, fw_ = capture(seqs, pos, batch); report = {}; closure = 0.0; vecs = {}
        for r in READERS:
            parts, live = res[r]; m = maps[r].cpu(); rms = live.pow(2).mean(1, keepdim=True).sqrt()
            wP = (live[:n] / rms[:n]) @ m; wS = (live[n:] / rms[n:]) @ m; dw = wP - wS
            terms = {k: ((v[:n] / rms[:n]) @ m - (v[n:] / rms[n:]) @ m) for k, v in parts.items()}
            recon = sum(terms.values()); closure = max(closure, float(((recon - dw).abs() / dw.abs().clamp_min(1e-6)).max()))
            pooled = {k: float(v.sum()) for k, v in terms.items()}; total = float(dw.sum())
            shares = {k: v / total for k, v in pooled.items()}
            vecs[f"{r[0]}.{r[1]}"] = {k: ((v[:n] / rms[:n]) - (v[n:] / rms[n:])).mean(0) for k, v in parts.items()}
            report[f"{r[0]}.{r[1]}"] = {"delta_w_total": total, "writer_shares": dict(sorted(shares.items(), key=lambda kv: -abs(kv[1]))), "mlp_share": sum(v for k, v in shares.items() if k.startswith("mlp")),
                                        "attn_abs_share": sum(abs(v) for k, v in shares.items() if k.startswith("attn")), "embedding_share": shares["embedding"], "top3": sorted(shares, key=lambda k: -abs(shares[k]))[:3]}
        return report, closure, fw_, vecs
    text, closure, forwards, vecs = fold(text_items, 64); panel = {"skipped": "adjacent natural rows only"}
    WUc = WU.cpu(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; UP = WUc[torch.tensor([b for _, b in pairs])]; VP0 = torch.linalg.svd(UP - UP.mean(0), full_matrices=False).Vh[0]
    V = vecs["18.0"]; late = sum(V[k] for k in V if k.startswith("mlp") and int(k[3:]) >= 12); early = sum(V[k] for k in V if k.startswith("mlp") and int(k[3:]) <= 8); whole = sum(V.values()); uc = u.cpu()
    cos = lambda a_, b_: float(a_ @ b_ / (a_.norm() * b_.norm()))
    if cos(VP0, uc) < 0: VP0 = -VP0
    vp0 = {"cos_late_vp0": cos(late, VP0), "cos_early_vp0": cos(early, VP0), "cos_whole_vp0": cos(whole, VP0), "share_whole_along_vp0": float((whole @ VP0) ** 2 / (whole @ whole)), "cos_u_v_vp0": cos(uc, VP0), "cos_late_u_v": cos(late, uc), "cos_early_u_v": cos(early, uc)}
    report = {"closure_max": closure, "text": text, "panel": panel, "vp0": vp0}
    t96 = text["18.0"]
    print(json.dumps(vp0, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_late_part_aligns_with_vp0": abs(vp0["cos_late_vp0"]) >= COS_MIN, "pred_c_late_part_more_vp0_than_early": abs(vp0["cos_late_vp0"]) > abs(vp0["cos_early_vp0"]),
                   "pred_d_vp0_carries_a_share_of_the_direct_change": vp0["share_whole_along_vp0"] >= SHARE_MIN, "pred_e_u_v_and_vp0_distinct": abs(vp0["cos_u_v_vp0"]) <= DISTINCT_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "direct_path_vp0_result_v456", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
