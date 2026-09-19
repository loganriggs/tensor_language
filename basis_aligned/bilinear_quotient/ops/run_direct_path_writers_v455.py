#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_mlps_carry_most_of_the_direct_path pred_c_late_mlps_carry_a_third pred_d_mlp8_in_the_top_three pred_e_attention_small
"""Who writes the direct path to an adjacent verb (v455). v446 / v454: when the verb follows the noun directly, the noun's own final residual carries the
agreement to the logits (values at the noun 0.26 of the margin only). The final residual at the noun splits EXACTLY by writer (embedding, each attention
block, each MLP, lambda chain applied, shared final rms), projected on u_v = W_U[are] - W_U[is] + W_U[were] - W_U[was]: the plural - singular change of the
direct agreement signal by writer, on the 43 adjacent natural pairs. v390-v396 found a panel-only class axis written by MLPs 12-17 — the direct path is where
late MLPs could matter for real.
PREDICTIONS (scored as written; failures preserved; priors from v390-v396 / v406 / v454)
    pred_a_closure                          per-writer terms sum to the change within relative 1e-3, every pair
    pred_b_mlps_carry_most_of_the_direct_path  the MLP writers together carry >= 0.70 of the direct agreement change
    pred_c_late_mlps_carry_a_third          MLPs 12-17 together carry >= 0.30. Prior: unsure
    pred_d_mlp8_in_the_top_three            mlp8 is among the three largest |writer terms|
    pred_e_attention_small                  the attention blocks together carry <= 0.20 of |change|
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
OUT = ROOT / "circuits/followups/direct_path_writers_v455_result.json"
CANDIDATE_ID = "chain.direct_path_writers_v455"
N_HEAD = 9
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
CLOSURE_TOL, MLP_MIN, LATE_MIN, ATTN_MAX = 1e-3, 0.70, 0.30, 0.20
READERS = ((18, 0),)   # the final residual (after block 17), read directly by W_U
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-3", "pred_b_mlps_carry_most_of_the_direct_path": ">= 0.70", "pred_c_late_mlps_carry_a_third": ">= 0.30", "pred_d_mlp8_in_the_top_three": "top 3", "pred_e_attention_small": "<= 0.20"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mlp_min": MLP_MIN, "late_min": LATE_MIN, "attn_max": ATTN_MAX}}
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
    text, closure, forwards = fold(text_items, 64); panel = {"skipped": "adjacent natural rows only"}
    report = {"closure_max": closure, "text": text, "panel": panel}
    print(json.dumps({"closure": closure, "direct": text["18.0"]}, indent=1))
    t96 = text["18.0"]; late = sum(v for k, v in t96["writer_shares"].items() if k.startswith("mlp") and int(k[3:]) >= 12)
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_mlps_carry_most_of_the_direct_path": t96["mlp_share"] >= MLP_MIN, "pred_c_late_mlps_carry_a_third": late >= LATE_MIN,
                   "pred_d_mlp8_in_the_top_three": "mlp8" in t96["top3"], "pred_e_attention_small": t96["attn_abs_share"] <= ATTN_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "direct_path_writers_result_v455", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
