#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_5_3_reads_the_post_noun_site pred_c_5_3_reads_the_noun_little pred_d_5_3_two_sites_at_least_the_post_noun pred_e_11_3_at_the_noun_replays
"""Where head 5.3 reads number from (v478). v477: head 5.3 writes 0.113 of the agreement margin into a distant verb, but block 5's exit from the NOUN was
0.03 (v447), so 5.3 must read number from another position — the post-noun token, where MLPs 4-8 rebuild it from the copier's seed (v383-v385, v409). Here
5.3's value swapped within the pair at the noun, at the post-noun token, and at both (79 distant rows, verb margin); 11.3 at the noun as the replay (0.147).
PREDICTIONS (scored as written; failures preserved; priors from v447 / v477)
    pred_a_baseline_replays              the unedited manual forward replays the model's verb margins within 1e-3
    pred_b_5_3_reads_the_post_noun_site  5.3's value at the post-noun token closes >= 0.05 of the margin gap. Prior: likely
    pred_c_5_3_reads_the_noun_little     5.3's value at the noun closes <= 0.04
    pred_d_5_3_two_sites_at_least_the_post_noun  5.3's values at both sites close >= the post-noun value alone - 0.01
    pred_e_11_3_at_the_noun_replays      11.3's value at the noun closes 0.147 +- 0.03 (v448)
PRICE (registered maximum): 3 batches x 5 passes = 15 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_value_copy_writers_v406 as v406
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/head_5_3_source_v478_result.json"
CANDIDATE_ID = "chain.head_5_3_source_v478"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, POST_MIN, NOUN_MAX, SLACK, H113_V448, BAND = 1e-3, 0.05, 0.04, 0.01, 0.147, 0.03
UNITS = (829, 953, 1030); MLP_LAYER = 8
SING_V = (" is", " was", " has", " does"); PLUR_V = (" are", " were", " have", " do")
_BASE = ["they", "their", "them", "themselves", "we", "our", "us", "ourselves", "he", "his", "him", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "I", "my", "me", "myself", "you", "your", "yourself"]
def _family():
    out = {}
    for w in _BASE:
        for form in (w, w.capitalize()):
            for sp in ("", " "):
                ids = L.ENCODING.encode(sp + form)
                if len(ids) == 1: out[ids[0]] = w
    return out
PLURAL3 = {"they", "their", "them", "themselves"}; SING3 = {"he", "his", "him", "himself", "she", "her", "herself"}
FORWARDS_MAX = 16
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_5_3_reads_the_post_noun_site": ">= 0.05", "pred_c_5_3_reads_the_noun_little": "<= 0.04", "pred_d_5_3_two_sites_at_least_the_post_noun": ">= post - 0.01", "pred_e_11_3_at_the_noun_replays": "0.147 +- 0.03"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    E0 = L.ENCODING
    def partner(tok):
        t = E0.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E0.encode(c)) == 1: return E0.encode(c)[0]
        return None
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]]; text_items, verb_items = [], []
    for r_ in recs:
        c = r_["cue_offset"]; alt = partner(r_["ids"][c])
        if alt is None: continue
        sw = list(r_["ids"]); sw[c] = alt; plural, singular = (r_["ids"], sw) if r_["cue"] == "plural" else (sw, r_["ids"])
        text_items.append((plural, singular, c))
        vs = r_["second_offset"]
        if vs > c: verb_items.append((plural[:vs], singular[:vs], c, vs - c))   # truncated so the answer position IS the verb slot (the verb is predicted there)
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "post_min": POST_MIN, "noun_max": NOUN_MAX, "slack": SLACK, "h113_v448": H113_V448, "band": BAND}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; ARE, IS, WERE, WAS = L._single(" are"), L._single(" is"), L._single(" were"), L._single(" was")
    RD = dict(READERS)
    ALL_H = {l: list(range(N_HEAD)) for l in range(18)}
    MODES = {"n": ({5: [3]}, (0,)), "n1": ({5: [3]}, (1,)), "both": ({5: [3]}, (0, 1)), "h113_n": ({11: [3]}, (0,))}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return torch.stack([(z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS]), z[:, ARE] - z[:, IS], z[:, WERE] - z[:, WAS]], 1), z
    def run(seqs, pos, mode):
        """mode: native or a MODES key; values only, at the listed offsets from the noun. Pairs are rows (2i, 2i+1)."""
        nonlocal forwards
        tokens = torch.tensor(seqs, device="cuda") if len({len(s) for s in seqs}) == 1 else torch.tensor([list(s) + [0] * (max(len(t) for t in seqs) - len(s)) for s in seqs], device="cuda")
        B_ = tokens.shape[0]; idx = torch.arange(B_, device="cuda"); pp = torch.tensor(pos, device="cuda"); fin = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        swap = torch.arange(B_, device="cuda"); swap[0::2], swap[1::2] = torch.arange(1, B_, 2, device="cuda"), torch.arange(0, B_, 2, device="cuda")
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin = F.rms_norm(live, (D,)); attn = block.attn
                if mode != "native" and l in MODES[mode][0]:
                    hs, offs = MODES[mode][0][l], MODES[mode][1]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in hs:
                        for o in offs: v[idx, pp + o, h] = v[swap, pp + o, h].clone()
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; xm = F.rms_norm(x, (D,))
                if mode == "UNITS" and l == MLP_LAYER:
                    mlp = block.mlp; Lx, Rx = mlp.Left(xm), mlp.Right(xm); hdn = ((F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx)).clone()
                    units = torch.tensor(UNITS, device="cuda"); hdn[idx[:, None], pp[:, None], units[None, :]] = hdn[swap[:, None], pp[:, None], units[None, :]].clone()
                    x = x + mlp.Down(hdn) + mlp.Down_bias
                else:
                    x = x + block.mlp(xm)
            forwards += 1
            m_, lp = logits_margin(x[idx, fin]); return m_.cpu(), lp.cpu()
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        nat_m, nat_z, ed = [], [], {m: [] for m in MODES}
        for s0 in range(0, len(seqs), 2 * batch):
            m_, z = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native"); nat_m.append(m_); nat_z.append(z)
            for m in ed: mm, _ = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m); ed[m].append(mm)
        nat_m, nat_z = torch.cat(nat_m), torch.cat(nat_z); ed = {m: torch.cat(v) for m, v in ed.items()}
        gap = nat_m[0::2] - nat_m[1::2]; out = {"pairs": len(items), "gap_mean": {"verb": float(gap[:, 0].mean()), "are_is": float(gap[:, 1].mean()), "were_was": float(gap[:, 2].mean())}, "closed": {}}
        for m, em in ed.items():
            out["closed"][m] = {"verb": float(((nat_m[0::2, 0] - em[0::2, 0]) + (em[1::2, 0] - nat_m[1::2, 0])).sum() / (2 * gap[:, 0].sum())), "are_is": float(((nat_m[0::2, 1] - em[0::2, 1]) + (em[1::2, 1] - nat_m[1::2, 1])).sum() / (2 * gap[:, 1].sum()))}
        out["native_margins"] = nat_m[:, 0].tolist()
        return out, nat_m[:, 0]
    adj_items = [(p, s_, c) for p, s_, c, d in verb_items if d == 1]; dist_items = [(p, s_, c) for p, s_, c, d in verb_items if d >= 2]
    dist, nat_t = study(dist_items, 32); verb = dist; text = dist; panel = {"skipped": "distant rows only", "distant_rows": len(dist_items)}; verb_items = dist_items
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in verb_items[:32]] + [list(s) for _, s, _ in verb_items[:32]]; seqs = [q + [0] * (max(len(t) for t in seqs) - len(q)) for q in seqs]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        fin_ = torch.tensor([len(p) - 1 for p, _, _ in verb_items[:32]] * 2, device="cuda"); z = hook["z"].float()[torch.arange(64, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = ((z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS])).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": panel}
    print(json.dumps(report, indent=1))
    report["verb_slot"] = {k: v for k, v in verb.items() if k != "native_margins"}
    c = verb["closed"]
    v = {m: c[m]["verb"] for m in c}
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_5_3_reads_the_post_noun_site": v["n1"] >= POST_MIN, "pred_c_5_3_reads_the_noun_little": abs(v["n"]) <= NOUN_MAX,
                   "pred_d_5_3_two_sites_at_least_the_post_noun": v["both"] >= v["n1"] - SLACK, "pred_e_11_3_at_the_noun_replays": abs(v["h113_n"] - H113_V448) <= BAND}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head_5_3_source_result_v478", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
