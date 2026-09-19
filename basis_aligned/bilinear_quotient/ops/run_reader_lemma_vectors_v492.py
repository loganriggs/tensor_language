#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_12_4_and_10_1_lean_masculine pred_c_9_6_is_gender_even pred_d_all_three_share_the_axis pred_e_all_three_third_singular_down
"""What the gender-leaking readers write, by lemma (v492). v491: 12.4 and 10.1 move the he - she margin 0.40 as much as they - he; 9.6 0.15. Here the
signed change by lemma (the 80-form pronoun family, v440) under each head's VALUE swap at the noun, for 9.6, 12.4 and 10.1 singly: does the leakage mean
the two heads push the masculine forms (he / his / him / himself) down more than the feminine ones (she / her / herself)?
PREDICTIONS (scored as written; failures preserved; priors from v440 / v491)
    pred_a_baseline_replays          the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_12_4_and_10_1_lean_masculine  for 12.4 and for 10.1 the mean fall of the masculine forms is >= 1.3 x the mean fall of the feminine forms
    pred_c_9_6_is_gender_even        for 9.6 that ratio is between 0.8 and 1.25
    pred_d_all_three_share_the_axis  each head's lemma vector has cosine >= 0.90 with the five-head vector of v440 (recomputed here as the FIVE_N mode)
    pred_e_all_three_third_singular_down  under each head's swap every third-person singular form falls and every third-person plural form rises
PRICE (registered maximum): (4 text + 3 panel batches) x 5 passes = 35 forwards; 0 backwards; 0 fits. Bar <= 36.
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
OUT = ROOT / "circuits/followups/reader_lemma_vectors_v492_result.json"
CANDIDATE_ID = "chain.reader_lemma_vectors_v492"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, MASC_MIN, EVEN_LO, EVEN_HI, COS_MIN = 1e-3, 1.3, 0.8, 1.25, 0.90
_BASE = ["they", "their", "them", "themselves", "we", "our", "us", "ourselves", "he", "his", "him", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "I", "my", "me", "myself", "you", "your", "yourself"]
def _family():
    out = {}
    for w in _BASE:
        for form in (w, w.capitalize()):
            for sp in ("", " "):
                ids = L.ENCODING.encode(sp + form)
                if len(ids) == 1: out[ids[0]] = w
    return out
PLURAL3 = {"they", "their", "them", "themselves"}; MASC = {"he", "his", "him", "himself"}; FEM = {"she", "her", "herself"}
FORWARDS_MAX = 36
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_12_4_and_10_1_lean_masculine": ">= 1.3 x, x 2", "pred_c_9_6_is_gender_even": "0.8-1.25", "pred_d_all_three_share_the_axis": "cos >= 0.90 x 3", "pred_e_all_three_third_singular_down": "signs x 3"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "masc_min": MASC_MIN, "even": [EVEN_LO, EVEN_HI], "cos_min": COS_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE, SHE = L._single(" they"), L._single(" he"), L._single(" she")
    RD = dict(READERS)
    MODES = {"FIVE_N": ({l: hs for l, hs in FIVE.items()}, (0,)), "9.6": ({9: [6]}, (0,)), "12.4": ({12: [4]}, (0,)), "10.1": ({10: [1]}, (0,))}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return torch.stack([z[:, THEY] - z[:, HE], z[:, THEY] - z[:, SHE], z[:, HE] - z[:, SHE]], 1), z
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
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
            m_, lp = logits_margin(x[idx, fin]); return m_.cpu(), lp.cpu()
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        nat_m, nat_z, ed = [], [], {m: [] for m in MODES}
        for s0 in range(0, len(seqs), 2 * batch):
            m_, z = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native"); nat_m.append(m_); nat_z.append(z)
            for m in ed: _, z = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m); ed[m].append(z)
        nat_m, nat_z = torch.cat(nat_m), torch.cat(nat_z); ed = {m: torch.cat(v) for m, v in ed.items()}
        FAM = _family(); out = {"pairs": len(items), "family_size": len(FAM)}; vecs = {}
        for m, z in ed.items():
            signed = (z - nat_z)[1::2]; by_lemma = {}
            for tid, w in FAM.items(): by_lemma.setdefault(w, []).append(float(signed[:, tid].mean()))
            lm = {w: sum(v) / len(v) for w, v in by_lemma.items()}; vecs[m] = lm
            masc = -sum(lm[w] for w in MASC) / len(MASC); fem = -sum(lm[w] for w in FEM) / len(FEM)
            out[m] = {"signed_change_by_lemma": dict(sorted(lm.items(), key=lambda kv: kv[1])), "masc_fall": masc, "fem_fall": fem, "masc_over_fem": masc / fem if abs(fem) > 1e-9 else float("nan"),
                      "signs_ok": all(lm[w] > 0 for w in PLURAL3) and all(lm[w] < 0 for w in MASC | FEM)}
        keys = sorted(vecs["FIVE_N"]); ref = torch.tensor([vecs["FIVE_N"][k] for k in keys])
        for m in vecs:
            v_ = torch.tensor([vecs[m][k] for k in keys]); out[m]["cos_with_five"] = float(v_ @ ref / (v_.norm() * ref.norm()))
        out["native_margins"] = nat_m[:, 0].tolist()
        return out, nat_m[:, 0]
    text, nat_t = study(text_items, 32); panel, nat_p = study(panel_items, 16)
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_12_4_and_10_1_lean_masculine": text["12.4"]["masc_over_fem"] >= MASC_MIN and text["10.1"]["masc_over_fem"] >= MASC_MIN,
                   "pred_c_9_6_is_gender_even": EVEN_LO <= text["9.6"]["masc_over_fem"] <= EVEN_HI, "pred_d_all_three_share_the_axis": all(text[k]["cos_with_five"] >= COS_MIN for k in ("9.6", "12.4", "10.1")),
                   "pred_e_all_three_third_singular_down": all(text[k]["signs_ok"] for k in ("9.6", "12.4", "10.1"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "reader_lemma_vectors_result_v492", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
