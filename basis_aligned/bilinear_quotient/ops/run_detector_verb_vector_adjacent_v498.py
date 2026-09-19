#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_detector_same_vector_on_the_direct_path pred_c_detector_signs_adjacent pred_d_detector_change_adjacent pred_e_11_3_small_on_adjacent
"""The detector's verb vector on the DIRECT path (v498). v497 (distant verbs): 701 / 2059 swapped at the verb write the readers' agreement vector (cosine 0.908
with 11.3). For a verb directly after the noun the answer position is the noun and the detector reads the noun's own residual (v460: 0.079 of the margin).
Here the same two swaps on the 43 adjacent rows: the detector at the answer (= noun) position, and 11.3's value at the noun (which cannot reach the answer
by copying here), scored by verb; the detector's vector compared with 11.3's DISTANT vector from v496 (stored constants).
PREDICTIONS (scored as written; failures preserved; priors from v444 / v479)
    pred_a_baseline_replays                  the unedited manual forward replays the model's verb margins within 1e-3
    pred_a_baseline_replays                      the unedited manual forward replays the model's verb margins within 1e-3
    pred_b_detector_same_vector_on_the_direct_path  cosine over the eight verbs between the detector's adjacent-row vector and 11.3's distant-row vector (v496) >= 0.85
    pred_c_detector_signs_adjacent               on adjacent rows the detector swap makes is / was / has / does fall and are / were / have / do rise
    pred_d_detector_change_adjacent              the detector's mean |verb change| on adjacent rows is >= 0.05 (its direct-path role, v460)
    pred_e_11_3_small_on_adjacent                11.3's value swap at the noun changes the verbs by <= 0.04 on adjacent rows (no copy reaches the noun's own position). Prior: unsure
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
OUT = ROOT / "circuits/followups/detector_verb_vector_adjacent_v498_result.json"
CANDIDATE_ID = "chain.detector_verb_vector_adjacent_v498"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, COS_MIN, DET_MIN, H113_MAX = 1e-3, 0.85, 0.05, 0.04
V496_113 = {"are": 0.195, "do": 0.098, "does": -0.071, "has": -0.071, "have": 0.151, "is": -0.070, "was": -0.064, "were": 0.177}
UNITS = (701, 2059); MLP_LAYER = 17
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
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_detector_same_vector_on_the_direct_path": "cos >= 0.85", "pred_c_detector_signs_adjacent": "signs", "pred_d_detector_change_adjacent": ">= 0.05", "pred_e_11_3_small_on_adjacent": "<= 0.04"}


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
        if vs > c: verb_items.append((plural[:vs], singular[:vs], c))   # truncated so the answer position IS the verb slot (the verb is predicted there)
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "cos_min": COS_MIN, "det_min": DET_MIN, "h113_max": H113_MAX, "v496_113": V496_113}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE, SHE = L._single(" they"), L._single(" he"), L._single(" she"); ARE, IS, WERE, WAS = L._single(" are"), L._single(" is"), L._single(" were"), L._single(" was")
    RD = dict(READERS)
    MODES = {"11.3": ({11: [3]}, (0,)), "UNITS": ({}, (0,))}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return torch.stack([(z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS]), z[:, ARE] - z[:, IS], z[:, WERE] - z[:, WAS]], 1), z   # the agreement margin, matching the module reference
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
                    units = torch.tensor(UNITS, device="cuda"); hdn[idx[:, None], fin[:, None], units[None, :]] = hdn[swap[:, None], fin[:, None], units[None, :]].clone()   # the detector units at the answer position
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
            for m in ed: _, z = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m); ed[m].append(z)
        nat_m, nat_z = torch.cat(nat_m), torch.cat(nat_z); ed = {m: torch.cat(v) for m, v in ed.items()}
        FAM = _family(); P = torch.tensor(sorted(FAM)); out = {"pairs": len(items), "family_size": len(FAM)}
        E = L.ENCODING; SV = [L._single(t) for t in SING_V]; PV = [L._single(t) for t in PLUR_V]; VT = torch.tensor(SV + PV); lem = {}; vv = {}
        for m, z in ed.items():
            signed = (z - nat_z)[1::2]; dz = (z - nat_z).abs()
            by_lemma = {}
            for tid, w in FAM.items(): by_lemma.setdefault(w, []).append(float(signed[:, tid].mean()))
            lem[m] = {w: sum(v) / len(v) for w, v in by_lemma.items()}; vv[m] = signed[:, VT].mean(0)
            verb_signed = {E.decode([t]).strip(): float(signed[:, t].mean()) for t in SV + PV}
            out[m] = {"verb_signed_change": verb_signed, "verb_mean_abs_change": float(dz[:, VT].mean()), "verb_signs_ok": all(verb_signed[t.strip()] < 0 for t in SING_V) and all(verb_signed[t.strip()] > 0 for t in PLUR_V),
                      "lemma_signed_change": dict(sorted(lem[m].items(), key=lambda kv: kv[1]))}
        cosv = lambda a_, b_: float(vv[a_] @ vv[b_] / (vv[a_].norm() * vv[b_].norm()))
        ref = torch.tensor([V496_113[E.decode([t]).strip()] for t in SV + PV]); out["verb_cos_detector_vs_11_3_distant"] = float(vv["UNITS"] @ ref / (vv["UNITS"].norm() * ref.norm()))
        out["native_margins"] = nat_m[:, 0].tolist()
        return out, nat_m[:, 0]
    verb_items = [it for it in verb_items if len(it[0]) - 1 - it[2] == 0]   # adjacent verbs: the answer position is the noun
    verb, nat_t = study(verb_items, 32); text = verb; panel = {"skipped": "distant verb slot only", "verb_slot_rows": len(verb_items)}
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in verb_items[:32]] + [list(s) for _, s, _ in verb_items[:32]]; seqs = [q + [0] * (max(len(t) for t in seqs) - len(q)) for q in seqs]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        fin_ = torch.tensor([len(p) - 1 for p, _, _ in verb_items[:32]] * 2, device="cuda"); z = hook["z"].float()[torch.arange(64, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = ((z[:, ARE] - z[:, IS]) + (z[:, WERE] - z[:, WAS])).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": panel}
    print(json.dumps(report, indent=1))
    report["verb_slot"] = {k: v for k, v in verb.items() if k != "native_margins"}
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_detector_same_vector_on_the_direct_path": verb["verb_cos_detector_vs_11_3_distant"] >= COS_MIN, "pred_c_detector_signs_adjacent": verb["UNITS"]["verb_signs_ok"],
                   "pred_d_detector_change_adjacent": verb["UNITS"]["verb_mean_abs_change"] >= DET_MIN, "pred_e_11_3_small_on_adjacent": verb["11.3"]["verb_mean_abs_change"] <= H113_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "detector_verb_vector_adjacent_result_v498", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
