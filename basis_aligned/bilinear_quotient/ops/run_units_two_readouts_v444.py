#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_units_move_verbs_like_the_copier pred_c_units_move_pronouns_like_the_readers pred_d_readers_move_verbs_less_at_verb_slot pred_e_units_verb_signs
"""Do MLP 8's three number units feed both readouts (v444)? v441-v443: the copier 4.5 (block 4) feeds the pronoun readout (readers' lemma vector, cos 0.97)
and the verb-agreement readout (verb signs at the verb slot). The chain's destination, MLP 8's units 829 / 953 / 1030 at the noun (v423: 0.12 of the pronoun
margin), sits after the copier. Here their replace-edit scored at both natural slots — the pronoun slot (lemma vector vs the readers') and the verb slot
(the eight agreement verbs vs the copier's) — plus the five readers at the verb slot as the pronoun-specific control.
PREDICTIONS (scored as written; failures preserved; priors from v423 / v441 / v443)
    pred_a_baseline_replays                   the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_units_move_verbs_like_the_copier   at the verb slot, cosine over the eight agreement verbs between the units' and the copier's signed changes >= 0.80. Prior: unsure
    pred_c_units_move_pronouns_like_the_readers  at the pronoun slot, cosine over lemmas between the units' and the readers' signed changes >= 0.90
    pred_d_readers_move_verbs_less_at_verb_slot  at the verb slot, the readers' mean |change| over the eight verbs is <= 0.50 x the copier's
    pred_e_units_verb_signs                   at the verb slot, the units' swap makes is / was / has / does fall and are / were / have / do rise
PRICE (registered maximum): 4 text batches x 4 passes x 2 slots = 32 forwards; 0 backwards; 0 fits. Bar <= 33.
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
OUT = ROOT / "circuits/followups/units_two_readouts_v444_result.json"
CANDIDATE_ID = "chain.units_two_readouts_v444"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, VERB_COS_MIN, LEMMA_COS_MIN, READER_RATIO_MAX = 1e-3, 0.80, 0.90, 0.50
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
FORWARDS_MAX = 33
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_units_move_verbs_like_the_copier": "cos >= 0.80", "pred_c_units_move_pronouns_like_the_readers": "cos >= 0.90", "pred_d_readers_move_verbs_less_at_verb_slot": "<= 0.50 x", "pred_e_units_verb_signs": "signs x 8"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "verb_cos_min": VERB_COS_MIN, "lemma_cos_min": LEMMA_COS_MIN, "reader_ratio_max": READER_RATIO_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE, SHE = L._single(" they"), L._single(" he"), L._single(" she")
    RD = dict(READERS)
    MODES = {"FIVE_N": ({l: hs for l, hs in FIVE.items()}, (0,)), "COPIER": ({4: [5]}, (0,)), "UNITS": ({}, (0,))}; forwards = 0
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
        nat_m, nat_z, ed = [], [], {"FIVE_N": [], "COPIER": [], "UNITS": []}
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
        keys = sorted(lem["FIVE_N"]); cosl = lambda a_, b_: float(torch.tensor([lem[a_][k] for k in keys]) @ torch.tensor([lem[b_][k] for k in keys]) / (torch.tensor([lem[a_][k] for k in keys]).norm() * torch.tensor([lem[b_][k] for k in keys]).norm()))
        out["lemma_cos_units_vs_readers"] = cosl("UNITS", "FIVE_N"); out["lemma_cos_copier_vs_readers"] = cosl("COPIER", "FIVE_N")
        out["verb_cos_units_vs_copier"] = float(vv["UNITS"] @ vv["COPIER"] / (vv["UNITS"].norm() * vv["COPIER"].norm())); out["reader_over_copier_verb_change"] = out["FIVE_N"]["verb_mean_abs_change"] / out["COPIER"]["verb_mean_abs_change"]
        out["native_margins"] = nat_m[:, 0].tolist()
        return out, nat_m[:, 0]
    text, nat_t = study(text_items, 32); verb, _ = study(verb_items, 32); panel = {"skipped": "text only (price)", "verb_slot_rows": len(verb_items)}
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": panel}
    print(json.dumps(report, indent=1))
    report["verb_slot"] = {k: v for k, v in verb.items() if k != "native_margins"}
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_units_move_verbs_like_the_copier": verb["verb_cos_units_vs_copier"] >= VERB_COS_MIN, "pred_c_units_move_pronouns_like_the_readers": text["lemma_cos_units_vs_readers"] >= LEMMA_COS_MIN,
                   "pred_d_readers_move_verbs_less_at_verb_slot": verb["reader_over_copier_verb_change"] <= READER_RATIO_MAX, "pred_e_units_verb_signs": verb["UNITS"]["verb_signs_ok"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "units_two_readouts_result_v444", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
