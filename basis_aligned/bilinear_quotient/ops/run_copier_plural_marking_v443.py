#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_copier_non_family_is_plural_marked pred_c_plural_marked_rise pred_d_readers_non_family_small pred_e_copier_verbs_at_verb_slots
"""The copier's non-pronoun output on text is plural marking (v443). v442: on the panel (a verb slot) the copier's non-pronoun movers are agreement verbs;
on text (a pronoun slot) they are theirs / respectively / plural nouns. Here the text side named: the share of the copier swap's non-family top-10 slots
that are PLURAL-MARKED tokens (a token whose stripped form ends in -s and whose stem is itself a token, plus theirs / respectively / both / all / many /
several / few), their sign (they should rise when the singular row gets the plural row's values), and, to tie the two frames, the copier's verb movers
measured at the natural rows' own VERB slot (second_offset) instead of the pronoun slot.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_replays                 the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_copier_non_family_is_plural_marked  plural-marked tokens are >= 0.40 of the copier swap's non-family top-10 slots on text
    pred_c_plural_marked_rise               the mean signed change over the plural-marked movers is > 0 on the singular rows (text)
    pred_d_readers_non_family_small         the readers' swap has <= 0.12 non-family share of its top-10 slots on text (replay of 0.087)
    pred_e_copier_verbs_at_verb_slots       at the natural rows' verb slot, the copier swap's singular verb forms (is / was / has / does) all fall and plural forms (are / were / have / do) all rise. Prior: unsure
PRICE (registered maximum): 4 text batches x 3 passes x 2 slots = 24 forwards (panel skipped); 0 backwards; 0 fits. Bar <= 25.
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
OUT = ROOT / "circuits/followups/copier_plural_marking_v443_result.json"
CANDIDATE_ID = "both_ends.copier_plural_marking_v443"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, PLURAL_SHARE_MIN, READER_NONFAM_MAX = 1e-3, 0.40, 0.12
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
FORWARDS_MAX = 25
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_copier_non_family_is_plural_marked": ">= 0.40", "pred_c_plural_marked_rise": "> 0", "pred_d_readers_non_family_small": "<= 0.12", "pred_e_copier_verbs_at_verb_slots": "signs x 8"}


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
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "plural_share_min": PLURAL_SHARE_MIN, "reader_nonfam_max": READER_NONFAM_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE, SHE = L._single(" they"), L._single(" he"), L._single(" she")
    RD = dict(READERS)
    MODES = {"FIVE_N": ({l: hs for l, hs in FIVE.items()}, (0,)), "COPIER": ({4: [5]}, (0,))}; forwards = 0
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
        nat_m, nat_z, ed = [], [], {"FIVE_N": [], "COPIER": []}
        for s0 in range(0, len(seqs), 2 * batch):
            m_, z = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native"); nat_m.append(m_); nat_z.append(z)
            for m in ed: _, z = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m); ed[m].append(z)
        nat_m, nat_z = torch.cat(nat_m), torch.cat(nat_z); ed = {m: torch.cat(v) for m, v in ed.items()}
        FAM = _family(); P = torch.tensor(sorted(FAM)); out = {"pairs": len(items), "family_size": len(FAM)}
        E = L.ENCODING; from collections import Counter
        SV = [L._single(t) for t in SING_V]; PV = [L._single(t) for t in PLUR_V]
        EXTRA = {"theirs", "respectively", "both", "all", "many", "several", "few", "these", "those"}
        def plural_marked(tok):
            t = E.decode([int(tok)]).strip()
            if t.lower() in EXTRA: return True
            return len(t) > 3 and t.endswith("s") and t[:-1].isalpha() and len(E.encode(" " + t[:-1])) == 1
        for m, z in ed.items():
            dz = (z - nat_z).abs(); top = dz.topk(10, dim=1).indices; signed = (z - nat_z)[1::2]
            non_fam = [int(t) for row in top for t in row if int(t) not in FAM]; cnt = Counter(E.decode([t]) for t in non_fam)
            pm = [t for t in non_fam if plural_marked(t)]; pm_share = len(pm) / max(len(non_fam), 1); pm_signed = float(sum(float(signed[:, t].mean()) for t in set(pm)) / max(len(set(pm)), 1))
            verb_signed = {E.decode([t]).strip(): float(signed[:, t].mean()) for t in SV + PV}
            out[m] = {"non_family_share_of_top10": len(non_fam) / (10 * top.shape[0]), "most_moved_non_family": cnt.most_common(20), "plural_marked_share_of_non_family": pm_share, "plural_marked_mean_signed": pm_signed,
                      "verb_signed_change": verb_signed, "verb_signs_ok": all(verb_signed[t.strip()] < 0 for t in SING_V) and all(verb_signed[t.strip()] > 0 for t in PLUR_V)}
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_copier_non_family_is_plural_marked": text["COPIER"]["plural_marked_share_of_non_family"] >= PLURAL_SHARE_MIN, "pred_c_plural_marked_rise": text["COPIER"]["plural_marked_mean_signed"] > 0,
                   "pred_d_readers_non_family_small": text["FIVE_N"]["non_family_share_of_top10"] <= READER_NONFAM_MAX, "pred_e_copier_verbs_at_verb_slots": verb["COPIER"]["verb_signs_ok"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "copier_plural_marking_result_v443", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
