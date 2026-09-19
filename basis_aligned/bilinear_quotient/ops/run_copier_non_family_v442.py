#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_non_family_movers_are_agreement_verbs pred_c_verbs_split_by_number pred_d_readers_move_verbs_less pred_e_holds_on_panel
"""What else the copier's copy feeds (v442). v441: the copier 4.5's value swap writes the readers' lemma vector (cos 0.97) but pronoun forms fill only 0.42
of its ten most-moved logits (readers' swap 0.91). Here the NON-family movers under each swap: the most-moved non-pronoun tokens pooled over rows, the share
of them that are number-marked verb forms (is / are / was / were / has / have / does / do / 's and the -s / bare third-person pairs in the rows' own verbs),
and their sign (singular forms should fall and plural forms rise when the singular row gets the plural row's values).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_replays                   the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_non_family_movers_are_agreement_verbs  under the copier swap, number-marked verb forms are >= 0.30 of the non-family slots among the ten most-moved per row (text)
    pred_c_verbs_split_by_number              under the copier swap, the singular verb forms (is / was / has / does) all fall and the plural forms (are / were / have / do) all rise on the singular rows (text)
    pred_d_readers_move_verbs_less            the readers' swap moves the number-marked verb forms by <= 0.50 x what the copier swap moves them (mean |change|, text)
    pred_e_holds_on_panel                     pred_c holds on the panel
PRICE (registered maximum): (4 text + 3 panel batches) x 3 passes = 21 forwards; 0 backwards; 0 fits. Bar <= 22.
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
OUT = ROOT / "circuits/followups/copier_non_family_v442_result.json"
CANDIDATE_ID = "both_ends.copier_non_family_v442"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, VERB_SHARE_MIN, READER_RATIO_MAX = 1e-3, 0.30, 0.50
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
FORWARDS_MAX = 22
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_non_family_movers_are_agreement_verbs": ">= 0.30", "pred_c_verbs_split_by_number": "signs x 8", "pred_d_readers_move_verbs_less": "<= 0.50 x", "pred_e_holds_on_panel": "signs x 8"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "verb_share_min": VERB_SHARE_MIN, "reader_ratio_max": READER_RATIO_MAX}}
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
        SV = [L._single(t) for t in SING_V]; PV = [L._single(t) for t in PLUR_V]; VERBS = torch.tensor(SV + PV)
        def marked(tok):
            t = E.decode([int(tok)]).strip()
            return t in {"is", "are", "was", "were", "has", "have", "does", "do", "'s"} or (t.endswith("s") and t[:-1].isalpha() and t[:-1].islower())
        for m, z in ed.items():
            dz = (z - nat_z).abs(); top = dz.topk(10, dim=1).indices; signed = (z - nat_z)[1::2]
            non_fam = [int(t) for row in top for t in row if int(t) not in FAM]; cnt = Counter(E.decode([t]) for t in non_fam)
            verb_share = sum(1 for t in non_fam if marked(t)) / max(len(non_fam), 1)
            verb_signed = {E.decode([t]).strip(): float(signed[:, t].mean()) for t in SV + PV}
            out[m] = {"non_family_share_of_top10": len(non_fam) / (10 * top.shape[0]), "most_moved_non_family": cnt.most_common(20), "marked_verb_share_of_non_family": verb_share, "verb_signed_change": verb_signed,
                      "verb_mean_abs_change": float(dz[:, VERBS].mean()), "verb_signs_ok": all(verb_signed[t.strip()] < 0 for t in SING_V) and all(verb_signed[t.strip()] > 0 for t in PLUR_V)}
        out["reader_over_copier_verb_change"] = out["FIVE_N"]["verb_mean_abs_change"] / out["COPIER"]["verb_mean_abs_change"]; out["native_margins"] = nat_m[:, 0].tolist()
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
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_non_family_movers_are_agreement_verbs": text["COPIER"]["marked_verb_share_of_non_family"] >= VERB_SHARE_MIN, "pred_c_verbs_split_by_number": text["COPIER"]["verb_signs_ok"],
                   "pred_d_readers_move_verbs_less": text["reader_over_copier_verb_change"] <= READER_RATIO_MAX, "pred_e_holds_on_panel": panel["COPIER"]["verb_signs_ok"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "copier_non_family_result_v442", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
