#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_pronouns_lead_the_logit_changes pred_c_pronoun_class_share_of_change pred_d_top_changed_are_pronouns pred_e_holds_on_panel
"""What else the five readers carry (v439). v438: the five-head value swap at the noun moves the whole next-token distribution by a quarter of what the noun
change does (KL ratio 0.24), and the gender margin a sixth as much as the number margin. Here the logit changes themselves: which tokens move most under
the swap, the share of the summed |logit change| that falls on the 11 pronoun-class tokens (they / we / them / us / he / she / it / him / her / I / you)
against their share of the vocabulary (11 / 50k), and whether the ten most-moved tokens per row are pronouns or plural / singular verb forms.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_baseline_replays              the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_pronouns_lead_the_logit_changes  the mean |logit change| over the 11 pronoun-class tokens is >= 5 x the mean over all tokens on text
    pred_c_pronoun_class_share_of_change  the pronoun class carries >= 0.05 of the summed |logit change| on text (its vocabulary share is 0.0002). Prior: unsure
    pred_d_top_changed_are_pronouns      among the ten most-moved tokens per row, pooled, pronoun-class tokens are >= 0.30 on text. Prior: unsure
    pred_e_holds_on_panel                pred_b holds on the panel
PRICE (registered maximum): (4 text + 3 panel batches) x 2 passes = 14 forwards; 0 backwards; 0 fits. Bar <= 15.
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
OUT = ROOT / "circuits/followups/five_readers_logit_changes_v439_result.json"
CANDIDATE_ID = "both_ends.five_readers_logit_changes_v439"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, LEAD_MIN, SHARE_MIN, TOP_MIN = 1e-3, 5.0, 0.05, 0.30
PRONOUNS = (" they", " we", " them", " us", " he", " she", " it", " him", " her", " I", " you")
FORWARDS_MAX = 15
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_pronouns_lead_the_logit_changes": ">= 5 x", "pred_c_pronoun_class_share_of_change": ">= 0.05", "pred_d_top_changed_are_pronouns": ">= 0.30", "pred_e_holds_on_panel": ">= 5 x"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "lead_min": LEAD_MIN, "share_min": SHARE_MIN, "top_min": TOP_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; THEY, HE, SHE = L._single(" they"), L._single(" he"), L._single(" she")
    RD = dict(READERS)
    MODES = {"FIVE_N": ({l: hs for l, hs in FIVE.items()}, (0,))}; forwards = 0
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
        nat_m, nat_lp, ed_m, ed_lp = [], [], [], []
        for s0 in range(0, len(seqs), 2 * batch):
            m_, lp = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "native"); nat_m.append(m_); nat_lp.append(lp)
            m_, lp = run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], "FIVE_N"); ed_m.append(m_); ed_lp.append(lp)
        nat_m, nat_z, ed_m, ed_z = torch.cat(nat_m), torch.cat(nat_lp), torch.cat(ed_m), torch.cat(ed_lp)
        dz = (ed_z - nat_z).abs(); P = torch.tensor([L._single(t) for t in PRONOUNS])
        lead = float(dz[:, P].mean() / dz.mean()); share = float(dz[:, P].sum() / dz.sum())
        top = dz.topk(10, dim=1).indices; is_p = torch.isin(top, P); top_share = float(is_p.float().mean())
        E = L.ENCODING; from collections import Counter; cnt = Counter(E.decode([int(t)]) for row in top for t in row); most = cnt.most_common(15)
        return {"pairs": len(items), "pronoun_lead": lead, "pronoun_share": share, "top10_pronoun_share": top_share, "most_moved_tokens": most, "mean_abs_dz": float(dz.mean()), "native_margins": nat_m[:, 0].tolist()}, nat_m[:, 0]
    text, nat_t = study(text_items, 32); panel, nat_p = study(panel_items, 16)
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_pronouns_lead_the_logit_changes": text["pronoun_lead"] >= LEAD_MIN, "pred_c_pronoun_class_share_of_change": text["pronoun_share"] >= SHARE_MIN,
                   "pred_d_top_changed_are_pronouns": text["top10_pronoun_share"] >= TOP_MIN, "pred_e_holds_on_panel": panel["pronoun_lead"] >= LEAD_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "five_readers_logit_changes_result_v439", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
