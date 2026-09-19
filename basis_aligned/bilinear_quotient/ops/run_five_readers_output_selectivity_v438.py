#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_they_she_moves_like_they_he pred_c_he_she_barely_moves pred_d_distribution_moves_little pred_e_holds_on_panel
"""Output selectivity of the five-head value swap (v438). v433: swapping the five readers' values at the noun closes 0.483 of the they - he margin gap.
Does the swap move ONLY number? Measured on the same edit: the they - she gap (should close like they - he), the he - she gap (gender: should not move —
the swap carries the pronoun-class axis VC1 'not he / she' as well, v404, but not gender within the singulars), and the whole next-token distribution
(KL of the edited to the native row, against the KL between the two rows of the pair).
PREDICTIONS (scored as written; failures preserved; priors from v404 / v433)
    pred_a_baseline_replays          the unedited manual forward replays the model's they - he margins within 1e-3
    pred_b_they_she_moves_like_they_he  the they - she gap closes within 0.10 of the they - he closure on text
    pred_c_he_she_barely_moves       the mean |change of the he - she margin| is <= 0.15 x the mean |change of the they - he margin| on text
    pred_d_distribution_moves_little KL(edited || native), averaged over rows, is <= 0.50 x KL(partner || native) on text (the swap moves less than half of what the noun change moves). Prior: unsure
    pred_e_holds_on_panel            pred_c holds on the panel
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
OUT = ROOT / "circuits/followups/five_readers_output_selectivity_v438_result.json"
CANDIDATE_ID = "both_ends.five_readers_output_selectivity_v438"
N_HEAD = 9
READERS = ((9, 6), (12, 4), (15, 1)); FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, SHE_BAND, GENDER_MAX, KL_MAX = 1e-3, 0.10, 0.15, 0.50
FORWARDS_MAX = 15
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_they_she_moves_like_they_he": "within 0.10", "pred_c_he_she_barely_moves": "<= 0.15 x", "pred_d_distribution_moves_little": "<= 0.50 x", "pred_e_holds_on_panel": "<= 0.15 x"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    partner_ = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    panel_items = [(rows[i].ids, rows[partner_[(rows[i].construction, rows[i].group, False)]].ids, next(k for k, t in enumerate(rows[i].ids) if t in nouns)) for i, r_ in enumerate(rows) if r_.present]
    text_items = v406.natural_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "panel_pairs": len(panel_items), "text_pairs": len(text_items), "readers": [f"{l}.{h}" for l, h in READERS], "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "she_band": SHE_BAND, "gender_max": GENDER_MAX, "kl_max": KL_MAX}}
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
        return torch.stack([z[:, THEY] - z[:, HE], z[:, THEY] - z[:, SHE], z[:, HE] - z[:, SHE]], 1), torch.log_softmax(z, -1)
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
        nat_m, nat_lp, ed_m, ed_lp = torch.cat(nat_m), torch.cat(nat_lp), torch.cat(ed_m), torch.cat(ed_lp)
        gap = nat_m[0::2] - nat_m[1::2]                                        # (pairs, 3): they-he, they-she, he-she gaps
        closed = lambda j: float(((nat_m[0::2, j] - ed_m[0::2, j]) + (ed_m[1::2, j] - nat_m[1::2, j])).sum() / (2 * gap[:, j].sum()))
        change = (ed_m - nat_m).abs().mean(0)                                  # mean |change| per margin
        kl = lambda a_, b_: (a_.exp() * (a_ - b_)).sum(-1)
        kl_edit = float(kl(ed_lp, nat_lp).mean()); part = torch.arange(nat_lp.shape[0]); part[0::2], part[1::2] = torch.arange(1, nat_lp.shape[0], 2), torch.arange(0, nat_lp.shape[0], 2)
        kl_partner = float(kl(nat_lp[part], nat_lp).mean())
        return {"pairs": len(items), "gap_mean_they_he": float(gap[:, 0].mean()), "closed": {"they_he": closed(0), "they_she": closed(1)}, "mean_abs_change": {"they_he": float(change[0]), "they_she": float(change[1]), "he_she": float(change[2])},
                "gender_over_number": float(change[2] / change[0]), "kl_edit_vs_native": kl_edit, "kl_partner_vs_native": kl_partner, "kl_ratio": kl_edit / kl_partner, "native_margins": nat_m[:, 0].tolist()}, nat_m[:, 0]
    text, nat_t = study(text_items, 32); panel, nat_p = study(panel_items, 16)
    # replay check against the module's own forward on the first text batch
    with torch.no_grad():
        seqs = [list(p) for p, _, _ in text_items[:32]] + [list(s) for _, s, _ in text_items[:32]]; tokens = torch.tensor(seqs, device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[:, -1]; z = 30 * torch.tanh(z / 30); ref = (z[:, THEY] - z[:, HE]).cpu()
        mine = torch.cat([nat_t[0:64:2], nat_t[1:64:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}, "panel": {k: v for k, v in panel.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_they_she_moves_like_they_he": abs(text["closed"]["they_she"] - text["closed"]["they_he"]) <= SHE_BAND, "pred_c_he_she_barely_moves": text["gender_over_number"] <= GENDER_MAX,
                   "pred_d_distribution_moves_little": text["kl_ratio"] <= KL_MAX, "pred_e_holds_on_panel": panel["gender_over_number"] <= GENDER_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "five_readers_output_selectivity_result_v438", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
