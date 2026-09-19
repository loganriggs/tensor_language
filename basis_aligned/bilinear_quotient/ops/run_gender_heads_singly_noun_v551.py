#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_9_6_is_the_largest_reader pred_c_gender_tilted_readers_carry pred_d_10_5_small pred_e_reader_singles_add
"""The gender heads singly (v551). v549 / v550: on 61 natural gender pairs the five pronoun readers' values at the noun close 0.600 of the he - she gap and
the two self-copies 8.1 + 6.1 0.310; jointly 0.828 of 0.901. Here each of the seven heads' values at the noun swapped alone. Priors: 9.6 is the atlas's
gender reader (the 18 Sep path runs through it); 12.4 tilts masculine and 10.1 feminine on the number line (v438-v440); 10.5 is a person / number head;
15.1 unsure; 8.1 (97% token-only self-copy) and 6.1 (90%) unsure between them.
PREDICTIONS (scored as written; failures preserved; priors from v438-v440, v549-v550, 18 Sep folds)
    pred_a_baseline_replays            the unedited manual forward replays the model's he - she margins within 1e-3
    pred_b_9_6_is_the_largest_reader   9.6 alone closes more than any other reader
    pred_c_gender_tilted_readers_carry 12.4 alone >= 0.08 and 10.1 alone >= 0.05
    pred_d_10_5_small                  10.5 alone closes <= 0.05
    pred_e_reader_singles_add          the five reader singles sum to 0.600 +- 0.10 (parallel exits at one position; number's readers added)
PRICE (registered maximum): 2 batches x 8 passes = 16 forwards (61 pairs); 0 backwards; 0 fits. Bar <= 17.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import pronoun_gender_dod_natural_rows as gm
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gender_heads_singly_noun_v551_result.json"
CANDIDATE_ID = "gender.heads_singly_noun_v551"
ROWS = ROOT / "circuits/followups/pronoun_gender_dod_natural_rows_v73.json"
N_HEAD = 9
FIVE = {9: [6], 10: [1, 5], 12: [4], 15: [1]}
REPLAY_TOL, M124_MIN, F101_MIN, H105_MAX, FIVE_V549, BAND = 1e-3, 0.08, 0.05, 0.05, 0.600, 0.10
FORWARDS_MAX = 17
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_9_6_is_the_largest_reader": "9.6 first", "pred_c_gender_tilted_readers_carry": "12.4 >= 0.08, 10.1 >= 0.05", "pred_d_10_5_small": "<= 0.05", "pred_e_reader_singles_add": "0.600 +- 0.10"}


def gender_pairs():
    """Aligned pairs from v73's natural rows: the gendered noun swapped in place for its partner (same surface form: ' king' <-> ' queen', 'King' <-> 'Queen')."""
    E = L.ENCODING; partner_word = {}
    for m, f in gm.PAIRS: partner_word[m], partner_word[f] = f, m
    items = []
    for r_ in json.loads(ROWS.read_text())["rows"]:
        c = r_["cue_offset"]; tok = r_["ids"][c]; t = E.decode([tok]); w = t.strip(); lead = " " if t.startswith(" ") else ""
        base = w.lower()
        if base not in partner_word: continue
        pw = partner_word[base]; pw = pw.capitalize() if w[0].isupper() else pw
        alt = E.encode(lead + pw)
        if len(alt) != 1: continue
        sw = list(r_["ids"]); sw[c] = alt[0]
        male, female = (r_["ids"], sw) if r_["gender"] == "male" else (sw, r_["ids"])
        items.append((male, female, c))
    return items


def main() -> None:
    text_items = gender_pairs()
    plan = {"candidate_id": CANDIDATE_ID, "text_pairs": len(text_items), "rows": str(ROWS.name), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"replay_tol": REPLAY_TOL, "m124_min": M124_MIN, "f101_min": F101_MIN, "h105_max": H105_MAX, "five_v549": FIVE_V549, "band": BAND}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    from jacclust.tt_model import apply_rotary_emb  # noqa
    D = model.config.n_embd; hd = D // N_HEAD; HE, SHE = L._single(" he"), L._single(" she")
    LOW_ALL = {l: list(range(N_HEAD)) for l in range(0, 9)}
    HIGH_ALL = {l: list(range(N_HEAD)) for l in range(9, 18)}
    HEADS = ((9, 6), (12, 4), (10, 1), (10, 5), (15, 1), (8, 1), (6, 1))
    MODES = {f"{l}.{h}": ({}, {l: [h]}, False) for l, h in HEADS}; forwards = 0
    def logits_margin(x):
        z = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0).float()
        return z[:, HE] - z[:, SHE]
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
                if mode != "native" and (l in MODES[mode][0] or l in MODES[mode][1] or MODES[mode][2]):
                    hs2, hs0, rest = MODES[mode][0].get(l, []), MODES[mode][1].get(l, []), MODES[mode][2]
                    q = attn.c_q(xin).view(B_, -1, N_HEAD, hd); k = attn.c_k(xin).view(B_, -1, N_HEAD, hd); q2 = attn.c_q2(xin).view(B_, -1, N_HEAD, hd); k2 = attn.c_k2(xin).view(B_, -1, N_HEAD, hd)
                    v = attn.c_v(xin).view(B_, -1, N_HEAD, hd)
                    if v1_ is None: v1n = v
                    else: v1n = v1_
                    v = (1 - attn.lamb) * v + attn.lamb * v1n.view_as(v)
                    cos, sin = attn.rotary(q)
                    q, k = apply_rotary_emb(F.rms_norm(q, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k, (hd,)), cos, sin)
                    q2, k2 = apply_rotary_emb(F.rms_norm(q2, (hd,)), cos, sin), apply_rotary_emb(F.rms_norm(k2, (hd,)), cos, sin)
                    v = v.clone()
                    for h in hs2:
                        for o in (0, 1): v[idx, pp + o, h] = v[swap, pp + o, h].clone()               # readers: both sites
                    for h in hs0: v[idx, pp, h] = v[swap, pp, h].clone()                              # low exits: the noun only
                    if rest:
                        ar = torch.arange(tokens.shape[1], device="cuda")[None, :]; sel = (ar >= pp[:, None] + 2)[:, :, None, None]
                        v = torch.where(sel, v[swap], v)                                              # every head's value from two after the noun onward
                    y = attn.squared_attention(q, k, v, q2, k2); y = y.transpose(1, 2).contiguous().view_as(xin); attention = attn.c_proj(y); v1_ = v1n
                else:
                    attention, v1_ = attn(xin, v1_)
                x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
            return logits_margin(x[idx, fin]).cpu()
    def study(items, batch):
        seqs, pos = [], []
        for p, s, c in items: seqs += [list(p), list(s)]; pos += [c, c]
        res = {m: [] for m in ("native", *MODES)}
        for s0 in range(0, len(seqs), 2 * batch):
            for m in res: res[m].append(run(seqs[s0:s0 + 2 * batch], pos[s0:s0 + 2 * batch], m))
        res = {m: torch.cat(v) for m, v in res.items()}; nat = res["native"]; gap = nat[0::2] - nat[1::2]
        closed = lambda e: float(((nat[0::2] - e[0::2]) + (e[1::2] - nat[1::2])).sum() / (2 * gap.sum()))
        sign = float(max((gap > 0).float().mean(), (gap < 0).float().mean()))
        return {"pairs": len(items), "gap_mean": float(gap.mean()), "sign_constancy": sign, "closed": {m: closed(res[m]) for m in MODES}, "native_margins": nat.tolist()}, nat
    text, nat_t = study(text_items, 32)
    with torch.no_grad():   # replay check against the module's own forward on the first text batch (rows padded to one length)
        n = min(32, len(text_items)); seqs = [list(p) for p, _, _ in text_items[:n]] + [list(s) for _, s, _ in text_items[:n]]
        T_ = max(len(q) for q in seqs); tokens = torch.tensor([q + [0] * (T_ - len(q)) for q in seqs], device="cuda"); fin_ = torch.tensor([len(q) - 1 for q in seqs], device="cuda")
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove()
        z = hook["z"].float()[torch.arange(2 * n, device="cuda"), fin_]; z = 30 * torch.tanh(z / 30); ref = (z[:, HE] - z[:, SHE]).cpu()
        mine = torch.cat([nat_t[0:2 * n:2], nat_t[1:2 * n:2]]); replay = float((mine - ref).abs().max())
    report = {"replay_max_abs": replay, "text": {k: v for k, v in text.items() if k != "native_margins"}}
    print(json.dumps(report, indent=1))
    c = text["closed"]
    readers = {k: c[k] for k in ("9.6", "12.4", "10.1", "10.5", "15.1")}
    predictions = {"pred_a_baseline_replays": replay <= REPLAY_TOL, "pred_b_9_6_is_the_largest_reader": max(readers, key=readers.get) == "9.6", "pred_c_gender_tilted_readers_carry": c["12.4"] >= M124_MIN and c["10.1"] >= F101_MIN,
                   "pred_d_10_5_small": abs(c["10.5"]) <= H105_MAX, "pred_e_reader_singles_add": abs(sum(readers.values()) - FIVE_V549) <= BAND}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gender_heads_singly_noun_result_v551", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
