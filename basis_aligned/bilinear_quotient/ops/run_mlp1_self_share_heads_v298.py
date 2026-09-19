#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_head_closure pred_b_self_share_is_head_concentrated pred_c_alpha_attribution_matches_share pred_d_block0_carries_most pred_e_same_heads_on_text
"""MLP 1: which heads set the self-share? (v298). v291 / v297: the gain alpha on MLP 1's token lookup equals the token's own-key share of attention 0 + 1
(r 0.65 on text). The share averaged over 18 heads hides whether a few heads keep the token's own value (a "self" head) while others read the context.
Per head h of blocks 0 and 1, at the target of phrase-A length-8 rows (7 classes x 32 targets) and at the 2,944 natural positions: own-key share s_h.
Attribution: for each head, replace its pattern row at the target by its own-key entry only (delete the head's context reading; everything else
native) and measure the change in alpha; the head's attribution a_h = alpha(self-only h) - alpha(native). Sum over heads vs the total alpha(all self-only) -
alpha(native) tests additivity (closure).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_head_closure                    sum_h a_h is within 0.25 (absolute, on the alpha scale) of the all-heads-self-only change, median over rows
    pred_b_self_share_is_head_concentrated the top 4 of 18 heads carry >= 0.50 of the summed (1 - s_h) context reading at the target (filler rows)
    pred_c_alpha_attribution_matches_share  Pearson r over heads between a_h (median over rows) and (1 - s_h) (median) >= 0.60
    pred_d_block0_carries_most             block-0 heads' summed attribution >= 0.50 of the total (block 0 sees the raw tokens; block 1 mixes)
    pred_e_same_heads_on_text              the top-4 heads by (1 - s_h) on filler rows and on natural text share >= 2 heads
Run 1 (02:34 UTC) is VOID: the hooked attention returned [B, T, H, D] where the module expects [B, H, T, D], scrambling every pass (native alpha 0.45 vs v291 0.335; all-self-only 1.78). Fixed; run 2 is the receipt.
PRICE (registered maximum): 1 table batch + 1 native + 18 head-edited + 1 all-edited filler passes + 2 natural passes = 23 forwards; 0 backwards; 0 fits. Bar <= 26.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_self_share_heads_v298_result.json"
CANDIDATE_ID = "mlp1.token_table.self_share_heads_v298"
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
PHRASE = (",", " and", " of", " the", " very"); K, N = 8, 32
CLOSURE_TOL, TOP4_MIN, R_MIN, BLOCK0_MIN, OVERLAP_MIN = 0.25, 0.50, 0.60, 0.50, 2
FORWARDS_MAX = 26
PREDICTIONS = {"pred_a_head_closure": "<= 0.25", "pred_b_self_share_is_head_concentrated": ">= 0.50", "pred_c_alpha_attribution_matches_share": "r >= 0.60", "pred_d_block0_carries_most": ">= 0.50", "pred_e_same_heads_on_text": ">= 2 of 4"}
HEADS = [(l, h) for l in (0, 1) for h in range(9)]


def run(backend, tokens, pos, self_only=()):
    """blocks 0-1 with the bilinear pattern hooked: returns mlp1 write and x1 at `pos` (per row; None = all positions >= 1) and per-head own-key shares.
    `self_only`: heads (l, h) whose pattern row at the query positions is replaced by its own-key entry alone."""
    torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h; store = {}
    def wrap(l):
        orig = blocks[l].attn.squared_attention
        def f(q, k, v, q2, k2):
            B, T, H, D = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / D) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / D)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0)
            own = torch.diagonal(pat, dim1=-2, dim2=-1).abs(); store[l] = (own / pat.abs().sum(-1).clamp_min(1e-9)).float().cpu()     # [B, H, T]
            for (ll, h) in self_only:
                if ll == l: pat[:, h] = torch.diag_embed(torch.diagonal(pat[:, h], dim1=-2, dim2=-1))
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)   # same layout as squared_attention: [B, H, T, D]
        return f
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
        for l in (0, 1):
            block = blocks[l]; live = block.lambdas[0] * x + block.lambdas[1] * x0; orig = block.attn.squared_attention; block.attn.squared_attention = wrap(l)
            try: attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_)
            finally: block.attn.squared_attention = orig
            x = live + attention; m = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            if l == 1: x1 = x
            x = x + m
    idx = torch.arange(tokens.shape[0])
    if pos is None:
        W = m[:, 1:].reshape(-1, m.shape[-1]).float().cpu(); shares = {(l, h): store[l][:, h, 1:].reshape(-1) for l in (0, 1) for h in range(9)}
    else:
        W = m[idx, pos].float().cpu(); shares = {(l, h): store[l][idx, h, pos.cpu()] for l in (0, 1) for h in range(9)}
    return W, shares


def main() -> None:
    cls, _ = v287.classes(); cls = {k: v[:N] for k, v in cls.items()}
    fill = [L._single(t) for t in PHRASE]; filler = [fill[i % len(fill)] for i in range(K)]; targets = sorted({t for v in cls.values() for t in v})
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "targets": len(targets), "natural_rows": len(recs), "phrase": list(PHRASE), "length": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top4_min": TOP4_MIN, "r_min": R_MIN, "block0_min": BLOCK0_MIN, "overlap_min": OVERLAP_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; forwards = 0
    ids = torch.tensor(targets, device="cuda").unsqueeze(1); tab = v289.capture(backend, ids, torch.zeros(len(ids), dtype=torch.long, device="cuda")); forwards += 1; T = tab["mlp1"]
    toks = torch.tensor([filler + [t] for t in targets], device="cuda"); pos = torch.full((len(targets),), K, dtype=torch.long, device="cuda")
    alpha_of = lambda W: (W * T).sum(1) / (T * T).sum(1)
    W0, shares = run(backend, toks, pos); forwards += 1; a0 = alpha_of(W0)
    attr = {}
    for hd in HEADS:
        Wh, _ = run(backend, toks, pos, self_only=(hd,)); forwards += 1; attr[hd] = alpha_of(Wh) - a0
    Wall, _ = run(backend, toks, pos, self_only=tuple(HEADS)); forwards += 1; a_all = alpha_of(Wall) - a0
    summed = sum(attr.values()); closure = float((summed - a_all).abs().median())
    ctx_read = {hd: float((1 - shares[hd]).median()) for hd in HEADS}; attr_med = {hd: float(attr[hd].median()) for hd in HEADS}
    order = sorted(HEADS, key=lambda hd: -ctx_read[hd]); tot_read = sum(ctx_read.values()); top4_share = sum(ctx_read[hd] for hd in order[:4]) / tot_read
    def pearson(a, b):
        a, b = torch.tensor(a), torch.tensor(b); a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
    r_attr = pearson([attr_med[hd] for hd in HEADS], [ctx_read[hd] for hd in HEADS])
    block0 = sum(attr_med[hd] for hd in HEADS if hd[0] == 0) / max(sum(attr_med.values()), 1e-9)
    nat = torch.tensor([r["ids"] for r in recs], device="cuda"); nat_shares = {hd: [] for hd in HEADS}
    for s0 in range(0, len(recs), 64):
        _, sh = run(backend, nat[s0:s0 + 64], None); forwards += 1
        for hd in HEADS: nat_shares[hd].append(sh[hd])
    nat_read = {hd: float((1 - torch.cat(v)).median()) for hd, v in nat_shares.items()}; nat_order = sorted(HEADS, key=lambda hd: -nat_read[hd])
    overlap = len(set(order[:4]) & set(nat_order[:4]))
    key = lambda hd: f"{hd[0]}.{hd[1]}"
    report = {"alpha_native_median": float(a0.median()), "alpha_all_self_only_median": float((a0 + a_all).median()), "closure_median_abs": closure, "context_read_filler": {key(hd): ctx_read[hd] for hd in order}, "attribution_median": {key(hd): attr_med[hd] for hd in order},
              "top4_share_filler": top4_share, "pearson_attr_vs_read": r_attr, "block0_attribution_share": block0, "context_read_text": {key(hd): nat_read[hd] for hd in nat_order}, "top4_filler": [key(hd) for hd in order[:4]], "top4_text": [key(hd) for hd in nat_order[:4]], "overlap": overlap}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_head_closure": closure <= CLOSURE_TOL, "pred_b_self_share_is_head_concentrated": top4_share >= TOP4_MIN, "pred_c_alpha_attribution_matches_share": r_attr >= R_MIN, "pred_d_block0_carries_most": block0 >= BLOCK0_MIN, "pred_e_same_heads_on_text": overlap >= OVERLAP_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_self_share_heads_result_v298", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
