"""Shared machinery for the V9 / V9b / width-768 follow-ups to RESULTS section 109
(Logan's queue). All three reuse the V8 stack (qk_v8_train.V8Route + the section-108
probe machinery in qk_deeproute_train_2):

  V9   = subspace-partitioned write slots + group-lasso reads + N=6 LOOKBACK WINDOW
         (window rule verbatim from qk_window_train.py: block l reads only the writes
         of blocks l-6..l-1, embedding visible only if l < 6; readout reads the last
         6 blocks' writes, no embedding at depth 12).
  V9b  = slots + group-lasso + LOOSER DILATION mask, visibility distances
         {1,2,3,4,8} instead of section 109's {1,2,4,8} (embedding visible iff
         l in {1,2,3,4,8}, plus the base case l=0; readout sees blocks with
         12-j in the set, i.e. {4,8,9,10,11}).
  W768 = the no-mask free config (slots + group-lasso, full visibility) at width
         768, depth 12, slot size 32, vs a width-768 vanilla control.

Conventions pinned by the directive: FineWeb cooc corpus train [0:5500] held
[5500:6000], 6-epoch budget, batch 8, lr swept per arch {0.001,0.002,0.003}
(w768: {0.001,0.002}), group-lasso coefficient FIXED at 1e-4 (the section-109
sweep winner), nonzero write init (mandatory: the windowed/masked readout has no
embedding fallback), GPU guard, sequential runs.

The lr sweeps here run WITH the 1e-4 penalty active (the coefficient is fixed by
the directive, so the sweep matches the full-run objective; section 109's 400-step
coefficient sweep measured the 1e-4 penalty's short-run cost at -0.007 nats, a tie,
so this deviation from V8's zero-penalty sweep order is immaterial and is noted).

full_probe() reproduces qk_v8_probe.py's measurement block: causal consumption
graph, wiring agreement (all / effectual / top-10 precision), per-block totals +
dead blocks, exact terms at blocks 4/6/9 with causal recovery, token-determined +
linear-in-embedding standard probes.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch

import qk_tokenline_train as Q
import qk_deeproute_train as R
import qk_v8_train as V8T
import qk_deeproute_train_2 as R2
import qk_v8_probe as P8
from qk_deeproute_train import DEPTH

QK = Q.QK
EFFECTUAL = 0.005
TERM_BLOCKS = [4, 6, 9]
GROUP_COEFF = 1e-4                     # fixed by the directive (section-109 winner)
LRS = [0.001, 0.002, 0.003]
WINDOW = 6
DIL_LOOSE = (1, 2, 3, 4, 8)


def register(variant):
    """Sum kernel + V5 slot projections + nonzero write init (mask via vis_fn)."""
    R.KERNEL[variant] = 'sum'
    R.PROJ.add(variant)
    R.WRITE_INIT.add(variant)


def window_vis(li, N=WINDOW, depth=DEPTH):
    """qk_window_train rule: writes of blocks max(0,li-N)..li-1; emb iff li < N.
    li == 0 reads the embedding (it has no other possible input)."""
    if li == 0:
        return [0]
    idxs = [0] if li < N else []
    for j in range(max(0, li - N), min(li, depth)):
        idxs += [1 + 2 * j, 2 + 2 * j]
    return sorted(idxs)


def dil_vis(li, dil=DIL_LOOSE, depth=DEPTH):
    """Dilation rule (qk_deeproute_train.visible) with a configurable distance set."""
    if li == 0:
        return [0]
    idxs = [0] if li in dil else []
    for j in range(min(li, depth)):
        if li - j in dil:
            idxs += [1 + 2 * j, 2 + 2 * j]
    return sorted(idxs)


def make_variant(variant, vis_fn=None):
    register(variant)
    torch.manual_seed(Q.SEED)
    m = V8T.V8Route(variant, DEPTH).to('cuda')
    if vis_fn is not None:
        m.vis = [vis_fn(li) for li in range(DEPTH + 1)]
    return m


def load_variant(stem, variant, vis_fn=None):
    ck = torch.load(f'{QK}/{stem}.pt', map_location='cuda', weights_only=False)
    m = make_variant(variant, vis_fn)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


# ---------------- positive control: full-visibility + zero writes == vanilla A ----------------
@torch.no_grad()
def identity_control(variant, vis_fn_full, ref_model=None):
    """The variant's model with FULL visibility (vis_fn_full must reduce the rule to
    all-streams), identity slot projections, and zeroed writes must equal the vanilla
    variant-A MiniBilin at init (max |logit diff| < 1e-3, fp32). Exercises the actual
    vis-override plumbing used in training."""
    idx = Q.HELD[:2, :Q.T]
    if ref_model is None:
        Q.NL = DEPTH
        ma = Q.make_model('A').eval().float()
    else:
        ma = ref_model
    ref = ma(idx)
    if ref_model is None:
        del ma
        torch.cuda.empty_cache()
    m = make_variant(variant, vis_fn_full).eval().float()
    full = [list(range(1 + 2 * min(li, DEPTH))) for li in range(DEPTH + 1)]
    assert m.vis == full, "vis_fn_full does not reduce to full visibility"
    m.wmask.fill_(1.0)
    for blk in m.h:
        blk.c_proj.weight.zero_()
        blk.Down.weight.zero_()
    d = (m(idx) - ref).abs().max().item()
    print(f"control {variant}(full vis, identity proj, zero writes)==A: "
          f"max |logit diff| {d:.2e}", flush=True)
    assert d < 1e-3
    del m
    torch.cuda.empty_cache()


# ---------------- paired CE vs a control heldloss file ----------------
def paired_ce(loss_file, control_file, label='control'):
    lv = np.load(f'{QK}/{loss_file}')
    lc = np.load(f'{QK}/{control_file}')
    dd = lv - lc
    ds = dd.reshape(len(Q.HELD), Q.T).mean(1)
    return {'held_ce': float(lv.mean()),
            f'{label}_held_ce': float(lc.mean()),
            f'minus_{label}': float(dd.mean()),
            f'minus_{label}_se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
            f'minus_{label}_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds)))}


# ---------------- the full probe block (verbatim measurements of qk_v8_probe) ----------------
def full_probe(model, ck, save_cb=None):
    r = {'held_ce_bf16': ck['log']['final_held_ce'], 'lr': ck['config']['lr'],
         'group_coeff': ck['config'].get('group_coeff'),
         'spikes': ck['log']['spikes']}
    t0 = time.time()
    base, means = R2.base_and_means(model)
    r['base_ce_fp32_abl'] = round(base, 5)

    # 1. causal consumption graph over all visible pairs
    dce = R2.consumption_graph(model, base, means)
    pairs = [(li, si, dce[li][si]) for li in dce for si in dce[li]]
    pairs.sort(key=lambda p: -p[2])
    r['consumption_top40'] = [
        {'consumer': ('readout' if li == DEPTH else f'block{li}'),
         'source': R2.stream_name(si), 'dce': round(v, 5)}
        for li, si, v in pairs[:40]]
    r['consumption_matrix'] = {str(li): {str(si): round(v, 6)
                                         for si, v in dce[li].items()}
                               for li in dce}
    if save_cb:
        save_cb(r)

    # per-source-block totals (dead-block check)
    totals = {}
    for j in range(DEPTH):
        tot = sum(dce[li].get(si, 0.0) for li in dce
                  for si in (1 + 2 * j, 2 + 2 * j) if si in dce[li])
        totals[j] = round(tot, 5)
    r['per_block_total_consumption'] = {str(j): totals[j] for j in totals}
    r['min_block_total'] = min(totals.values())
    r['dead_blocks_below_0.001'] = [j for j in totals if totals[j] < 0.001]
    print(f"per-block totals: {totals}", flush=True)

    # 2. wiring agreement (weight support vs causal consumption)
    scores, _ = R2.weight_wiring(model, 'VX')
    wp = [(li, si) for li in scores for si in scores[li]]
    sup = [scores[li][si] for li, si in wp]
    cau = [dce[li][si] for li, si in wp]
    r['wiring_n_pairs'] = len(wp)
    r['wiring_spearman_all'] = round(R2.spearman(sup, cau), 4)
    eff = [k for k in range(len(wp)) if cau[k] > EFFECTUAL]
    r['wiring_n_effectual'] = len(eff)
    r['wiring_spearman_effectual'] = round(
        R2.spearman([sup[k] for k in eff], [cau[k] for k in eff]), 4)
    top_sup = set(np.argsort(sup)[::-1][:10].tolist())
    top_cau = set(np.argsort(cau)[::-1][:10].tolist())
    r['wiring_top10_precision'] = len(top_sup & top_cau) / 10.0
    r['wiring_top10_support_edges'] = [
        {'consumer': ('readout' if wp[k][0] == DEPTH else f'block{wp[k][0]}'),
         'source': R2.stream_name(wp[k][1]), 'support': round(sup[k], 3),
         'dce': round(cau[k], 5)} for k in sorted(top_sup, key=lambda k: -sup[k])]
    r['weight_support_matrix'] = {str(li): {str(si): round(v, 3)
                                            for si, v in scores[li].items()}
                                  for li in scores}
    print(f"wiring: n {len(wp)}, Spearman all {r['wiring_spearman_all']}, "
          f"effectual ({len(eff)} pairs) {r['wiring_spearman_effectual']}, "
          f"top-10 precision {r['wiring_top10_precision']}", flush=True)
    if save_cb:
        save_cb(r)

    # 3. routing sparsity
    sp = R2.routing_sparsity(model)
    r['routing_sparsity'] = {str(k): v for k, v in sp.items()}
    r['mean_effective_inputs'] = round(
        float(np.mean([v['effective_inputs'] for v in sp.values()])), 3)

    # 4. exact-term decomposability at blocks 4, 6, 9
    r['terms'] = {}
    for mid in TERM_BLOCKS:
        R2.MID = mid
        tr = R2.term_decomposition(model, 'VX', base)
        r['terms'][f'block{mid}'] = tr
        print(f"terms block{mid}: n {tr['n_terms']} "
              f"rel_err {tr['reassembly_rel_err']:.2e} k95 {tr['k95_energy']} "
              f"recovered {tr['topk95_recovered']}", flush=True)
    if save_cb:
        save_cb(r)

    # 5. standard probes (token-determined + linear-in-embedding, guarded version)
    r['probes'] = P8.standard_probes_guarded(model)
    print(f"probe done in {time.time() - t0:.0f}s", flush=True)
    if save_cb:
        save_cb(r)
    return r


def json_saver(path, key):
    def cb(r):
        out = json.load(open(path)) if os.path.exists(path) else {}
        out[key] = r
        json.dump(out, open(path, 'w'), indent=2)
    return cb
