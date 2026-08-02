"""TOKEN-LINE EXPERIMENT, probes (Logan's second prediction: what the line changes
about what layers COMPUTE). Runs on the four trained models qk_tokenline_{A,B,C,D}.pt.

  1. TOKEN-DETERMINED FRACTION per layer: variance of each layer's mlp write (and,
     as a bonus, attn write) explained by current-token identity -- group-mean method
     VERBATIM from qk_msg_bottleneck_2.py (min_count 5 + shuffled-label control),
     on held data (held[0:200], 102400 positions).
  2. LINEAR-IN-EMBEDDING CAUSAL FRACTION for layers 0 and 1: substitute the layer's
     mlp write with (i) its train token-conditional mean table and (ii) the best
     ridge linear map of the embedding (raw and rms-normed regressors, train-fit,
     closed form over the token-frequency-weighted vocab table); measure recovered
     fraction of the layer's mean-ablation delta cross-entropy (machinery pattern
     from qk_redteam_arcs.py attack 2: recovered = 1 - dCE_surrogate/dCE_floor).
  3. WRITE NORMS per layer: mlp/attn write norm vs the stream norm at block entry and
     the effective token-line contribution |b|*||e0||; plus layer-0 token-table norms
     (the block-0 "loud table" check).
  4. Learned mixing scalars a/b per block (B, C, D).

Predictions scored (P1-P4). Everything -> qk_tokenline.json. fp32 forward throughout
(training eval used bf16; base CE is recomputed in fp32 here for internal consistency).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
DEV = 'cuda'
D, V, T, NL, NH, HD = Q.D, Q.V, Q.T, Q.NL, Q.NH, Q.HD
HELD, TRAIN = Q.HELD, Q.TRAIN
Q.gpu_guard()

# ---------------- group-mean variance method (VERBATIM qk_msg_bottleneck_2.py) ----------------
def group_r2(X64, gids, elig, min_count):
    g = gids[elig]
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    keep = counts >= min_count
    cov = np.zeros(len(gids), bool)
    idx_el = np.where(elig)[0]
    cov[idx_el[keep[inv]]] = True
    Xc = X64[torch.from_numpy(cov)]
    gcov = gids[cov]
    u2, inv2 = np.unique(gcov, return_inverse=True)
    G = len(u2)
    s = torch.zeros(G, X64.shape[1], dtype=torch.float64)
    s.index_add_(0, torch.from_numpy(inv2), Xc)
    n = torch.from_numpy(np.bincount(inv2).astype(np.float64))
    sq = float(Xc.pow(2).sum())
    within = sq - float((s.pow(2).sum(1) / n).sum())
    grand = Xc.sum(0)
    tot = sq - float(grand.pow(2).sum()) / Xc.shape[0]
    r2 = 1.0 - within / max(tot, 1e-12)
    return r2, G, cov

def shuffle_control(X64, gids, cov, seed=0):
    rng = np.random.default_rng(seed)
    g = gids.copy()
    idxc = np.where(cov)[0]
    g[idxc] = g[rng.permutation(idxc)]
    r2, _, _ = group_r2(X64, g, cov, 1)
    return r2

# ---------------- standalone forward with mlp-write substitution ----------------
@torch.no_grad()
def fwd(model, idx, mlp_sub=None):
    """mlp_sub: dict {layer_idx: (B,T,D) tensor} replacing that layer's mlp write."""
    B, Tq = idx.shape
    e_raw = model.wte(idx)
    e_norm = F.rms_norm(e_raw, (D,))
    x = e_norm
    e0 = e_raw if model.variant == 'D' else e_norm
    cos = model.cos[None, :Tq, None, :]
    sin = model.sin[None, :Tq, None, :]
    mask = model.mask[:Tq, :Tq]
    for li, blk in enumerate(model.h):
        x = blk.a * x + blk.b * e0
        h = F.rms_norm(x, (D,))
        def qk(lin):
            z = lin(h).view(B, Tq, NH, HD)
            return Q.apply_rot(F.rms_norm(z, (HD,)), cos, sin)
        q, k = qk(blk.c_q), qk(blk.c_k)
        q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
        v = blk.c_v(h).view(B, Tq, NH, HD)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D)
        x = x + blk.c_proj(y)
        if mlp_sub is not None and li in mlp_sub:
            mw = mlp_sub[li]
        else:
            hn = F.rms_norm(x, (D,))
            mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias
        x = x + mw
    x = F.rms_norm(x, (D,))
    return 30 * torch.tanh((x @ model.wte.weight.t()) / 30)

@torch.no_grad()
def held_ce(model, sub_table=None, layer=None, batch=8):
    """sub_table: (V,D) per-token substitution for `layer`'s mlp write (None = clean).
    Returns (mean CE, per-token CE numpy)."""
    pts = []
    for i in range(0, len(HELD), batch):
        b = HELD[i:i + batch]
        inp = b[:, :T]
        sub = None if sub_table is None else {layer: sub_table[inp]}
        logits = fwd(model, inp, sub)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        pts.append(ce.cpu())
    pt = torch.cat(pts).numpy()
    return float(pt.mean()), pt

# ---------------- per-variant probes ----------------
N_COLLECT = 200          # held seqs for the variance probe
N_TABLE = 1500           # train seqs for table/linear fits
MIN_COUNT = 5

def load_model(variant, suffix=''):
    ck = torch.load(f'{QK}/qk_tokenline_{variant}{suffix}.pt', map_location=DEV, weights_only=False)
    m = Q.make_model(variant)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck

@torch.no_grad()
def collect_writes(model):
    """Held[:N_COLLECT] mlp/attn writes + norms. Returns cpu fp32 (N,NL,D) x2 + norm dict."""
    mws, aws, entry = [], [], []
    for i in range(0, N_COLLECT, 8):
        b = HELD[i:i + 8, :T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        mws.append(torch.stack([w.float().cpu() for w in col['mlp_write']], 1))   # (B,NL,T,D)
        aws.append(torch.stack([w.float().cpu() for w in col['attn_write']], 1))
        entry.append(col['entry_norm'])
    mw = torch.cat(mws).transpose(1, 2).reshape(-1, NL, D)    # (N, NL, D)
    aw = torch.cat(aws).transpose(1, 2).reshape(-1, NL, D)
    entry_norm = np.array(entry).mean(0).tolist()             # per-layer mean ||x|| at entry
    return mw, aw, entry_norm

@torch.no_grad()
def build_tables(model, layer):
    """Train token-conditional mean of layer's mlp write + ridge linear-in-embedding fits.
    Returns dict of (V,D) substitution tables on DEV + table stats."""
    sums = torch.zeros(V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    for i in range(0, N_TABLE, 8):
        b = TRAIN[i:i + 8, :T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        w = col['mlp_write'][layer].float().reshape(-1, D)
        ids = b.reshape(-1)
        sums.index_add_(0, ids, w)
        cnts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    gmean = sums.sum(0) / cnts.sum()
    seen = cnts > 0
    means = torch.where(seen[:, None], sums / cnts.clamp_min(1)[:, None], gmean[None])
    # ridge linear map of the embedding, closed form over the frequency-weighted vocab table
    def ridge(E):
        Ea = torch.cat([E, torch.ones(V, 1, device=DEV)], 1)          # intercept
        A = Ea.t() @ (cnts[:, None] * Ea)
        Bm = Ea.t() @ sums
        lam = 1e-4 * torch.trace(A) / (D + 1)
        W = torch.linalg.solve(A + lam * torch.eye(D + 1, device=DEV), Bm)
        return Ea @ W                                                  # (V,D) predictions
    lin_raw = ridge(model.wte.weight.detach())
    lin_norm = ridge(F.rms_norm(model.wte.weight.detach(), (D,)))
    tstat_mask = cnts >= MIN_COUNT
    tn = means[tstat_mask].norm(dim=1)
    stats = {'tokens_seen': int(seen.sum()), 'tokens_ge5': int(tstat_mask.sum()),
             'table_norm_mean': float(tn.mean()), 'table_norm_max': float(tn.max()),
             'global_mean_norm': float(gmean.norm())}
    return {'floor': gmean[None].expand(V, D).contiguous(), 'tok': means,
            'lin_raw': lin_raw, 'lin_norm': lin_norm}, stats

def probe_variant(variant, suffix=''):
    print(f"==== probing {variant}{suffix} ====", flush=True)
    t0 = time.time()
    model, ck = load_model(variant, suffix)
    r = {'mix': ck['log']['mix'], 'held_ce_bf16_train_eval': ck['log']['final_held_ce']}

    # ---- 1. token-determined fraction + 3. norms ----
    mw, aw, entry_norm = collect_writes(model)
    gids = HELD[:N_COLLECT, :T].reshape(-1).cpu().numpy().astype(np.int64)
    elig = np.ones(len(gids), bool)
    r['token_determined'] = {}
    for kind, W in [('mlp', mw), ('attn', aw)]:
        per_layer = []
        for li in range(NL):
            X64 = W[:, li, :].double()
            r2, G, cov = group_r2(X64, gids, elig, MIN_COUNT)
            ctl = shuffle_control(X64, gids, cov)
            per_layer.append({'r2': round(r2, 4), 'shuffle_ctl': round(ctl, 4),
                              'groups': G, 'coverage': int(cov.sum())})
        r['token_determined'][kind] = per_layer
    e_raw = model.wte(HELD[:N_COLLECT, :T]).float()
    raw_norm = e_raw.norm(dim=-1)
    r['norms'] = {
        'entry_stream': [round(v, 2) for v in entry_norm],
        'mlp_write': [round(float(mw[:, li].norm(dim=1).mean()), 3) for li in range(NL)],
        'attn_write': [round(float(aw[:, li].norm(dim=1).mean()), 3) for li in range(NL)],
        'emb_raw_mean': round(float(raw_norm.mean()), 3),
        'emb_raw_p10_p90': [round(float(torch.quantile(raw_norm.reshape(-1)[:50000], q)), 3)
                            for q in (0.1, 0.9)],
        'emb_normed': round(math.sqrt(D), 3),
        'line_contrib': [round(abs(b) * (float(raw_norm.mean()) if variant == 'D'
                                         else math.sqrt(D)), 3)
                         for b in ck['log']['mix']['b']],
    }
    del mw, aw

    # ---- 2. linear-in-embedding causal fraction, layers 0 and 1 ----
    base, base_pt = held_ce(model)
    r['base_ce_fp32'] = round(base, 5)
    r['causal'] = {}
    for layer in (0, 1):
        tabs, stats = build_tables(model, layer)
        entry = {'table_stats': stats}
        dfloor = None
        for name in ('floor', 'tok', 'lin_raw', 'lin_norm'):
            ce, pt = held_ce(model, tabs[name], layer)
            d = ce - base
            dse = float((pt - base_pt).std(ddof=1) / math.sqrt(len(pt)))
            entry[name] = {'ce': round(ce, 5), 'dce': round(d, 5), 'dce_se': round(dse, 6)}
            if name == 'floor':
                dfloor = d
            else:
                entry[name]['recovered'] = round(1 - d / dfloor, 4) if dfloor > 1e-6 else None
        r['causal'][f'layer{layer}'] = entry
        del tabs
        torch.cuda.empty_cache()
    print(f"  {variant} done in {time.time() - t0:.0f}s: "
          f"tokdet mlp {[e['r2'] for e in r['token_determined']['mlp']]} "
          f"L0 tok rec {r['causal']['layer0']['tok'].get('recovered')} "
          f"lin_raw rec {r['causal']['layer0']['lin_raw'].get('recovered')}", flush=True)
    del model
    torch.cuda.empty_cache()
    return r

# ---------------- run all + score predictions ----------------
def score(res, ce):
    td = {v: [e['r2'] for e in res[v]['token_determined']['mlp']] for v in 'ABCD'}
    l0rec = {v: res[v]['causal']['layer0']['tok'].get('recovered') for v in 'ABCD'}
    l0norm = {v: res[v]['norms']['mlp_write'][0] for v in 'ABCD'}
    return {
        'P1_line_helps_ce': {'ce': ce,
                             'confirmed': all(ce['A'] > ce[v] for v in 'BCD')},
        'P2_A_more_token_determined_past0': {
            'r2_mlp': td,
            'confirmed': all(td['A'][li] > td[v][li] for v in 'BCD' for li in (1, 2, 3))},
        'P3_A_layer0_most_tablelike': {
            'tok_recovered_L0': l0rec, 'mlp_write_norm_L0': l0norm,
            'confirmed': (all(l0rec['A'] >= l0rec[v] for v in 'BCD') and
                          all(l0norm['A'] >= l0norm[v] for v in 'BCD'))},
        'P4_D_layer0_least_tablelike': {
            'tok_recovered_L0': l0rec,
            'confirmed': all(l0rec['D'] <= l0rec[v] for v in 'ABC')},
    }

if __name__ == '__main__':
    out = {}
    for suffix, cefile, key in [('', 'qk_tokenline_ce.json', 'budget15ep'),
                                ('S', 'qk_tokenline_ce_short.json', 'budget6ep')]:
        if not os.path.exists(f'{QK}/{cefile}'):
            print(f"skip {key}: {cefile} missing", flush=True)
            continue
        cej = json.load(open(f'{QK}/{cefile}'))
        res = {v: probe_variant(v, suffix) for v in 'ABCD'}
        res['ce'] = cej
        res['predictions'] = score(res, {v: cej[v]['held_ce'] for v in 'ABCD'})
        out[key] = res
        print(f"-- {key} predictions:", json.dumps(res['predictions'], indent=2), flush=True)
    json.dump(out, open(f'{QK}/qk_tokenline.json', 'w'), indent=2)
    print('saved qk_tokenline.json', flush=True)
