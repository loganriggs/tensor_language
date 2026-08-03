"""DENSEFORMER VARIANT E, probes (companion to qk_denseform.py). Runs on E at depths
4 and 12 AND on the existing 6-epoch A (vanilla) and B (clamped token line) checkpoints
at both depths, so the linear-percentage comparison Logan asked for is complete.

  1. LINEAR-IN-EMBEDDING PERCENTAGE, ALL LAYERS: for every layer's mlp write, ridge-fit
     (train[:1500], closed form over the token-frequency-weighted vocab table, VERBATIM
     machinery from qk_tokenline_probe.build_tables) the best linear map from the
     current-token RMS-NORMED embedding to the write. Report (a) held variance explained
     (held[:200], 102400 positions) and (b) the causal recovered fraction: substitute the
     layer's mlp write with the linear prediction, recovered = 1 - dCE_lin / dCE_floor
     where floor = global-mean ablation of that write. fp32 forwards, full held slice.
  2. TOKEN-DETERMINED FRACTION of each layer's mlp write: group-mean method + shuffled
     control, VERBATIM group_r2/shuffle_control from qk_tokenline_probe.
  3. E's learned weight matrix w_{l,j} (rows = block entries + readout row, columns =
     embedding, a_0, m_0, a_1, m_1, ...), with every |w - 1| > 0.5 flagged.
  4. Held CE table E vs A vs B with paired sequence-clustered standard errors (from
     qk_denseform_ce.json written by the training script).

Everything -> qk_denseform.json.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
import qk_tokenline_probe as TP          # group_r2, shuffle_control, fwd (A/B substitution)
from qk_denseform import DenseMini

QK = Q.QK
DEV = 'cuda'
D, V, T, NH, HD = Q.D, Q.V, Q.T, Q.NH, Q.HD
HELD, TRAIN = Q.HELD, Q.TRAIN
N_COLLECT = 200
N_TABLE = 1500
MIN_COUNT = 5
DEPTHS = [4, 12]

CKPT = {
    (4, 'A'): 'qk_tokenline_AS.pt', (4, 'B'): 'qk_tokenline_BS.pt',
    (4, 'E'): 'qk_denseform_4.pt',
    (12, 'A'): 'qk_tldepth_A12.pt', (12, 'B'): 'qk_tldepth_B12.pt',
    (12, 'E'): 'qk_denseform_12.pt',
}


def load_model(depth, variant):
    ck = torch.load(f'{QK}/{CKPT[(depth, variant)]}', map_location=DEV, weights_only=False)
    assert ck['config']['NL'] == depth
    if variant == 'E':
        torch.manual_seed(Q.SEED)
        m = DenseMini(depth).to(DEV)
    else:
        Q.NL = depth
        m = Q.make_model(variant)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck


@torch.no_grad()
def collect_mlp(model, depth, seqs, want_ids=False):
    """fp32 mlp writes for all layers over seqs. Returns (N, depth, D) cpu + ids."""
    mws, ids = [], []
    for i in range(0, len(seqs), 8):
        b = seqs[i:i + 8, :T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        mws.append(torch.stack([w.float().cpu() for w in col['mlp_write']], 1))
        if want_ids:
            ids.append(b.reshape(-1).cpu())
    mw = torch.cat(mws).transpose(1, 2).reshape(-1, depth, D)
    return mw, (torch.cat(ids).numpy().astype(np.int64) if want_ids else None)


@torch.no_grad()
def build_linear_tables(model, depth):
    """One sweep over TRAIN[:N_TABLE]: per-layer token-conditional sums -> global-mean
    floor + ridge linear-in-normed-embedding prediction tables (depth, V, D)."""
    sums = torch.zeros(depth, V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    for i in range(0, N_TABLE, 8):
        b = TRAIN[i:i + 8, :T]
        col = {'mlp_write': [], 'attn_write': [], 'entry_norm': []}
        model(b, collect=col)
        idsf = b.reshape(-1)
        for l in range(depth):
            sums[l].index_add_(0, idsf, col['mlp_write'][l].float().reshape(-1, D))
        cnts.index_add_(0, idsf, torch.ones_like(idsf, dtype=torch.float32))
    gmeans = sums.sum(1) / cnts.sum()                       # (depth, D)
    # ridge over the frequency-weighted vocab table (VERBATIM logic from
    # qk_tokenline_probe.build_tables, normed-embedding regressors, shared Gram matrix)
    E = F.rms_norm(model.wte.weight.detach(), (D,))
    Ea = torch.cat([E, torch.ones(V, 1, device=DEV)], 1)
    A = Ea.t() @ (cnts[:, None] * Ea)
    lam = 1e-4 * torch.trace(A) / (D + 1)
    Areg = A + lam * torch.eye(D + 1, device=DEV)
    lin = torch.empty_like(sums)
    for l in range(depth):
        W = torch.linalg.solve(Areg, Ea.t() @ sums[l])
        lin[l] = Ea @ W
    return gmeans, lin


@torch.no_grad()
def held_ce_sub(model, sub_table=None, layer=None, batch=8):
    """fp32 held CE with optional per-token (V,D) substitution of `layer`'s mlp write."""
    pts = []
    dense = isinstance(model, DenseMini)
    for i in range(0, len(HELD), batch):
        b = HELD[i:i + batch]
        inp = b[:, :T]
        sub = None if sub_table is None else {layer: sub_table[inp]}
        logits = model(inp, mlp_sub=sub) if dense else TP.fwd(model, inp, sub)
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:T + 1].reshape(-1),
                             reduction='none')
        pts.append(ce.cpu())
    pt = torch.cat(pts).numpy()
    return float(pt.mean()), pt


def probe(depth, variant):
    print(f"==== probing depth {depth} variant {variant} ====", flush=True)
    t0 = time.time()
    model, ck = load_model(depth, variant)
    r = {'held_ce_bf16_train_eval': ck['log']['final_held_ce']}
    if variant == 'E':
        r['mix_matrix'] = ck['log']['mix_matrix']
        r['w_stats'] = ck['log']['w_stats']
    elif variant == 'B':
        r['b_coeffs'] = ck['log']['mix']['b']

    # ---- held writes: token-determined fraction + linear variance explained ----
    mw, _ = collect_mlp(model, depth, HELD[:N_COLLECT])
    gids = HELD[:N_COLLECT, :T].reshape(-1).cpu().numpy().astype(np.int64)
    elig = np.ones(len(gids), bool)
    gmeans, lin = build_linear_tables(model, depth)
    lin_cpu = lin.float().cpu()
    gid_t = torch.from_numpy(gids)
    td, ve, norms = [], [], []
    for l in range(depth):
        X64 = mw[:, l, :].double()
        r2, G, cov = TP.group_r2(X64, gids, elig, MIN_COUNT)
        ctl = TP.shuffle_control(X64, gids, cov)
        td.append({'r2': round(r2, 4), 'shuffle_ctl': round(ctl, 4),
                   'groups': G, 'coverage': int(cov.sum())})
        pred = lin_cpu[l][gid_t].double()                   # (N, D) linear prediction
        resid = float((X64 - pred).pow(2).sum())
        tot = float((X64 - X64.mean(0)).pow(2).sum())
        ve.append(round(1 - resid / tot, 4))
        norms.append(round(float(mw[:, l].norm(dim=1).mean()), 3))
    r['token_determined_mlp'] = td
    r['linear_held_variance_explained'] = ve
    r['mlp_write_norm'] = norms
    del mw

    # ---- causal: mean-ablation floor vs linear substitution, every layer ----
    base, base_pt = held_ce_sub(model)
    r['base_ce_fp32'] = round(base, 5)
    causal = []
    for l in range(depth):
        floor_tab = gmeans[l][None].expand(V, D).contiguous()
        cef, ptf = held_ce_sub(model, floor_tab, l)
        cel, ptl = held_ce_sub(model, lin[l], l)
        dfloor, dlin = cef - base, cel - base
        rec = round(1 - dlin / dfloor, 4) if dfloor > 1e-6 else None
        causal.append({
            'dce_floor': round(dfloor, 5),
            'dce_floor_se': round(float((ptf - base_pt).std(ddof=1) / math.sqrt(len(ptf))), 6),
            'dce_lin': round(dlin, 5),
            'dce_lin_se': round(float((ptl - base_pt).std(ddof=1) / math.sqrt(len(ptl))), 6),
            'lin_recovered': rec})
        print(f"  L{l}: tokdet {td[l]['r2']:.3f}  linVE {ve[l]:.3f}  "
              f"dCEfloor {dfloor:.4f}  dCElin {dlin:.4f}  rec {rec}", flush=True)
    r['causal_per_layer'] = causal
    print(f"  depth {depth} {variant} done in {time.time() - t0:.0f}s", flush=True)
    del model, gmeans, lin
    torch.cuda.empty_cache()
    return r


if __name__ == '__main__':
    Q.gpu_guard()
    out = {'ce': json.load(open(f'{QK}/qk_denseform_ce.json'))}
    for depth in DEPTHS:
        out[str(depth)] = {}
        for variant in ['A', 'B', 'E']:
            out[str(depth)][variant] = probe(depth, variant)
        json.dump(out, open(f'{QK}/qk_denseform.json', 'w'), indent=2)

    # ---- compact summary tables ----
    for depth in DEPTHS:
        print(f"\n==== depth {depth} summary ====", flush=True)
        for variant in ['A', 'B', 'E']:
            r = out[str(depth)][variant]
            print(f" {variant} linVE  {[x for x in r['linear_held_variance_explained']]}",
                  flush=True)
            print(f" {variant} linREC {[c['lin_recovered'] for c in r['causal_per_layer']]}",
                  flush=True)
            print(f" {variant} tokdet {[t['r2'] for t in r['token_determined_mlp']]}",
                  flush=True)
    json.dump(out, open(f'{QK}/qk_denseform.json', 'w'), indent=2)
    print('saved qk_denseform.json', flush=True)
