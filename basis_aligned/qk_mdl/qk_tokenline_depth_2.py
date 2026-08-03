"""TOKEN-LINE DEPTH TEST, probes + hypothesis scoring (companion to qk_tokenline_depth.py).

Per depth (2, 4, 8, 12; depth 4 = the §106 6-epoch models qk_tokenline_{A,B}S.pt):
  1. Held CE A/B with per-token PAIRED sequence-clustered standard error on B-A
     (from qk_tldepth_ce.json written by the training script).
  2. WASH-OUT CURVE: at each block entry of trained variant A (stream x AFTER the entry
     mix, which for A is the identity), the variance fraction explained by current-token
     identity -- group-mean method VERBATIM from qk_tokenline_probe/qk_msg_bottleneck_2
     (min_count 5 + shuffled-label control), held[0:200] = 102400 positions.
     Also measured for B (the hypothesis says the line variant KEEPS the token).
  3. B's learned line coefficients b per block (from the checkpoints' train logs).
  4. Token-determined fraction of each layer's mlp write (same probe as §106) for BOTH
     variants at EVERY depth (depth 12 is the required one; depth 4 cross-checks §106).
Parameter counts per depth (embedding vs transformer body) reported as the confound.

HYPOTHESIS SCORE (depth story): (i) B-A on held CE becomes increasingly NEGATIVE with
depth; (ii) vanilla entry streams lose the token with depth (blocks 8+ of the deep
models far below anything in the depth-4 model) while B keeps it; (iii) deep B keeps
late-block coefficients strong (no depth-4-style fade); (iv) the §106 inversion (line
token-anchors the writes) persists at depth 12. Everything -> qk_tokenline_depth.json.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, math, time
import numpy as np
import torch
import torch.nn.functional as F

import qk_tokenline_train as Q
from qk_tokenline_probe import group_r2, shuffle_control   # verbatim group-mean method

QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
DEV = 'cuda'
D, V, T, NH, HD = Q.D, Q.V, Q.T, Q.NH, Q.HD
HELD = Q.HELD
N_COLLECT = 200
MIN_COUNT = 5
DEPTHS = [2, 4, 8, 12]

def ckpt_path(variant, depth):
    return (f'{QK}/qk_tokenline_{variant}S.pt' if depth == 4
            else f'{QK}/qk_tldepth_{variant}{depth}.pt')

def load_model(variant, depth):
    ck = torch.load(ckpt_path(variant, depth), map_location=DEV, weights_only=False)
    assert ck['config']['NL'] == depth, (depth, ck['config']['NL'])
    Q.NL = depth
    m = Q.make_model(variant)
    m.load_state_dict(ck['state_dict'])
    m.eval().float()
    return m, ck

@torch.no_grad()
def collect_entry_and_mlp(model, depth):
    """Held[:N_COLLECT]: stream at each block entry (post-mix) + each mlp write.
    Returns cpu fp32 (N, depth, D) x2."""
    ents, mws = [], []
    cos_f = model.cos[None, :T, None, :]
    sin_f = model.sin[None, :T, None, :]
    mask = model.mask[:T, :T]
    for i in range(0, N_COLLECT, 8):
        idx = HELD[i:i + 8, :T]
        B = idx.shape[0]
        e_raw = model.wte(idx)
        e_norm = F.rms_norm(e_raw, (D,))
        x = e_norm
        e0 = e_raw if model.variant == 'D' else e_norm
        ent_l, mw_l = [], []
        for blk in model.h:
            x = blk.a * x + blk.b * e0
            ent_l.append(x.float().cpu())
            h = F.rms_norm(x, (D,))
            def qk(lin):
                z = lin(h).view(B, T, NH, HD)
                return Q.apply_rot(F.rms_norm(z, (HD,)), cos_f, sin_f)
            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(h).view(B, T, NH, HD)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, D)
            x = x + blk.c_proj(y)
            hn = F.rms_norm(x, (D,))
            mw = blk.Down(blk.Left(hn) * blk.Right(hn)) + blk.Down_bias
            mw_l.append(mw.float().cpu())
            x = x + mw
        ents.append(torch.stack(ent_l, 1))   # (B, depth, T, D)
        mws.append(torch.stack(mw_l, 1))
    ent = torch.cat(ents).transpose(1, 2).reshape(-1, depth, D)
    mw = torch.cat(mws).transpose(1, 2).reshape(-1, depth, D)
    return ent, mw

def r2_curve(W, gids, depth):
    elig = np.ones(len(gids), bool)
    out = []
    for li in range(depth):
        X64 = W[:, li, :].double()
        r2, G, cov = group_r2(X64, gids, elig, MIN_COUNT)
        ctl = shuffle_control(X64, gids, cov)
        out.append({'r2': round(r2, 4), 'shuffle_ctl': round(ctl, 4),
                    'groups': G, 'coverage': int(cov.sum())})
    return out

def param_counts(depth, variant):
    emb = V * D
    per_block = 6 * D * D + 3 * Q.EXP * D * D + D          # attn linears + Left/Right/Down + Down_bias
    body = depth * per_block + (depth if variant == 'B' else 0)
    return {'total': emb + body, 'embedding': emb, 'body': body,
            'body_frac': round(body / (emb + body), 4)}

if __name__ == '__main__':
    Q.gpu_guard()
    gids = HELD[:N_COLLECT, :T].reshape(-1).cpu().numpy().astype(np.int64)
    ce = json.load(open(f'{QK}/qk_tldepth_ce.json'))
    out = {'ce': ce, 'depths': {}}
    for depth in DEPTHS:
        dres = {'params': {v: param_counts(depth, v) for v in 'AB'}}
        for variant in 'AB':
            t0 = time.time()
            model, ck = load_model(variant, depth)
            ent, mw = collect_entry_and_mlp(model, depth)
            r = {'b_coeffs': [round(b, 4) for b in ck['log']['mix']['b']],
                 'entry_stream_norm': [round(float(ent[:, li].norm(dim=1).mean()), 2)
                                       for li in range(depth)],
                 'washout_entry_r2': r2_curve(ent, gids, depth),
                 'token_determined_mlp': r2_curve(mw, gids, depth)}
            dres[variant] = r
            print(f"depth {depth} {variant} ({time.time()-t0:.0f}s): "
                  f"entry r2 {[e['r2'] for e in r['washout_entry_r2']]}", flush=True)
            print(f"    mlp tokdet {[e['r2'] for e in r['token_determined_mlp']]} "
                  f"b={r['b_coeffs']}", flush=True)
            del model, ent, mw
            torch.cuda.empty_cache()
        out['depths'][str(depth)] = dres

    # ---------------- hypothesis scoring ----------------
    ds = [2, 4, 8, 12]
    ba = [ce[str(d)]['B_minus_A'] for d in ds]
    se = [ce[str(d)]['B_minus_A_se_seq'] for d in ds]
    washA = {d: [e['r2'] for e in out['depths'][str(d)]['A']['washout_entry_r2']] for d in ds}
    washB = {d: [e['r2'] for e in out['depths'][str(d)]['B']['washout_entry_r2']] for d in ds}
    tdA12 = [e['r2'] for e in out['depths']['12']['A']['token_determined_mlp']]
    tdB12 = [e['r2'] for e in out['depths']['12']['B']['token_determined_mlp']]
    min_wash4 = min(washA[4])
    late_wash12 = washA[12][8:]
    score = {
        'pred1_BminusA_increasingly_negative': {
            'B_minus_A_by_depth': dict(zip(map(str, ds), [round(x, 5) for x in ba])),
            'se_seq': dict(zip(map(str, ds), [round(x, 5) for x in se])),
            'monotone_decreasing': all(ba[i+1] < ba[i] for i in range(3)),
            'negative_at_12': ba[-1] < -2 * se[-1],
            'confirmed': all(ba[i+1] < ba[i] for i in range(3)) and ba[-1] < -2 * se[-1]},
        'pred2_vanilla_washout': {
            'A_entry_r2': {str(d): washA[d] for d in ds},
            'B_entry_r2': {str(d): washB[d] for d in ds},
            'min_depth4_A': min_wash4, 'depth12_blocks8plus_A': late_wash12,
            'confirmed': max(late_wash12) < min_wash4},
        'pred3_deep_B_keeps_line_late': {
            'b_by_depth': {str(d): out['depths'][str(d)]['B']['b_coeffs'] for d in ds},
            'depth4_last_b': out['depths']['4']['B']['b_coeffs'][-1],
            'depth12_last4_b': out['depths']['12']['B']['b_coeffs'][-4:]},
        'pred4_inversion_persists_at_12': {
            'A_mlp_r2': tdA12, 'B_mlp_r2': tdB12,
            'confirmed': all(b > a for a, b in zip(tdA12[1:], tdB12[1:]))},
    }
    out['score'] = score
    json.dump(out, open(f'{QK}/qk_tokenline_depth.json', 'w'), indent=2)
    print(json.dumps(score, indent=2), flush=True)
    print('saved qk_tokenline_depth.json', flush=True)
