"""Is block 0's (linear) write redundant with a downstream LINEAR read of the persistent embedding? (Logan)

The embedding x0 = rms(wte) is re-injected at every block (lambda1*x0), so any downstream layer can read a
fixed linear function of it for free. Block 0's linear cap is W * xhat0 + b, where xhat0 = rms(emb + attn0).
Compare, as whole-model substitutions of block 0's MLP output (rest of model exact):

    mean           per-position mean (floor)               -> +1.2341 (gate)
    xhat_lin       W  * rms(emb + attn0) + b   (full cap)  -> +0.0796 (gate)
    emb_lin        We * rms(emb)        + be   (emb only)   -> ? = what a downstream linear read of the
                                                                 persistent embedding could itself provide
    embattn_lin    linear in [emb, attn0] (pre-norm)        -> ? isolates the INPUT-norm's contribution

If emb_lin ~ xhat_lin  -> block 0's linear write is redundant with a linear read of the embedding (Logan's
    worry: the linear part is "free"; block 0's real job is the small nonlinear remainder).
If emb_lin >> xhat_lin -> block 0 provides something a linear-read-of-emb cannot: the decaying attn0 signal
    and/or the input-norm combination rms(emb+attn0) that downstream cannot reconstruct.

Held FW[448:600,:128]; paired SE. Fits on TRAIN.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0); DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18'); NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6
SUB = {}   # filled with fitted maps + mean


@torch.no_grad()
def fwd(idx, mode=None, collect=False):
    B, T = idx.shape
    emb = F.rms_norm(m.transformer.wte(idx), (D,)); x = emb; x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    got = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        hcur = F.rms_norm(x, (D,))
        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn
        xhat = F.rms_norm(x, (D,))
        mo_true = blk.mlp(xhat)
        if li == 0 and collect:
            got['emb'] = emb.detach(); got['attn0'] = attn.detach()
            got['xhat0'] = xhat.detach(); got['mo0'] = mo_true.detach()
        if li == 0 and mode is not None and mode != 'full':
            if mode == 'mean':
                mo = SUB['mean'].unsqueeze(0).expand(B, -1, -1)
            elif mode == 'xhat_lin':
                mo = xhat @ SUB['Wx'] + SUB['bx']
            elif mode == 'emb_lin':
                mo = emb @ SUB['We'] + SUB['be']
            elif mode == 'embattn_lin':
                feat = torch.cat([emb, attn], -1)
                mo = feat @ SUB['Wea'] + SUB['bea']
        else:
            mo = mo_true
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T - 1), got


def held(mode):
    return torch.cat([fwd(HELD[i:i + B0], mode=mode)[0] for i in range(0, HELD.shape[0], B0)])


def fit(feat_key_list, extra=None):
    """ridge least squares predicting block-0 output from concatenated features (each B,T,*)."""
    Fdim = None
    A = None; Bm = None
    for i in range(0, TRAIN.shape[0], B0):
        _, g = fwd(TRAIN[i:i + B0], collect=True)
        feats = torch.cat([g[k] for k in feat_key_list], -1)
        fa = torch.cat([feats, torch.ones_like(feats[..., :1])], -1).reshape(-1, feats.shape[-1] + 1).double()
        if A is None:
            Fdim = feats.shape[-1]; A = torch.zeros(Fdim + 1, Fdim + 1, device=DEV, dtype=torch.float64)
            Bm = torch.zeros(Fdim + 1, D, device=DEV, dtype=torch.float64)
        A += fa.T @ fa; Bm += fa.T @ g['mo0'].reshape(-1, D).double()
    sol = torch.linalg.solve(A + 1.0 * torch.eye(Fdim + 1, device=DEV, dtype=torch.float64), Bm).float()
    return sol[:Fdim], sol[Fdim]


print('fitting block-0 substitutes ...', flush=True)
SUB['Wx'], SUB['bx'] = fit(['xhat0'])
SUB['We'], SUB['be'] = fit(['emb'])
SUB['Wea'], SUB['bea'] = fit(['emb', 'attn0'])
S, T = HELD.shape; msum = torch.zeros(T, D, device=DEV)
for i in range(0, S, B0):
    _, g = fwd(HELD[i:i + B0], collect=True); msum += g['mo0'].sum(0)
SUB['mean'] = msum / S

base = torch.cat([fwd(HELD[i:i + B0])[0] for i in range(0, HELD.shape[0], B0)])
BASE = float(base.mean())
print(f'GATE base CE {BASE:.4f}', flush=True)


def rep(name, mode):
    ce = held(mode); d = ce - base
    v = (round(float(d.mean()), 4), round(float(d.mean(1).std() / np.sqrt(d.shape[0])), 4))
    print(f'  {name:26s} dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
    return v


res = {'meta': {'base_ce': round(BASE, 4)}}
res['mean'] = rep('mean floor', 'mean')
res['xhat_lin'] = rep('linear in rms(emb+attn0)', 'xhat_lin')
res['emb_lin'] = rep('linear in emb only', 'emb_lin')
res['embattn_lin'] = rep('linear in [emb, attn0] prenorm', 'embattn_lin')

fl = res['mean'][0]
res['analysis'] = {
    'floor': fl,
    'xhat_lin_recovers': round(fl - res['xhat_lin'][0], 4),
    'emb_lin_recovers': round(fl - res['emb_lin'][0], 4),
    'embattn_lin_recovers': round(fl - res['embattn_lin'][0], 4),
    'emb_lin_frac_of_full': round((fl - res['emb_lin'][0]) / (fl - res['xhat_lin'][0]), 3),
    'inputnorm_gain': round(res['embattn_lin'][0] - res['xhat_lin'][0], 4),
}
print('=== analysis ===', flush=True)
print(f"  linear-read-of-embedding recovers {res['analysis']['emb_lin_frac_of_full']:.1%} of what the full linear cap does", flush=True)
print(f"  input-norm rms(emb+attn0) vs prenorm-linear gap: {res['analysis']['inputnorm_gain']:+.4f}", flush=True)
json.dump(res, open(f'{QK}/qk_block0_source.json', 'w'), indent=1)
print('SAVED', flush=True)
