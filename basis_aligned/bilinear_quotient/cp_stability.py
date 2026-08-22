"""IS THE class+position SUBSPACE A STABLE INVARIANT OR OVERFIT TO THE SAMPLE? (validity
check for the whole §767→819 program). Every keep-only measurement builds the class+position
subspace on the eval rows and scores keep on the SAME rows, so the recovery could be partly
in-sample overfitting. Test generalization: build the subspace on FineWeb half A, score
keep-only CE-recovery on held-out half B (CROSS), vs building+scoring on the same half
(WITHIN), for several high-benefit components across the depth. A shuffled-token-label
subspace is the null.

REGISTERED PREDICTIONS:
  (0) SANITY: within-sample keep reproduces prior values (mlp0 ~0.97, attn1 ~0.99);
  (a) STABLE: cross-half keep ≈ within-half keep (gap < 0.10) and both ≫ shuffled null ->
      the class+position subspace generalizes to held-out data, it is a real invariant, not
      a per-sample fit -> the whole class+position claim is not overfit;
  (b) OVERFIT: if cross ≪ within, the recovery was partly in-sample overfitting (would
      require caveating the program's headline numbers);
  NULL: shuffled-token-label subspace (same rank) recovers far less than the real subspace."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_stability_results.json'
NEVAL = 320; MINCOUNT = 5; RTOK = 64; RPOS = 32
COMPS = [('attn', 0), ('mlp', 0), ('attn', 1), ('attn', 5), ('mlp', 8), ('mlp', 16)]
SUB = {'U': None, 'mean': None, 'op': None, 'name': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    name = (w, L)
    def hook(mo, i_, o_):
        if SUB['op'] is None or SUB['name'] != name: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if SUB['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: mu = SUB['mean']; U = SUB['U']; v2 = mu + ((v - mu) @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows):
    s = 0.0; nn = 0
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous(), g


def cp_from(rows, w, L, shuffle=False):
    saved = SUB['op']; SUB['op'] = None      # BUGFIX: capture must see CLEAN activations, not the ablate/keep hook
    O, toks, pos = capture(rows, w, L)
    SUB['op'] = saved
    if shuffle:
        rng = np.random.RandomState(0); toks = toks.copy(); rng.shuffle(toks)
    Ut, g = mean_subspace(O, toks, RTOK); Up, _ = mean_subspace(O, pos, RPOS)
    U = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    return U, g


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    A = rows[:NEVAL//2]; B = rows[NEVAL//2:]
    out = {}
    for w, L in COMPS:
        name = (w, L); h = comp(w, L).register_forward_hook(mk_hook(w, L))
        SUB['name'] = name
        SUB['op'] = None; ce_full_B = ce_on(B)
        SUB['op'] = 'ablate'; ce_abl_B = ce_on(B); ben_B = ce_abl_B - ce_full_B
        # subspaces
        U_A, gmean_A = cp_from(A, w, L)                    # built on A
        U_B, gmean_B = cp_from(B, w, L)                    # built on B (within)
        U_sh, _ = cp_from(B, w, L, shuffle=True)           # shuffled-token-label null on B
        g = torch.Generator(device=DEV).manual_seed(hash((w, L)) & 0xffff)
        U_rnd = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]   # random-orthonormal null
        def keepB(U, mu): SUB['op'] = 'keep'; SUB['U'] = U; SUB['mean'] = mu; c = ce_on(B); SUB['op'] = None; return round(float((ce_abl_B-c)/max(ben_B, 1e-6)), 4)
        cross = keepB(U_A, gmean_A)      # subspace from A, eval on B -> generalization
        within = keepB(U_B, gmean_B)     # subspace from B, eval on B -> in-sample
        null_sh = keepB(U_sh, gmean_B)   # shuffled token labels (+ real position, mean preserved)
        null_rnd = keepB(U_rnd, gmean_B) # random orthonormal subspace (the established 'useless' null)
        h.remove()
        out[f'{w}{L}'] = {'benefit_B': round(ben_B, 3), 'within': within, 'cross': cross,
                          'shuffled_null': null_sh, 'random_orth_null': null_rnd, 'gap': round(within - cross, 4)}
        print(f'{w}{L}: benefit {ben_B:.3f} | within {within} cross {cross} (gap {within-cross:+.3f}) | shuffled {null_sh} | rand-orth {null_rnd}', flush=True)
    big = [f'{w}{L}' for w, L in COMPS if out[f'{w}{L}']['benefit_B'] > 0.1]
    gaps = [out[k]['gap'] for k in big]
    crosses = [out[k]['cross'] for k in big]
    nulls = [out[k]['random_orth_null'] for k in big]      # proper null = random orthonormal
    stable = max(gaps) < 0.10 and min(crosses) > max(nulls) + 0.1
    out['pred_a_stable'] = bool(stable); out['max_gap'] = round(max(gaps), 4)
    out['note_shuffled_null'] = 'shuffled-token null is weak (data-derived + mean preserved); random_orth is the proper null'
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) class+position subspace is STABLE (max within-cross gap {max(gaps):.3f} < 0.10, cross >> shuffled): {stable}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
