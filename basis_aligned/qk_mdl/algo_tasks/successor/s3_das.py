"""Step 3: DAS-lite. Learn an r-dim orthonormal residual subspace (r in
{1,4,16}) at the top component's layer (head L8.H3 -> residual stream ENTERING
block 8, i.e. after block 7) whose interchange at the last-element position
(pos 4) moves the prediction to the source sequence's successor.

Base run = clean prompt, source = corrupted twin; patched x4 =
x4 - QQ^T x4 + QQ^T x4_src; success = argmax at pred position == corrupted
answer. Controls: random orthonormal subspace, full-vector swap (ceiling).
Cross-family: weekday-trained subspace applied to month/alphabet pairs."""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl/algo_tasks/successor')
from successor_lib import (HERE, DEV, LAST_POS, PRED_POS, load_model,
                           load_stimuli, pairs_tensors)
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import rope_tables, apply_rot

torch.manual_seed(0)
m, cfg = load_model()
NH, D = cfg['n_head'], cfg['n_embd']
HD, NL = D // NH, cfg['n_layer']
L_PATCH = 8   # edit applied to residual entering block 8

stim = load_stimuli()


@torch.no_grad()
def prefix_state(idx):
    """Run blocks 0..L_PATCH-1; return (x, x0, v1) needed to resume."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,))
    x0, v1 = x, None
    cos, sin = rope_tables(T, HD, idx.device, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    for li in range(L_PATCH):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (D,))

        def qk(lin):
            z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return x, x0, v1


def tail_logits(x, x0, v1, grad=False):
    """Run blocks L_PATCH..NL-1 from cached state; x may carry grad."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        B, T = x.shape[:2]
        cos, sin = rope_tables(T, HD, x.device, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        for li in range(L_PATCH, NL):
            blk = m.transformer.h[li]
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (D,))

            def qk(lin):
                z = F.rms_norm(lin(h).view(B, T, NH, HD), (HD,))
                return apply_rot(z, cosb, sinb)

            v = a.c_v(h).view(B, T, NH, HD)
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (D,)))
        return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30).float()


def get_states(rows_filter=None, family=None, split=None):
    ci, xi, ca, xa, rows = pairs_tensors(stim, split=split, family=family)
    st_c, st_x = [], []
    for i in range(0, len(ci), 8):
        st_c.append(prefix_state(ci[i:i + 8]))
        st_x.append(prefix_state(xi[i:i + 8]))
    xC = torch.cat([s[0] for s in st_c])
    x0C = torch.cat([s[1] for s in st_c])
    v1C = torch.cat([s[2] for s in st_c])
    xX = torch.cat([s[0] for s in st_x])
    src4 = xX[:, LAST_POS, :].clone()          # source (corrupted) resid @ pos4
    return dict(xC=xC, x0C=x0C, v1C=v1C, src4=src4, ca=ca, xa=xa, n=len(ci))


def patched_logits(S, Q, sl, grad=False):
    x = S['xC'][sl].clone()
    x4 = x[:, LAST_POS, :]
    delta = (S['src4'][sl] - x4) @ Q @ Q.T
    x = x.clone()
    x[:, LAST_POS, :] = x4 + delta
    return tail_logits(x, S['x0C'][sl], S['v1C'][sl], grad=grad)


def evaluate(S, Q):
    n = S['n']
    flips, still, mrec = 0, 0, []
    for i in range(0, n, 8):
        sl = slice(i, min(i + 8, n))
        lg = patched_logits(S, Q, sl)
        pred = lg[:, PRED_POS].argmax(-1)
        flips += (pred == S['xa'][sl]).sum().item()
        still += (pred == S['ca'][sl]).sum().item()
        nn = lg.shape[0]
        mrec.append(lg[range(nn), PRED_POS, S['xa'][sl]]
                    - lg[range(nn), PRED_POS, S['ca'][sl]])
    return {'flip_rate': flips / n, 'still_clean': still / n,
            'mean_margin_to_source': torch.cat(mrec).mean().item()}


def train_Q(S_train, r, epochs=250, lr=5e-3, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    M = torch.randn(D, r, generator=g).to(DEV)
    M.requires_grad_(True)
    opt = torch.optim.Adam([M], lr=lr)
    n = S_train['n']
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 8):
            sl = perm[i:i + 8]
            Q, _ = torch.linalg.qr(M)
            lg = patched_logits(S_train, Q, sl, grad=True)
            loss = F.cross_entropy(lg[:, PRED_POS], S_train['xa'][sl])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return torch.linalg.qr(M.detach())[0]


def rand_Q(r, seed=1):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.linalg.qr(torch.randn(D, r, generator=g))[0].to(DEV)


S_an = get_states(split='analysis')
S_ho = get_states(split='heldout')
res = {'site': 'residual entering block 8, last-element position (pos 4)',
       'direction': 'base=clean, source=corrupted; success=argmax->corrupted successor'}

# ceiling: full-vector swap
eye = torch.eye(D, device=DEV)
res['full_swap_heldout'] = evaluate(S_ho, eye)
res['full_swap_analysis'] = evaluate(S_an, eye)
print('full-swap ceiling heldout:', res['full_swap_heldout'], flush=True)

res['learned'], res['random_control'] = {}, {}
for r in [1, 4, 16]:
    Q = train_Q(S_an, r)
    torch.save(Q.cpu(), f'{HERE}/das_Q_all_r{r}.pt')
    res['learned'][r] = {'heldout': evaluate(S_ho, Q),
                         'analysis': evaluate(S_an, Q)}
    ctrl = [evaluate(S_ho, rand_Q(r, seed=s))['flip_rate'] for s in range(5)]
    res['random_control'][r] = {'heldout_flip_rates': ctrl}
    print(f'r={r} heldout {res["learned"][r]["heldout"]} | random ctrl flips {ctrl}',
          flush=True)

# cross-family transfer: train on one family's ANALYSIS pairs, eval everywhere
fams = ['weekday', 'month', 'alphabet']
S_fam_an = {f: get_states(family=f, split='analysis') for f in fams}
S_fam_all = {f: get_states(family=f) for f in fams}
res['cross_family'] = {}
for r in [4, 16]:
    mat = {}
    for ftr in fams:
        Q = train_Q(S_fam_an[ftr], r, epochs=400)
        row = {}
        for fev in fams:
            S = S_fam_all[fev] if fev != ftr else get_states(family=fev, split='heldout')
            tag = 'heldout' if fev == ftr else 'all20'
            row[fev] = {'flip_rate': evaluate(S, Q)['flip_rate'], 'eval': tag}
        mat[ftr] = row
        print(f'r={r} trained-on={ftr}: ' +
              ', '.join(f"{k}={v['flip_rate']:.2f}" for k, v in row.items()),
              flush=True)
    res['cross_family'][r] = mat

# subspace overlap between family-trained subspaces (r=16, fresh seed):
# singular values of Qa^T Qb (1 = aligned dims, 0 = orthogonal)
Qs = {f: train_Q(S_fam_an[f], 16, epochs=400, seed=2) for f in fams}
ov = {}
for a in fams:
    for b in fams:
        if a < b:
            sv = torch.linalg.svdvals(Qs[a].T @ Qs[b])
            ov[f'{a}~{b}'] = [round(x, 3) for x in sv.tolist()]
res['subspace_overlap_svals_r16'] = ov
print('subspace overlap svals:', ov, flush=True)

json.dump(res, open(f'{HERE}/das.json', 'w'), indent=1, default=str)
print('saved das.json', flush=True)
