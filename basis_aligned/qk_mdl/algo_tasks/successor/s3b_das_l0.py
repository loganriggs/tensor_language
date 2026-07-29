"""Step 3b: DAS-lite at the CORRECTED site. The literal spec site (residual
entering block 8, pos 4) has a 0% full-swap ceiling because head L8.H3's value
payload is v1 (computed at layer 0; lamb(L8)=4.0 -> value = -3*v_L8 + 4*v1),
which a layer-8 residual patch cannot touch. The payload enters at layer 0, so
we learn the subspace of the EMBEDDING STREAM (rms-normed wte, input to block
0) at the last-element position. Full swap here == the corrupted run
(prompts differ only at pos 4), giving a behavioral ceiling.
Same protocol: r in {1,4,16}, QR-parameterized, trained on analysis pairs,
held-out flip rate vs random control, cross-family transfer, r=16 overlap."""
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

stim = load_stimuli()


def full_logits(x, grad=False):
    """Run all blocks from a (possibly patched) rms-normed embedding x.
    x0 = x (the patch is upstream of the skip anchor); v1 computed at block 0."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        B, T = x.shape[:2]
        x0, v1 = x, None
        cos, sin = rope_tables(T, HD, x.device, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        for li in range(NL):
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
        return 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30).float()


@torch.no_grad()
def emb(idx):
    return F.rms_norm(m.transformer.wte(idx), (D,))


def get_states(family=None, split=None):
    ci, xi, ca, xa, rows = pairs_tensors(stim, split=split, family=family)
    return dict(eC=emb(ci), src4=emb(xi)[:, LAST_POS, :].clone(),
                ca=ca, xa=xa, n=len(ci))


def patched_logits(S, Q, sl, grad=False):
    x = S['eC'][sl].clone()
    x4 = x[:, LAST_POS, :]
    x = x.clone()
    x[:, LAST_POS, :] = x4 + (S['src4'][sl] - x4) @ Q @ Q.T
    return full_logits(x, grad=grad)


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
res = {'site': 'rms-normed embedding stream (input to block 0), pos 4',
       'direction': 'base=clean, source=corrupted; success=argmax->corrupted successor',
       'why_moved': 'layer-8 site ceiling was 0: payload rides v1 skip from layer 0'}

eye = torch.eye(D, device=DEV)
res['full_swap_heldout'] = evaluate(S_ho, eye)
res['full_swap_analysis'] = evaluate(S_an, eye)
print('full-swap ceiling heldout:', res['full_swap_heldout'], flush=True)

res['learned'], res['random_control'] = {}, {}
for r in [1, 4, 16]:
    Q = train_Q(S_an, r)
    torch.save(Q.cpu(), f'{HERE}/das_l0_Q_all_r{r}.pt')
    res['learned'][r] = {'heldout': evaluate(S_ho, Q),
                         'analysis': evaluate(S_an, Q)}
    ctrl = [evaluate(S_ho, rand_Q(r, seed=s))['flip_rate'] for s in range(5)]
    res['random_control'][r] = {'heldout_flip_rates': ctrl}
    print(f'r={r} heldout {res["learned"][r]["heldout"]} | random ctrl {ctrl}',
          flush=True)

fams = ['weekday', 'month', 'alphabet']
S_fam_an = {f: get_states(family=f, split='analysis') for f in fams}
S_fam_ho = {f: get_states(family=f, split='heldout') for f in fams}
S_fam_all = {f: get_states(family=f) for f in fams}
res['cross_family'] = {}
for r in [4, 16]:
    mat = {}
    for ftr in fams:
        Q = train_Q(S_fam_an[ftr], r, epochs=400)
        row = {}
        for fev in fams:
            S = S_fam_ho[fev] if fev == ftr else S_fam_all[fev]
            row[fev] = {'flip_rate': evaluate(S, Q)['flip_rate'],
                        'eval': 'heldout' if fev == ftr else 'all20'}
        mat[ftr] = row
        print(f'r={r} trained-on={ftr}: ' +
              ', '.join(f"{k}={v['flip_rate']:.2f}" for k, v in row.items()),
              flush=True)
    res['cross_family'][r] = mat

Qs = {f: train_Q(S_fam_an[f], 16, epochs=400, seed=2) for f in fams}
ov = {}
for a in fams:
    for b in fams:
        if a < b:
            sv = torch.linalg.svdvals(Qs[a].T @ Qs[b])
            ov[f'{a}~{b}'] = [round(x, 3) for x in sv.tolist()]
res['subspace_overlap_svals_r16'] = ov
print('subspace overlap svals:', ov, flush=True)

json.dump(res, open(f'{HERE}/das_l0.json', 'w'), indent=1, default=str)
print('saved das_l0.json', flush=True)
