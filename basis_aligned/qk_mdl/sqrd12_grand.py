"""sqrd12 grand-combined analog: QK vq256 + OV sparse (512x16), jointly
CE-trained with the tick-18 protocol (2.1M train tokens). Single branch;
ReLU^2 MLP has no bilinear block split, so MLP is untouched. Contrast column
for the flagship: the less-compressible model under the same treatment."""
import json, math, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from tier2_folding import branch_factors

torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('sqrd12')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:].to(DEV)
E = m.transformer.wte.weight.detach().float()
EH = F.rms_norm(E, (D,))
VT = (EH @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
qh_all, kh_all = branch_factors(m, 1, dtype=torch.float32)

def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 4096):
            xx = X[i:i + 4096]
            assign[i:i + 4096] = ((xx**2).sum(1, keepdim=True) - 2*xx@C.T + (C**2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X); cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign

def train_topk_dict(X, n, k, steps=3000, batch=8192, lr=3e-3, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = X.mean(0).clone()
    for t in (Dm, We, b): t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    for _ in range(steps):
        x = X[torch.randint(0, len(X), (batch,), device=DEV)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        _, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
        loss = (((b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)) - x)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        z = (X - b) @ We.T
        _, idx = z.abs().topk(k, dim=1)
        coeff = torch.gather(z, 1, idx)
    return Dn, idx, coeff.detach(), b.detach()

print('building codebooks...')
QK = {'q': [], 'k': [], 'a': []}
for hh in range(NH):
    C, a_ = kmeans(torch.cat([qh_all[:, hh], kh_all[:, hh]], 1), 256)
    QK['q'].append(C[:, :HD].clone()); QK['k'].append(C[:, HD:].clone()); QK['a'].append(a_)
OV = {'D': [], 'sup': [], 'cf': [], 'b': []}
for hh in range(NH):
    Dn, idx, coeff, b = train_topk_dict(VT[:, hh].contiguous(), 512, 16)
    OV['D'].append(Dn); OV['sup'].append(idx); OV['cf'].append(coeff); OV['b'].append(b)
print('built')

def forward(tokens, use=('qk', 'ov')):
    x = m.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    B, T = tokens.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        cos, sin = rope_tables(T, HD, tokens.device, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        def qkf(lin):
            z = lin(h).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cosr, sinr)
        if li == 0 and 'ov' in use:
            v = torch.stack([OV['b'][hh] + (OV['cf'][hh][tokens].unsqueeze(-1)
                             * OV['D'][hh][OV['sup'][hh][tokens]]).sum(-2)
                             for hh in range(NH)], 2).to(x.dtype)
        else:
            v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb)*v + a.lamb*v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        if li == 0 and 'qk' in use:
            Fq = torch.stack([QK['q'][hh][QK['a'][hh][tokens]] for hh in range(NH)], 2)
            Fk = torch.stack([QK['k'][hh][QK['a'][hh][tokens]] for hh in range(NH)], 2)
            d = HD // 2
            cs, sn = rope_tables(T, HD, tokens.device, torch.float32, 'bf16')
            cosD = torch.einsum('if,jf->ijf', cs, cs) + torch.einsum('if,jf->ijf', sn, sn)
            sinD = torch.einsum('if,jf->ijf', sn, cs) - torch.einsum('if,jf->ijf', cs, sn)
            qa, qb = Fq[..., :d], Fq[..., d:]; ka, kb = Fk[..., :d], Fk[..., d:]
            s = (torch.einsum('bihf,bjhf,ijf->bhij', qa, ka, cosD)
                 + torch.einsum('bihf,bjhf,ijf->bhij', qb, kb, cosD)
                 + torch.einsum('bihf,bjhf,ijf->bhij', qb, ka, sinD)
                 - torch.einsum('bihf,bjhf,ijf->bhij', qa, kb, sinD)).to(x.dtype) / HD
        else:
            q, k = qkf(a.c_q), qkf(a.c_k)
            s = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        pat = s.square().masked_fill(~mask, 0.0)
        pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30*torch.tanh(m.lm_head(x)/30)

@torch.no_grad()
def ce(use=('qk', 'ov'), batch=8):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), batch):
        b = AUDIT[i:i + batch].to(DEV)
        logits = forward(b[:, :-1], use).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1)).item()*b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n

CE0 = ce(use=())
results = {'baseline_ce': CE0}
for use in [('qk',), ('ov',), ('qk', 'ov')]:
    d = ce(use) - CE0
    results['l2_' + '+'.join(use)] = d
    print(f'L2-fit {"+".join(use):8s}: dCE {d:+.4f}', flush=True)

m.to(torch.bfloat16)
for p in m.parameters(): p.requires_grad_(False)
params = []
for lst in (QK['q'], QK['k'], OV['D'], OV['cf'], OV['b']):
    for t in lst:
        t.requires_grad_(True); params.append(t)
print(f'joint finetune: {sum(p.numel() for p in params)/1e6:.1f}M trainable')
opt = torch.optim.Adam(params, lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4500)
g = torch.Generator(); g.manual_seed(0)
for step in range(4500):
    b = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)]
    logits = forward(b[:, :-1]).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % 900 == 0: print(f'  step {step} CE {loss.item():.4f}', flush=True)
CE0b = ce(use=())
d = ce() - CE0b
results['ce_trained_grand'] = d
print(f'SQRD12 GRAND: dCE {d:+.4f}')
json.dump(results, open('sqrd12_grand.json', 'w'), indent=2)
print('sqrd12 grand done')
