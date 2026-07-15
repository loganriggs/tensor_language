"""CE-train the layer-0 OV vq tables of bilin18 (v-side analog of the QK
codebook training): assignments frozen from k-means, centroid v-tables trained
through the frozen model on pile-10k CE (bf16, clipped). k in {1024, 4096}."""
import json, math, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('bilin18')
m.to(torch.bfloat16)
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D, V = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd'], cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 128, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:].to(DEV)
E = m.transformer.wte.weight.detach().float()
VT = (F.rms_norm(E, (D,)) @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)

def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            xx = X[i:i + 8192]
            assign[i:i + 8192] = ((xx**2).sum(1, keepdim=True) - 2*xx@C.T + (C**2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X); cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn/cnt.clamp(min=1)[:, None])
    return C, assign

def forward(tokens, tabs=None, assigns=None):
    x = m.transformer.wte(tokens); x = F.rms_norm(x, (x.size(-1),)); x0 = x; v1 = None
    B, T = tokens.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        cos, sin = rope_tables(T, HD, tokens.device, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        def qk(lin):
            z = lin(h).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cosr, sinr)
        if li == 0 and tabs is not None:
            v = torch.stack([tabs[hh][assigns[hh][tokens]] for hh in range(NH)], 2).to(x.dtype)
        else:
            v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb)*v + a.lamb*v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30*torch.tanh(m.lm_head(x)/30)

@torch.no_grad()
def audit(tabs=None, assigns=None):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i+4].to(DEV)
        logits = forward(b[:, :-1], tabs, assigns).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1)).item()*b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot/n

CE0 = audit()
print(f'baseline {CE0:.4f}')
out = {'baseline_ce': CE0}
for K in [1024, 4096]:
    tabs, assigns = [], []
    for hh in range(NH):
        C, assign = kmeans(VT[:, hh].contiguous(), K)
        tabs.append(C.clone().requires_grad_(True)); assigns.append(assign)
    before = audit(tabs, assigns) - CE0
    opt = torch.optim.Adam(tabs, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1200)
    g = torch.Generator(); g.manual_seed(0)
    for step in range(1200):
        b = TRAIN[torch.randint(0, len(TRAIN), (4,), generator=g)]
        logits = forward(b[:, :-1], tabs, assigns).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tabs, 1.0)
        opt.step(); sched.step()
        if step % 300 == 0: print(f'  K={K} step {step} CE {loss.item():.4f}', flush=True)
    after = audit(tabs, assigns) - CE0
    dl = NH*(32*K*HD + V*math.log2(K))
    out[f'vq{K}'] = {'dce_before': before, 'dce_after': after, 'ratio': dl/(32*V*HD*NH)}
    print(f'OV vq{K}: dCE {before:+.4f} -> {after:+.4f} (ratio {dl/(32*V*HD*NH):.3f})', flush=True)
json.dump(out, open('/workspace/tensor_language/basis_aligned/qk_mdl/ov_ce_trained.json', 'w'), indent=2)
print('ov ce done')
