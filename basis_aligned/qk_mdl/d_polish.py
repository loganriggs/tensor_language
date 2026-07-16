"""CE-polish of the flagship: W=4 windowed-D with vq1024 stream tables;
train ONLY the vq codebook values (atoms) of streams created at layers 0-5
(12 streams x 1024 atoms x 1024 dims = 12.6M floats — protocol-sized for
3.15M train tokens), assignments and everything else frozen, model frozen.
Upper-stream tables stay at their untrained vq1024 values.
Start point (untrained, D-2): +0.094. Residual is window-boundary error
(D-3/D-4 ruled out noise and region)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/d_polish.json'
W = 4
KATOMS = 1024
TRAIN_STREAM_L = 5          # train codebooks of streams created at layers <= this
STEPS = 3000
BATCH = 2
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:1044]      # train tokens = the good early 524k
AUDIT_LATE = ALL[-16:]
print(f'train tokens {TRAIN.numel()/1e6:.2f}M', flush=True)

RAW = torch.load(f'{QK}/stream_tables.pt')


def created_layer(nm):
    return int(nm[4:]) if nm.startswith('attn') else int(nm[3:])


# vq1024 each stream table; trainable atoms for bottom streams
assigns, atoms, fixed_rows = {}, {}, {}
for nm, t in RAW.items():
    X = t.float().to(DEV)
    g = torch.Generator(); g.manual_seed(hash(nm) % 2**31)
    C = X[torch.randperm(len(X), generator=g)[:KATOMS].to(DEV)].clone()
    for _ in range(10):
        a_ = torch.empty(len(X), dtype=torch.long, device=DEV)
        for i in range(0, len(X), 2048):
            xx = X[i:i + 2048]
            a_[i:i + 2048] = ((xx * xx).sum(1, True) - 2 * xx @ C.T
                              + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        c2 = torch.zeros(KATOMS, device=DEV)
        Cn.index_add_(0, a_, X)
        c2.index_add_(0, a_, torch.ones(len(X), device=DEV))
        nz = c2 > 0
        C[nz] = Cn[nz] / c2[nz][:, None]
    assigns[nm] = a_
    if created_layer(nm) <= TRAIN_STREAM_L:
        atoms[nm] = C.requires_grad_(True)
    else:
        fixed_rows[nm] = C[a_].half().cpu()
    del X
    torch.cuda.empty_cache()
print(f'vq built; trainable atoms: {sum(a.numel() for a in atoms.values())/1e6:.1f}M '
      f'({len(atoms)} streams), fixed: {len(fixed_rows)}', flush=True)


def forward(idx, grad=False):
    B, T = idx.shape
    idx_cpu = idx.cpu()
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    streams, tabs = {}, {}
    for li, blk in enumerate(m.transformer.h):
        lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
        x = lam0 * x + lam1 * x0
        for nm in streams:
            streams[nm] = lam0 * streams[nm]
            tabs[nm] = lam0 * tabs[nm]
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        h_qk = h
        if li >= 1:
            old = [nm for nm in streams if created_layer(nm) < li - W]
            if old:
                xp = x
                for nm in old:
                    xp = xp - streams[nm] + tabs[nm]
                h_qk = F.rms_norm(xp, (xp.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h_qk).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn_out
        streams[f'attn{li}'] = attn_out
        nm = f'attn{li}'
        tabs[nm] = (atoms[nm][assigns[nm][idx]] if nm in atoms
                    else fixed_rows[nm][idx_cpu].to(DEV, x.dtype))
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        streams[f'mlp{li}'] = mlp_out
        nm = f'mlp{li}'
        tabs[nm] = (atoms[nm][assigns[nm][idx]] if nm in atoms
                    else fixed_rows[nm][idx_cpu].to(DEV, x.dtype))
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30)


@torch.no_grad()
def audit(aud):
    tot, n = 0.0, 0
    for i in range(0, len(aud), 4):
        b = aud[i:i + 4].to(DEV)
        logits = forward(b[:, :-1]).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


@torch.no_grad()
def base_ce(aud):
    tot, n = 0.0, 0
    for i in range(0, len(aud), 4):
        b = aud[i:i + 4].to(DEV)
        logits = reference_forward(m, b[:, :-1], 'bf16').float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


base = base_ce(AUDIT)
base_l = base_ce(AUDIT_LATE)
d0 = audit(AUDIT) - base
print(f'baseline {base:.4f}; W={W} vq{KATOMS} L2-fit dCE {d0:+.4f} (sanity vs +0.094)', flush=True)
res = {'baseline': base, 'l2fit': d0, 'checkpoints': {}}

params = list(atoms.values())
opt = torch.optim.Adam(params, lr=5e-4)
g = torch.Generator(); g.manual_seed(1)
for step in range(STEPS):
    b = TRAIN[torch.randint(0, len(TRAIN), (BATCH,), generator=g)].to(DEV)
    logits = forward(b[:, :-1], grad=True).float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 300 == 0:
        print(f'  step {step} CE {loss.item():.4f}', flush=True)
    if step in (1000, 2000):
        d = audit(AUDIT) - base
        res['checkpoints'][step] = d
        print(f'  held-out @{step}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
dT = audit(AUDIT) - base
dL = audit(AUDIT_LATE) - base_l
res['ce_polished'] = dT
res['ce_polished_late_audit'] = dL
print(f'D-POLISH W={W} vq{KATOMS}: dCE {dT:+.4f} (late audit {dL:+.4f})', flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
torch.save({nm: a.detach().cpu() for nm, a in atoms.items()}, f'{QK}/d_polish_atoms.pt')
print('d polish done', flush=True)
