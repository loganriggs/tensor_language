"""SR-2 MECHANISM TEST: is the composed-rulebook damage on sqrd12 caused by
denominator coupling? Arm: mask the numerator but keep ORIGINAL row sums
(denominator as if tail present). If composed cost drops toward bilin18-like,
the coupling mechanism is confirmed; if not, refuted. Original: SQRD12 RULEBOOKS: is block-sparse same-kind selection model-general?
Class map = kmeans-256 on sqrd12 own embedding; block energy per (head,
class-pair); keep-top-B ladder at L3/L8 single + ALL-layers composed.
Row-normalized pattern: mask s^2 entries, then renormalize over kept.
Original: Windowed-D transfer test on sqrd12 (162M, single QK branch, row-normalized
squared attention — the model that resisted score-space compression ~15x).
QK-reads windowed only, W in {2,4,6}. Original: Logan's method D composed, window form (guided by SI-1 + C-1): at EVERY
layer L>=1, the QK read (q,k inputs only; v/OV and the residual stay live)
replaces streams CREATED more than W layers back with their cond-mean-by-token
tables (estimated once at creation, rescaled analytically by the lambda
products); the embedding stream is exactly token-determined so it is free.
Streams inside the window are the PATCHED model's own live streams, so errors
can only chain W layers deep instead of 17. Arms: W=2, W=3, W=0 (all tabled —
composed control, expect wall-scale), plus per-arm sanity vs c_window.
No training anywhere."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/sqrd12_coupling.json'
m, cfg = load_elriggs('sqrd12')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = cfg['n_layer']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
AUD = ALL[4:20]
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (cfg['n_embd'],)).to(DEV)
gK = torch.Generator(); gK.manual_seed(3)
C0 = E_hat[torch.randperm(V, generator=gK)[:256].to(DEV)].clone()
for _ in range(10):
    a_ = torch.empty(V, dtype=torch.long, device=DEV)
    for i in range(0, V, 4096):
        xx = E_hat[i:i + 4096]
        a_[i:i + 4096] = ((xx*xx).sum(1,True) - 2*xx@C0.T + (C0*C0).sum(1)[None]).argmin(1)
    Cn = torch.zeros_like(C0); c2 = torch.zeros(256, device=DEV)
    Cn.index_add_(0, a_, E_hat); c2.index_add_(0, a_, torch.ones(256 if False else V, device=DEV))
    nz = c2 > 0; C0[nz] = Cn[nz]/c2[nz][:,None]
CLS = a_
print('class map built', flush=True)
NL = cfg['n_layer']

energy = torch.zeros(NL, NH, 256, 256, device=DEV)


@torch.no_grad()
def sweep(keep=None, collect=False):
    """keep: (NL,NH,256,256) bool or None. Returns mean CE."""
    tot, n = 0.0, 0
    for i in range(0, len(AUD), 4):
        b = AUD[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        cls_pos = CLS[idx]
        code = cls_pos[:, :, None] * 256 + cls_pos[:, None, :]
        x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        tri = mask
        codef = code[:, tri].reshape(-1)
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k = qn(a.c_q), qn(a.c_k)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            p2 = s1.square().masked_fill(~mask, 0.0)
            if collect:
                for hh in range(NH):
                    pf = (p2[:, hh] / p2[:, hh].sum(-1, keepdim=True).clamp_min(1e-9))[:, tri].reshape(-1).float()
                    energy[li, hh].view(-1).index_add_(0, codef, pf)
            denom_orig = p2.sum(-1, keepdim=True).clamp_min(1e-9)
            if keep is not None:
                kq = cls_pos[:, :, None].expand(B, T, T)
                kk = cls_pos[:, None, :].expand(B, T, T)
                for hh in range(NH):
                    kmh = keep[li, hh][kq.reshape(-1), kk.reshape(-1)].view(B, T, T)
                    p2[:, hh] = p2[:, hh] * kmh
            if keep is not None and RAW_DENOM:
                pat = p2 / denom_orig
            else:
                pat = p2 / p2.sum(-1, keepdim=True).clamp_min(1e-9)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot / n


RAW_DENOM = False
base = sweep(collect=True)
res = {'baseline': base, 'arms': {}}
print(f'baseline {base:.4f} (energy collected)', flush=True)


def kmask(Bk, layers):
    keep = torch.ones(NL, NH, 256, 256, dtype=torch.bool, device=DEV)
    for li in layers:
        for hh in range(NH):
            km = torch.zeros(256 * 256, dtype=torch.bool, device=DEV)
            km[energy[li, hh].view(-1).topk(Bk).indices] = True
            keep[li, hh] = km.view(256, 256)
    return keep


for Bk in (2048, 512):
    for rd in (False, True):
        globals()['RAW_DENOM'] = rd
        tag = 'RAW-denominator' if rd else 'renormalized'
        d = sweep(keep=kmask(Bk, list(range(NL)))) - base
        res['arms'][f'ALL layers top-{Bk} {tag}'] = round(d, 4)
        print(f'ALL layers top-{Bk} {tag}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)

print('sqrd12 coupling done', flush=True)
