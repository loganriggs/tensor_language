"""When does the backward (output-relevance) metric win? (Logan 2026-07-21)
The method-E null said: backward = forward when quantization error behaves as
noise the model filters uniformly. The positive prediction: on an object with
ADVERSARIAL error consumption — a few directions that matter enormously for the
output among many that are harmless — a backward (Fisher-weighted) reduction
should beat plain forward least-squares. This constructs that regime cleanly and
tests it, turning the null into a characterization of *when direction matters*.

Object: the layer-0 value table VT[:, head] (V x 128). We rank-reduce it (keep a
low-dim subspace) two ways and audit real ΔCE:
  FORWARD: SVD in raw activation space (keep top-r variance directions).
  BACKWARD: SVD in the OUTPUT-GRADIENT-whitened space (keep the top-r directions
    of the value that most affect the loss, from backprop through the model) —
    i.e. the same rank budget but chosen by behavioural importance, not variance.
If forward >= backward: null holds (this object is noise-filtered). If backward <
forward at small r: direction matters here, validating the prediction.
The value table is a good candidate because content is behaviourally SENSITIVE
(carriage needs identity — small activation directions can be load-bearing),
unlike the noise-filtered stream tables method E tried."""
import json
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 128, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
VT = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
HEAD = 0
X = VT[:, HEAD].to(DEV)                       # (V, 128)


def reference_forward_grad(idx, vgrad_head):
    """forward with grad on layer-0 head-HEAD value; returns loss for backprop."""
    B, T = idx.shape
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
    x = x.detach()
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        if li == 0:
            v = v.clone()
            vh = v[:, :, HEAD].detach().requires_grad_(True)
            vgrad_head.append(vh)
            v = torch.cat([v[:, :, :HEAD], vh.unsqueeze(2), v[:, :, HEAD + 1:]], 2)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    xf = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(xf) / 30)


# ---- backward importance: E[(dLoss/d v_head)^2] per dim, over TRAIN ----
G = torch.zeros(HD, device=DEV)
nb = 0
for i in range(0, 96, 2):
    idx = TRAIN[i:i + 2, :-1].to(DEV)
    tgt = TRAIN[i:i + 2, 1:].to(DEV)
    holder = []
    lg = reference_forward_grad(idx, holder)
    loss = F.cross_entropy(lg.float().reshape(-1, V), tgt.reshape(-1))
    loss.backward()
    G += holder[0].grad.float().pow(2).sum((0, 1))
    nb += 1
G = (G / nb).clamp_min(1e-12)                 # (HD,) diagonal output-importance
print('output-gradient importance built', flush=True)

# forward whitening = identity; backward whitening = diag(sqrt(G))
Wb = G.sqrt()

from tier2_model import reference_forward


@torch.no_grad()
def ce(vt_head):
    vt = VT.clone().to(DEV); vt[:, HEAD] = vt_head
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        x = m.transformer.wte(b[:, :-1]); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        B, T = x.shape[0], x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
            if li == 0:
                v = a.c_v(h).view(B, T, NH, HD).clone()
                v[:, :, HEAD] = vt[b[:, :-1], HEAD].to(x.dtype)
            else:
                v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        lg = 30 * torch.tanh(m.lm_head(x) / 30)
        tot += F.cross_entropy(lg.float().reshape(-1, V), b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce(X)
mu = X.mean(0)
res = {'baseline_ce': CE0, 'forward': {}, 'backward': {}}
print(f'baseline {CE0:.4f}', flush=True)
Xc = (X - mu).double()
for r in (4, 8, 16, 32, 64):
    # FORWARD SVD (raw activation space)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    Xf = ((U[:, :r] * S[:r]) @ Vh[:r]).float() + mu
    res['forward'][f'r={r}'] = round(ce(Xf) - CE0, 4)
    # BACKWARD SVD (output-gradient-whitened): scale dims by Wb, SVD, unscale
    Xw = Xc * Wb.double()
    Uw, Sw, Vhw = torch.linalg.svd(Xw, full_matrices=False)
    Xb = (((Uw[:, :r] * Sw[:r]) @ Vhw[:r]) / Wb.double()).float() + mu
    res['backward'][f'r={r}'] = round(ce(Xb) - CE0, 4)
    print(f'r={r}: forward {res["forward"][f"r={r}"]:+.4f}  backward {res["backward"][f"r={r}"]:+.4f}  '
          f'({"backward wins" if res["backward"][f"r={r}"] < res["forward"][f"r={r}"] else "forward wins"})', flush=True)
    json.dump(res, open(f'{QK}/backward_when_wins.json', 'w'), indent=2)
print('backward when wins done', flush=True)
