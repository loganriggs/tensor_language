"""THE FULLY-NAMED MODEL TEST. Once every attention output is projected onto its named per-layer
basis (PCA-64/head, 576 dims), the entire residual stream is spanned by {embedding} + named
attention coordinates, and every MLP output is an exact analytic function of named content (the
quadratics are exact given their input; gauges carried by the model itself). So the complete
named-bottleneck substitution = run the model with aout -> Proj_QB(aout) at every layer <= L.
Progressive depth L in {5, 11, 17}; width PER in {32, 64}. dCE vs base measures how much of the
WHOLE model's function flows through the named attention coordinates. Historical reference:
windowed-D global compressed explanation was +0.059.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[:200]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
SUBBASE = json.load(open(f'{QK}/qk_completeness_ledger.json'))['subset_base']

# per-layer per-head covariances over all 18 layers
accs = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(NL)]
@torch.no_grad()
def collect(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        accs[li] += torch.einsum('nhd,nhe->hde', yh.reshape(-1, NH, HD).double(), yh.reshape(-1, NH, HD).double())
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
for i in range(0, 64, 8):
    collect(COOC[i:i+8].to(DEV)[:, :128])

def bases(PER):
    Q = []
    for li in range(NL):
        cw = m.transformer.h[li].attn.c_proj.weight.detach().float(); cs = []
        for hh in range(NH):
            ev, evec = torch.linalg.eigh(accs[li][hh])
            cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:PER]].float())
        Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); Q.append(Qx)
    return Q
QB = {PER: bases(PER) for PER in (32, 64)}
print("bases ready", flush=True)


@torch.no_grad()
def audit(PER, L):
    Q = QB[PER]; tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV); idx = b[:, :-1]; B, T2 = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
        cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; a = blk.attn
            x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T2, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
            aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
            if li <= L:
                fa = aout.reshape(-1, D)
                aout = ((fa @ Q[li]) @ Q[li].T).view(B, T2, D).to(aout.dtype)
            x = x + aout; x = x + blk.mlp(F.rms_norm(x, (D,)))
        lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {}
for PER in (32, 64):
    for L in (5, 11, 17):
        d = audit(PER, L) - SUBBASE
        res[f'PER{PER}_L{L}'] = round(d, 5)
        print(f"named bottleneck: width {PER}/head, layers 0-{L}: dCE +{d:.5f}", flush=True)
json.dump(res, open(f'{QK}/qk_named_bottleneck.json', 'w'), indent=2)
print("QK NAMED BOTTLENECK DONE", flush=True)
