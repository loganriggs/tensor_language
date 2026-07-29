"""Interpretability-preserving training for the JOINT six-MLP chain (Logan). Tiers:
  T1+T2 (zero/near-zero interpretation cost): per-layer stream-coefficient scalars (init from the
    lambda algebra), gauge rescalers g_l, output biases (D per layer), and DIAGONAL gains on each
    attention stream's named-basis coordinates (576 per layer). ~10.5k params, all scalars/diag/bias.
  T3 (labeled, meaning-mixing): + within-head block-diagonal 64x64 mixing on each attention stream
    (head attribution preserved, coordinates mix within head; foldable). ~2.2M params.
Trained on cooc CE through the frozen model in the joint configuration (all six m-hats substituted,
oracle attention streams -- same design as the measured 72.5% for comparability). Held-out FineWeb
joint dCE vs floor 7.782.
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
FLOOR6 = 7.78249; REF_JOINT = 2.13854
LMAX = 5; PER = 64
BLKS = [m.transformer.h[i] for i in range(LMAX+1)]
WT = {}
for li, blk in enumerate(BLKS):
    WT[li] = (blk.mlp.Left.weight.detach().float(), blk.mlp.Right.weight.detach().float(),
              blk.mlp.Down.weight.detach().float(), blk.mlp.Down_bias.detach().float())
def T_ev(li, u, v):
    Lw, Rw, Dw, _ = WT[li]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)
lam = [(blk.lambdas[0].item(), blk.lambdas[1].item()) for blk in BLKS]
CO = []; cur = {'e': lam[0][0] + lam[0][1]}
for l in range(LMAX+1):
    xpre = dict(cur); xpre[('a', l)] = 1.0; CO.append(xpre)
    nxt = dict(xpre); nxt[('m', l)] = 1.0
    if l < LMAX:
        cur = {k: lam[l+1][0]*v for k, v in nxt.items()}; cur['e'] = cur.get('e', 0.0) + lam[l+1][1]

# bases (identical recipe to qk_chain_deep: PCA-64/head)
accs = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(LMAX+1)]
@torch.no_grad()
def run_all(idx, want_heads=False):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    outs = {'e': x0.reshape(-1, D)}; x = None; v1 = None
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
        aout = a.c_proj(yh.reshape(B, T, -1))
        if li <= LMAX:
            if want_heads: outs[f'ah{li}'] = yh.reshape(-1, NH, HD)
            outs[f'a{li}'] = aout.reshape(-1, D)
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li <= LMAX: outs[f'm{li}'] = mo.reshape(-1, D)
        x = x + mo
        if li == LMAX: break
    return outs
for i in range(0, 64, 8):
    o = run_all(COOC[i:i+8].to(DEV)[:, :128], want_heads=True)
    for li in range(LMAX+1):
        accs[li] += torch.einsum('nhd,nhe->hde', o[f'ah{li}'].double(), o[f'ah{li}'].double())
QB = []
for li in range(LMAX+1):
    cw = BLKS[li].attn.c_proj.weight.detach().float(); cs = []
    for hh in range(NH):
        ev, evec = torch.linalg.eigh(accs[li][hh])
        cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:PER]].float())
    Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); QB.append(Qx)
print("bases ready", flush=True)

# trainable knobs
K6 = 576
class Knobs(torch.nn.Module):
    def __init__(self, tier3=False):
        super().__init__()
        self.coef = torch.nn.ParameterDict()
        for l in range(LMAX+1):
            for k, v0 in CO[l].items():
                key = f"l{l}_{'e' if k=='e' else k[0]+str(k[1])}"
                self.coef[key] = torch.nn.Parameter(torch.tensor(float(v0)))
        self.g = torch.nn.Parameter(torch.ones(LMAX+1))
        self.bias = torch.nn.Parameter(torch.zeros(LMAX+1, D))
        self.diag = torch.nn.Parameter(torch.ones(LMAX+1, K6))
        self.tier3 = tier3
        if tier3:
            eye = torch.eye(PER).unsqueeze(0).unsqueeze(0).repeat(LMAX+1, NH, 1, 1)
            self.mix = torch.nn.Parameter(eye.clone())
    def stream_a(self, l, a_flat):
        z = a_flat @ QB[l]                              # (n, 576) named coords
        z = z * self.diag[l]
        if self.tier3:
            z = torch.einsum('nhk,hkj->nhj', z.view(-1, NH, PER), self.mix[l]).reshape(-1, K6)
        return z @ QB[l].T
    def co(self, l, k):
        key = f"l{l}_{'e' if k=='e' else k[0]+str(k[1])}"
        return self.coef[key]

def chain_k(knobs, x0f, a_list):
    mh = []
    for l in range(LMAX+1):
        co = CO[l]
        xp = knobs.co(l, 'e')*x0f
        for j in range(l+1):
            if ('a', j) in co: xp = xp + knobs.co(l, ('a', j))*knobs.stream_a(j, a_list[j])
            if ('m', j) in co: xp = xp + knobs.co(l, ('m', j))*mh[j]
        r = (xp.pow(2).sum(1)/D) * knobs.g[l]
        mh.append(T_ev(l, xp, xp)/r.unsqueeze(1).clamp_min(1e-6) + WT[l][3] + knobs.bias[l])
    return mh

def joint_forward(knobs, idx):
    B, T2 = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    cache_a = {}; cache_m = {}
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
        x = x + aout; hin = F.rms_norm(x, (D,))
        if li <= LMAX: cache_a[li] = aout.reshape(-1, D)
        if li == LMAX:
            mh = chain_k(knobs, x0.reshape(-1, D), [cache_a[j] for j in range(LMAX+1)])
            delta = sum((mh[l] - cache_m[l]) for l in range(LMAX)).view(B, T2, D)
            x = x + delta.to(x.dtype) + mh[LMAX].view(B, T2, D).to(x.dtype); continue
        mo = blk.mlp(hin)
        if li <= LMAX: cache_m[li] = mo.reshape(-1, D)
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def audit(knobs):
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV)
        lg = joint_forward(knobs, b[:, :-1]).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {'ref_joint_dCE': REF_JOINT, 'ref_joint_frac': round(1-REF_JOINT/FLOOR6, 4), 'floor6': FLOOR6}
for tier, steps in [('T12_scalars_diag_bias', 350), ('T3_plus_headmix', 300)]:
    knobs = Knobs(tier3=(tier.startswith('T3'))).to(DEV)
    d0 = audit(knobs) - SUBBASE
    opt = torch.optim.Adam(knobs.parameters(), lr=2e-3)
    for step in range(steps):
        i = 2400 + np.random.randint(0, 2500); b = COOC[i:i+2].to(DEV)[:, :128]
        lg = joint_forward(knobs, b[:, :-1]).float()
        loss = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0: print(f"[{tier}] step {step} loss {loss.item():.4f}", flush=True)
    d1 = audit(knobs) - SUBBASE
    nparam = sum(p.numel() for p in knobs.parameters())
    res[tier] = {'pre_dCE': round(d0, 5), 'post_dCE': round(d1, 5),
                 'pre_frac': round(1-d0/FLOOR6, 4), 'post_frac': round(1-d1/FLOOR6, 4), 'params': nparam}
    print(f"{tier} ({nparam} params): pre +{d0:.4f} ({1-d0/FLOOR6:.1%}) -> post +{d1:.4f} ({1-d1/FLOOR6:.1%})", flush=True)
    torch.save(knobs.state_dict(), f'{QK}/qk_joint_polish_{tier}.pt')
json.dump(res, open(f'{QK}/qk_joint_polish.json', 'w'), indent=2)
print("QK JOINT POLISH DONE", flush=True)
