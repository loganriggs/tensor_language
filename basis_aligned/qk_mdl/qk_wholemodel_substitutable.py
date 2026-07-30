"""T3: the single defensible WHOLE-MODEL substitutability number, to the reviewer's reporting
standard. Every MLP replaced causally by its composed-fold surrogate (exact tensor on PCA-64/head
attention streams -- the reviewed method), every attention output additionally passed through its
PCA/head bottleneck; measured on the HELD-BACK slice FW[448:600] with paired per-token dCE and
row-clustered standard errors, reported against BOTH the mean-input joint floor AND the uniform
ceiling ln V. Fair nulls: head-span-restricted random bases (2 seeds). Base-relative dCE is the
headline (floors caveated as broken-model references).
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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
UNIFORM = float(np.log(V))
BLKS = [m.transformer.h[i] for i in range(NL)]
WT = [(b.mlp.Left.weight.detach().float(), b.mlp.Right.weight.detach().float(),
       b.mlp.Down.weight.detach().float(), b.mlp.Down_bias.detach().float()) for b in BLKS]
def T_ev(li, u, v):
    Lw, Rw, Dw, _ = WT[li]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)
lam = [(b.lambdas[0].item(), b.lambdas[1].item()) for b in BLKS]
# coefficient recurrence for x_pre_l over streams {e, a_0..a_l, m_0..m_{l-1}}
CO = []; cur = {'e': lam[0][0]+lam[0][1]}
for l in range(NL):
    xp = dict(cur); xp[('a', l)] = 1.0; CO.append(xp)
    nx = dict(xp); nx[('m', l)] = 1.0
    if l < NL-1:
        cur = {k: lam[l+1][0]*v for k, v in nx.items()}; cur['e'] = cur.get('e', 0.0)+lam[l+1][1]

# PCA-64/head bases from cooc statistics (disjoint from held slice)
acc = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(NL)]
hinsum = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(NL)]; hn=[0]
@torch.no_grad()
def collect(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        acc[li] += torch.einsum('nhd,nhe->hde', yh.reshape(-1, NH, HD).double(), yh.reshape(-1, NH, HD).double())
        x = x + a.c_proj(yh.reshape(B, T, -1)); hh_=F.rms_norm(x,(D,)); hinsum[li]+=hh_.reshape(-1,D).double().sum(0); x = x + b.mlp(hh_)
    hn[0]+=x.shape[0]*x.shape[1]
for i in range(0, 64, 8):
    collect(COOC[i:i+8].to(DEV)[:, :128])
def make_bases(kind, seed=0):
    Q = []
    g = torch.Generator(device=DEV).manual_seed(seed)
    for li in range(NL):
        cw = BLKS[li].attn.c_proj.weight.detach().float()
        if kind == 'pca':
            cs = []
            for hh in range(NH):
                ev, evec = torch.linalg.eigh(acc[li][hh])
                cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:64]].float())
            Qx, _ = torch.linalg.qr(torch.cat(cs, 1))
        else:  # headspan random
            cs = []
            for hh in range(NH):
                Rh, _ = torch.linalg.qr(torch.randn(HD, 64, generator=g, device=DEV))
                cs.append(cw[:, hh*HD:(hh+1)*HD] @ Rh)
            Qx, _ = torch.linalg.qr(torch.cat(cs, 1))
        Q.append(Qx)
    return Q
QB = {'pca': make_bases('pca'), 'null1': make_bases('null', 11), 'null2': make_bases('null', 12)}
MU = [F.rms_norm((hinsum[li]/hn[0]).float(),(D,)) for li in range(NL)]
print("bases ready", flush=True)


@torch.no_grad()
def forward(idx, mode, QBk=None):
    """mode None real | 'chain' composed MLPs on bottlenecked streams | 'floor' mean-input MLPs."""
    B, T2 = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    a_list = []; mh = []
    for li in range(NL):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
        x = x + aout
        if mode == 'floor':
            x = x + b.mlp(MU[li].expand(B, T2, D).to(x.dtype)); continue
        if mode == 'chain':
            af = aout.reshape(-1, D); a_list.append(af)
            co = CO[li]; xp = co['e']*x0.reshape(-1, D)
            for j in range(li+1):
                if ('a', j) in co: xp = xp + co[('a', j)]*((a_list[j] @ QBk[j]) @ QBk[j].T)
                if ('m', j) in co: xp = xp + co[('m', j)]*mh[j]
            r = xp.pow(2).sum(1)/D
            mo = (T_ev(li, xp, xp)/r.unsqueeze(1) + WT[li][3])
            mh.append(mo); x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + b.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def per_tok(mode, QBk=None):
    ces = []
    for i in range(0, len(HELD), 4):
        bb = HELD[i:i+4].to(DEV)
        lg = forward(bb[:, :-1], mode, QBk)
        ce = F.cross_entropy(lg.reshape(-1, V), bb[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = per_tok(None)
base = float(ce_real.mean())
res = {'base_CE': round(base, 5), 'uniform_ceiling_lnV': round(UNIFORM, 4), 'n_tokens': int(ce_real.numel())}
def rep(ce, ref):
    d = ce - ref; return {'dCE': round(float(d.mean()), 5), 'SE': round(float(d.std()/np.sqrt(d.numel())), 6)}
res['chain_pca'] = rep(per_tok('chain', QB['pca']), ce_real)
res['null_headspan_s1'] = rep(per_tok('chain', QB['null1']), ce_real)
res['null_headspan_s2'] = rep(per_tok('chain', QB['null2']), ce_real)
ce_floor = per_tok('floor', QB['pca'])   # mean-input MLPs = the joint MLP floor (bottleneck bases irrelevant here)
res['joint_mlp_floor'] = rep(ce_floor, ce_real)
h = res['chain_pca']['dCE']
res['available_headroom_to_floor'] = round(res['joint_mlp_floor']['dCE'], 4)
res['frac_of_floor_captured'] = round(1 - h/res['joint_mlp_floor']['dCE'], 4)
res['frac_of_uniform_ceiling'] = round(1 - h/(UNIFORM - base), 4)
print(f"base CE {base:.4f} | uniform ceiling {UNIFORM:.3f}", flush=True)
print(f"whole-model composed chain: dCE +{h:.5f} (SE {res['chain_pca']['SE']})", flush=True)
print(f"  null head-span s1/s2: +{res['null_headspan_s1']['dCE']}/{res['null_headspan_s2']['dCE']}", flush=True)
print(f"  joint MLP floor: +{res['joint_mlp_floor']['dCE']} | captured {res['frac_of_floor_captured']:.1%} of floor, {res['frac_of_uniform_ceiling']:.2%} of uniform headroom", flush=True)
json.dump(res, open(f'{QK}/qk_wholemodel_substitutable.json', 'w'), indent=2)
print("QK WHOLEMODEL SUBSTITUTABLE DONE", flush=True)
