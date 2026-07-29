"""CAUSAL joint chain -- the strictest test: layers 0-5's MLPs are replaced IN the running residual,
so every attention layer (1-5 and all layers above) reads the SUBSTITUTED stream; the chain consumes
causally-computed attention outputs, incrementally. No oracle anywhere. Arms:
  (a) untrained knobs, causal          (b) oracle-trained T12 knobs, causal (transfer test)
  (c) T12 knobs RETRAINED in the causal configuration (proper exposure-bias fix)
vs the six-MLP floor 7.782; oracle-joint references: untrained 72.5%, T12-trained 91.5%.
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
FLOOR6 = 7.78249
LMAX = 5; PER = 64; K6 = 576
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

# bases: same PCA-64/head recipe (real-model statistics -- basis choice unchanged)
accs = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(LMAX+1)]
@torch.no_grad()
def collect_heads(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    x = None; v1 = None; out = []
    for li in range(LMAX+1):
        blk = BLKS[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v); out.append(yh.reshape(-1, NH, HD))
        x = x + a.c_proj(yh.reshape(B, T, -1)); x = x + blk.mlp(F.rms_norm(x, (D,)))
    return out
for i in range(0, 64, 8):
    hs = collect_heads(COOC[i:i+8].to(DEV)[:, :128])
    for li in range(LMAX+1):
        accs[li] += torch.einsum('nhd,nhe->hde', hs[li].double(), hs[li].double())
QB = []
for li in range(LMAX+1):
    cw = BLKS[li].attn.c_proj.weight.detach().float(); cs = []
    for hh in range(NH):
        ev, evec = torch.linalg.eigh(accs[li][hh])
        cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:PER]].float())
    Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); QB.append(Qx)
print("bases ready", flush=True)

class Knobs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coef = torch.nn.ParameterDict()
        for l in range(LMAX+1):
            for k, v0 in CO[l].items():
                key = f"l{l}_{'e' if k=='e' else k[0]+str(k[1])}"
                self.coef[key] = torch.nn.Parameter(torch.tensor(float(v0)))
        self.g = torch.nn.Parameter(torch.ones(LMAX+1))
        self.bias = torch.nn.Parameter(torch.zeros(LMAX+1, D))
        self.diag = torch.nn.Parameter(torch.ones(LMAX+1, K6))
    def stream_a(self, l, a_flat):
        z = (a_flat @ QB[l]) * self.diag[l]
        return z @ QB[l].T
    def co(self, l, k):
        return self.coef[f"l{l}_{'e' if k=='e' else k[0]+str(k[1])}"]

def causal_forward(knobs, idx):
    """layers 0-5: attention from the SUBSTITUTED residual; MLPs replaced by chain stages."""
    B, T2 = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    a_list = []; mh = []
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
        x = x + aout
        if li <= LMAX:
            a_list.append(aout.reshape(-1, D))
            co = CO[li]
            xp = knobs.co(li, 'e')*x0.reshape(-1, D)
            for j in range(li+1):
                if ('a', j) in co: xp = xp + knobs.co(li, ('a', j))*knobs.stream_a(j, a_list[j])
                if ('m', j) in co: xp = xp + knobs.co(li, ('m', j))*mh[j]
            r = (xp.pow(2).sum(1)/D) * knobs.g[li]
            mo = T_ev(li, xp, xp)/r.unsqueeze(1).clamp_min(1e-6) + WT[li][3] + knobs.bias[li]
            mh.append(mo)
            x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

@torch.no_grad()
def audit(knobs):
    tot = 0.0; n = 0
    for i in range(0, len(FINEWEB), 4):
        b = FINEWEB[i:i+4].to(DEV)
        lg = causal_forward(knobs, b[:, :-1]).float()
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item()*b[:, 1:].numel(); n += b[:, 1:].numel()
    return tot/n

res = {'floor6': FLOOR6, 'oracle_refs': {'untrained': 0.725, 'T12_trained': 0.915}}
kn = Knobs().to(DEV)
d_a = audit(kn) - SUBBASE
res['causal_untrained'] = {'dCE': round(d_a, 5), 'frac': round(1-d_a/FLOOR6, 4)}
print(f"(a) causal, untrained knobs: +{d_a:.4f} ({1-d_a/FLOOR6:.1%})", flush=True)
sd = torch.load(f'{QK}/qk_joint_polish_T12_scalars_diag_bias.pt', map_location=DEV)
kn.load_state_dict(sd)
d_b = audit(kn) - SUBBASE
res['causal_oracle_knobs'] = {'dCE': round(d_b, 5), 'frac': round(1-d_b/FLOOR6, 4)}
print(f"(b) causal, oracle-trained knobs: +{d_b:.4f} ({1-d_b/FLOOR6:.1%})", flush=True)
opt = torch.optim.Adam(kn.parameters(), lr=2e-3)
for step in range(350):
    i = 2400 + np.random.randint(0, 2500); b = COOC[i:i+2].to(DEV)[:, :128]
    lg = causal_forward(kn, b[:, :-1]).float()
    loss = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)
d_c = audit(kn) - SUBBASE
res['causal_retrained'] = {'dCE': round(d_c, 5), 'frac': round(1-d_c/FLOOR6, 4)}
print(f"(c) causal, knobs retrained in causal config: +{d_c:.4f} ({1-d_c/FLOOR6:.1%})", flush=True)
torch.save(kn.state_dict(), f'{QK}/qk_joint_causal_knobs.pt')
json.dump(res, open(f'{QK}/qk_joint_causal.json', 'w'), indent=2)
print("QK JOINT CAUSAL DONE", flush=True)
