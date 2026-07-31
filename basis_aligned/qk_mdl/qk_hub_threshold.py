"""Characterize the SHAPE of the MLP1 hub's holistic causal effect (§82 follow-up): is it a THRESHOLD
phenomenon? Two sweeps on held-back FW[448:600], global delta cross-entropy vs the full model, paired SE:
(A) ALPHA sweep: mo = per-position-mean + alpha*(mo - mean), alpha in {0,.25,.5,.75,.9,1} — graded vs cliff.
(B) SUBSPACE-FRACTION sweep: remove the projection of the deviation onto a RANDOM d-dim subspace,
    d/1152 in {25%,50%,75%,90%,100%} — how does cost grow with the FRACTION of deviation removed?
Forward conventions verbatim from qk_prior_examples.py / qk_extend_coverage.py lineage. Batch 6, <4GB."""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18')
NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']; V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD=FW[448:600,:128].to(DEV); B0=6; LI=1

@torch.no_grad()
def fwd(idx, mode=None, alpha=1.0, P=None, mean=None, collect_mean=False):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool)); mo1=None
    for li in range(NL):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        x=x+a.c_proj(yh.reshape(B,T,-1)); mo=blk.mlp(F.rms_norm(x,(D,)))
        if li==LI:
            if collect_mean: mo1=mo.clone()
            if mode=='alpha': mo=mean.unsqueeze(0)+alpha*(mo-mean.unsqueeze(0))
            elif mode=='proj':
                dev=mo-mean.unsqueeze(0); mo=mean.unsqueeze(0)+dev-(dev@P)@P.T
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return ce, mo1

# per-position mean of mo over held set
print("mean pass ...",flush=True)
S,T=HELD.shape; msum=torch.zeros(T,D,device=DEV)
for i in range(0,S,B0):
    _,mo1=fwd(HELD[i:i+B0],collect_mean=True); msum[:mo1.shape[1]]+=mo1.sum(0)
MEAN=msum/S
def run(mode=None,alpha=1.0,P=None):
    ces=[]
    for i in range(0,S,B0):
        ce,_=fwd(HELD[i:i+B0],mode=mode,alpha=alpha,P=P,mean=MEAN); ces.append(ce.cpu())
    return torch.cat(ces,0)
base=run()
def dstat(ce):
    d=(ce-base).flatten().double(); return float(d.mean()),float(d.std()/np.sqrt(d.numel()))
res={'alpha_sweep':{},'subspace_sweep':{}}
print("ALPHA sweep ...",flush=True)
for al in [0.0,0.25,0.5,0.75,0.9,1.0]:
    mn,se=dstat(run(mode='alpha',alpha=al)) if al<1.0 else (0.0,0.0)
    res['alpha_sweep'][str(al)]={'dCE':round(mn,4),'SE':round(se,5)}
    print(f"  alpha={al}: dCE {mn:+.4f} ± {se:.5f}",flush=True)
print("SUBSPACE-fraction sweep (random orthonormal) ...",flush=True)
g=torch.Generator(device=DEV).manual_seed(1)
Q=torch.linalg.qr(torch.randn(D,D,generator=g,device=DEV))[0]
for frac in [0.25,0.5,0.75,0.9,1.0]:
    d=int(D*frac); P=Q[:,:d].contiguous()
    mn,se=dstat(run(mode='proj',P=P))
    res['subspace_sweep'][str(frac)]={'dim':d,'dCE':round(mn,4),'SE':round(se,5)}
    print(f"  frac={frac} (d={d}): dCE {mn:+.4f} ± {se:.5f}",flush=True)
json.dump(res,open(f'{QK}/qk_hub_threshold.json','w'),indent=1)
print("DONE",flush=True)
