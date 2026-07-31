"""Logan's mean+epicycles question: is the hub SUFFICIENT under a compact HIERARCHY — keep mean + top-K SVD
directions of the deviation only (delete the complement)? Compare against §83's random-subspace-keep at the
same dims. If SVD-K is compactly sufficient (K<<576 restores function), a hierarchical view works; if SVD
barely beats random halves, the code has no privileged compact basis even for sufficiency."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18'); NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN=FW[0:256,:128].to(DEV); HELD=FW[448:600,:128].to(DEV); B0=6; LI=1
gram=torch.zeros(D,D,device=DEV)
@torch.no_grad()
def fwd(idx, keepP=None, mean=None, collect=False, want_gram=False):
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
            if want_gram: gram.add_(torch.einsum('btd,bte->de',mo,mo))
            if collect: mo1=mo.clone()
            if keepP is not None:
                dev=mo-mean.unsqueeze(0); mo=mean.unsqueeze(0)+(dev@keepP)@keepP.T   # KEEP only subspace
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return ce,mo1
print("gram pass (TRAIN) ...",flush=True)
for i in range(0,TRAIN.shape[0],B0): fwd(TRAIN[i:i+B0],want_gram=True)
_ev,_evec=torch.linalg.eigh(gram); SVD=_evec.flip(1)                       # cols ordered desc
print("mean pass (HELD) ...",flush=True)
S,T=HELD.shape; msum=torch.zeros(T,D,device=DEV)
for i in range(0,S,B0): _,mo1=fwd(HELD[i:i+B0],collect=True); msum+=mo1.sum(0)
MEAN=msum/S
def run(keepP=None):
    ces=[]
    for i in range(0,S,B0): ce,_=fwd(HELD[i:i+B0],keepP=keepP,mean=MEAN); ces.append(ce.cpu())
    return torch.cat(ces,0)
base=run()  # keepP None = full model
def dstat(ce):
    d=(ce-base).flatten().double(); return float(d.mean()),float(d.std()/np.sqrt(d.numel()))
res={}
print("KEEP mean + top-K SVD directions (sufficiency hierarchy) ...",flush=True)
for K in [4,36,144,288,576,864]:
    P=SVD[:,:K].contiguous(); mn,se=dstat(run(keepP=P))
    res[f'svd_keep_{K}']={'dCE':round(mn,4),'SE':round(se,5)}
    print(f"  keep top-{K} SVD: dCE {mn:+.4f} ± {se:.5f}",flush=True)
g=torch.Generator(device=DEV).manual_seed(2)
Q=torch.linalg.qr(torch.randn(D,D,generator=g,device=DEV))[0]
for K in [144,288,576]:
    P=Q[:,:K].contiguous(); mn,se=dstat(run(keepP=P))
    res[f'rand_keep_{K}']={'dCE':round(mn,4),'SE':round(se,5)}
    print(f"  keep random-{K}: dCE {mn:+.4f} ± {se:.5f}",flush=True)
json.dump(res,open(f'{QK}/qk_hub_hierarchy.json','w'),indent=1)
print("DONE",flush=True)
