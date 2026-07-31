"""COMPUTATION-level hierarchy test (Logan): restrict the MLP1 MAP itself — input side to top-K input
directions (of the pre-norm residual x it reads), output side to the §84 top-144 output subspace — and
measure faithfulness. mo' = mlp(rms(meanx + P_inK(x-meanx))), optionally out-projected. If faithful at
modest K, the COMPUTATION (not just its output trace) is hierarchically compressible. Grams from TRAIN,
eval held-back FW[448:600], paired standard errors. Random input-K control included."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18'); NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN=FW[0:256,:128].to(DEV); HELD=FW[448:600,:128].to(DEV); B0=6; LI=1
gin=torch.zeros(D,D,device=DEV); gout=torch.zeros(D,D,device=DEV)
@torch.no_grad()
def fwd(idx, mode=None, Pin=None, Pout=None, mx=None, mo_mean=None, collect=False, want_gram=False):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool)); xin=None; moc=None
    for li in range(NL):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        x=x+a.c_proj(yh.reshape(B,T,-1))
        if li==LI:
            if want_gram: gin.add_(torch.einsum('btd,bte->de',x,x))
            if collect: xin=x.clone()
            if mode is not None:
                xr=mx.unsqueeze(0)+ ( ((x-mx.unsqueeze(0))@Pin)@Pin.T if Pin is not None else (x-mx.unsqueeze(0)) )
                mo=blk.mlp(F.rms_norm(xr,(D,)))
                if Pout is not None:
                    mo=mo_mean.unsqueeze(0)+((mo-mo_mean.unsqueeze(0))@Pout)@Pout.T
            else:
                mo=blk.mlp(F.rms_norm(x,(D,)))
            if want_gram: gout.add_(torch.einsum('btd,bte->de',mo,mo))
            if collect: moc=mo.clone()
        else:
            mo=blk.mlp(F.rms_norm(x,(D,)))
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return ce,xin,moc
print("gram pass (TRAIN) ...",flush=True)
for i in range(0,TRAIN.shape[0],B0): fwd(TRAIN[i:i+B0],want_gram=True)
IN=torch.linalg.eigh(gin)[1].flip(1); OUT=torch.linalg.eigh(gout)[1].flip(1)
print("mean pass (HELD) ...",flush=True)
S,T=HELD.shape; xs=torch.zeros(T,D,device=DEV); ms=torch.zeros(T,D,device=DEV)
for i in range(0,S,B0):
    _,xin,moc=fwd(HELD[i:i+B0],collect=True); xs+=xin.sum(0); ms+=moc.sum(0)
MX=xs/S; MO=ms/S
def run(Pin=None,Pout=None,mode='r'):
    ces=[]
    for i in range(0,S,B0):
        ce,_,_=fwd(HELD[i:i+B0],mode=mode,Pin=Pin,Pout=Pout,mx=MX,mo_mean=MO); ces.append(ce.cpu())
    return torch.cat(ces,0)
base_ces=[]
for i in range(0,S,B0):
    ce,_,_=fwd(HELD[i:i+B0]); base_ces.append(ce.cpu())
base=torch.cat(base_ces,0)
def dstat(ce):
    d=(ce-base).flatten().double(); return float(d.mean()),float(d.std()/np.sqrt(d.numel()))
res={}
print("INPUT-side restriction (map input top-K, output full) ...",flush=True)
for K in [36,144,288,576]:
    mn,se=dstat(run(Pin=IN[:,:K].contiguous()))
    res[f'in_{K}']={'dCE':round(mn,4),'SE':round(se,5)}; print(f"  in-{K}: {mn:+.4f} ± {se:.5f}",flush=True)
g=torch.Generator(device=DEV).manual_seed(3); Q=torch.linalg.qr(torch.randn(D,D,generator=g,device=DEV))[0]
mn,se=dstat(run(Pin=Q[:,:288].contiguous()))
res['in_rand288']={'dCE':round(mn,4),'SE':round(se,5)}; print(f"  in-rand288: {mn:+.4f} ± {se:.5f}",flush=True)
print("JOINT map restriction (in-K x out-144) ...",flush=True)
for K in [144,288,576]:
    mn,se=dstat(run(Pin=IN[:,:K].contiguous(),Pout=OUT[:,:144].contiguous()))
    res[f'in_{K}_out_144']={'dCE':round(mn,4),'SE':round(se,5)}; print(f"  in-{K} x out-144: {mn:+.4f} ± {se:.5f}",flush=True)
json.dump(res,open(f'{QK}/qk_hub_maprestrict.json','w'),indent=1)
print("DONE",flush=True)
