"""WHOLE-MODEL restricted-core sweep (fold-audit item 3): restrict EVERY MLP's map to (in-Kin x out-Kout)
simultaneously (per-layer train-gram SVD bases, per-position held means) and measure cumulative delta
cross-entropy. Does §85's ~3%-at-one-layer compound over 18 blocks? Reports the compression factor of the
implied cores. Configs: (288,144), (576,288), input-only 288. Held-back FW[448:600], paired SE."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18'); NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN=FW[0:256,:128].to(DEV); HELD=FW[448:600,:128].to(DEV); B0=6
GIN=[torch.zeros(D,D,device=DEV) for _ in range(NL)]; GOUT=[torch.zeros(D,D,device=DEV) for _ in range(NL)]
@torch.no_grad()
def fwd(idx, mode=None, PIN=None, POUT=None, MX=None, MO=None, collect=False, want_gram=False):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    xs=[] if collect else None; ms=[] if collect else None
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
        if want_gram: GIN[li].add_(torch.einsum('btd,bte->de',x,x))
        if collect: xs.append(x.clone())
        if mode is not None:
            xr=MX[li].unsqueeze(0)+((x-MX[li].unsqueeze(0))@PIN[li])@PIN[li].T
            mo=blk.mlp(F.rms_norm(xr,(D,)))
            if POUT is not None:
                mo=MO[li].unsqueeze(0)+((mo-MO[li].unsqueeze(0))@POUT[li])@POUT[li].T
        else:
            mo=blk.mlp(F.rms_norm(x,(D,)))
        if want_gram: GOUT[li].add_(torch.einsum('btd,bte->de',mo,mo))
        if collect: ms.append(mo.clone())
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return ce,xs,ms
print("gram pass ...",flush=True)
for i in range(0,TRAIN.shape[0],B0): fwd(TRAIN[i:i+B0],want_gram=True)
INb=[torch.linalg.eigh(GIN[l])[1].flip(1) for l in range(NL)]
OUTb=[torch.linalg.eigh(GOUT[l])[1].flip(1) for l in range(NL)]
print("mean pass ...",flush=True)
S,T=HELD.shape
MXs=[torch.zeros(T,D,device=DEV) for _ in range(NL)]; MOs=[torch.zeros(T,D,device=DEV) for _ in range(NL)]
for i in range(0,S,B0):
    _,xs,ms=fwd(HELD[i:i+B0],collect=True)
    for l in range(NL): MXs[l]+=xs[l].sum(0); MOs[l]+=ms[l].sum(0)
MX=[t/S for t in MXs]; MO=[t/S for t in MOs]
def run(Kin=None,Kout=None):
    PIN=[INb[l][:,:Kin].contiguous() for l in range(NL)]
    POUT=[OUTb[l][:,:Kout].contiguous() for l in range(NL)] if Kout else None
    ces=[]
    for i in range(0,S,B0):
        ce,_,_=fwd(HELD[i:i+B0],mode='r',PIN=PIN,POUT=POUT,MX=MX,MO=MO); ces.append(ce.cpu())
    return torch.cat(ces,0)
bce=[]
for i in range(0,S,B0): ce,_,_=fwd(HELD[i:i+B0]); bce.append(ce.cpu())
base=torch.cat(bce,0)
def dstat(ce):
    d=(ce-base).flatten().double(); return float(d.mean()),float(d.std()/np.sqrt(d.numel()))
res={'base_CE':float(base.mean())}
for Kin,Kout in [(288,144),(576,288),(288,None),(576,None)]:
    mn,se=dstat(run(Kin,Kout))
    tag=f'in{Kin}_out{Kout if Kout else "full"}'
    # implied core params: symmetric T restricted: Kout * Kin*(Kin+1)/2 per layer
    core=NL*( (Kout if Kout else D) * Kin*(Kin+1)//2 )
    full=NL*( D * D*(D+1)//2 )
    res[tag]={'dCE':round(mn,4),'SE':round(se,5),'core_params':core,'compression_x':round(full/core,1)}
    print(f"  ALL-18-MLPs {tag}: dCE {mn:+.4f} ± {se:.5f}  (core {core/1e6:.0f}M, {full/core:.0f}x smaller)",flush=True)
json.dump(res,open(f'{QK}/qk_allcore_restrict.json','w'),indent=1)
print("DONE",flush=True)
