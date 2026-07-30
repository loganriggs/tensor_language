"""T8: does the composition arc GENERALIZE? Port the two load-bearing results to bilinsm12
(gpt2-bilinear-12l-6h-768embd; single-branch NORMALIZED squared attention, different family):
  (1) MLP gauge identity MLP(rms(x)) = D*T(x,x)/||x||^2 + bias (should hold: bilinear MLP is
      quadratic regardless of attention type).
  (2) whole-model PCA/head-bottleneck substitutability: every attention output truncated to its
      per-layer PCA-K/head basis, dCE vs base on a held-out slice, with a head-span random null.
If both port, the analytic-decomposition result is architecture-general, not a bilin18 artifact.
"""
import json, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilinsm12')
NH, HD, D = cfg['n_head'], cfg['n_embd']//cfg['n_head'], cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h); PER=HD//2  # 64 of 128
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:560][:, :129]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
b0=m.transformer.h[0]
Lw,Rw,Dw,bw = b0.mlp.Left.weight.detach().float(), b0.mlp.Right.weight.detach().float(), b0.mlp.Down.weight.detach().float(), b0.mlp.Down_bias.detach().float()

@torch.no_grad()
def attn_pat(a, hcur, B, T, cosb, sinb, mask):
    def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
    q,k=qk(a.c_q),qk(a.c_k)
    sc=(torch.einsum('bqhd,bkhd->bhqk',q,k)/(HD**0.5)).masked_fill(~mask,float('-inf'))
    return F.softmax(sc,-1)

# GATE 1: MLP gauge on block-0 input
@torch.no_grad()
def gate1():
    idx=COOC[:4].to(DEV)[:,:128]; B,T=idx.shape
    x0=F.rms_norm(m.transformer.wte(idx),(D,)); x=b0.lambdas[0]*x0+b0.lambdas[1]*x0; a=b0.attn; hcur=F.rms_norm(x,(D,))
    cos,sin=rope_tables(T,HD,DEV,x0.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    v=a.c_v(hcur).view(B,T,NH,HD); pat=attn_pat(a,hcur,B,T,cosb,sinb,mask)
    x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1)); hin=F.rms_norm(x,(D,))
    ytrue=b0.mlp(hin).reshape(-1,D); xp=x.reshape(-1,D); r2=xp.pow(2).sum(1)/D
    Tuv=0.5*(((xp@Lw.T)*(xp@Rw.T))@Dw.T + ((xp@Lw.T)*(xp@Rw.T))@Dw.T)
    ygauge=Tuv/r2.unsqueeze(1)+bw
    return float((ygauge-ytrue).norm()/ytrue.norm())
g1=gate1(); print(f"GATE1 MLP-gauge (bilinsm12): rel-err {g1:.2e}", flush=True)

# PCA bases from cooc
acc=[torch.zeros(NH,HD,HD,device=DEV,dtype=torch.float64) for _ in range(NL)]
@torch.no_grad()
def collect(idx):
    B,T=idx.shape; x0=F.rms_norm(m.transformer.wte(idx),(D,)); x=None
    cos,sin=rope_tables(T,HD,DEV,x0.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    for li in range(NL):
        b=m.transformer.h[li]; a=b.attn
        x=(b.lambdas[0]+b.lambdas[1])*x0 if li==0 else b.lambdas[0]*x+b.lambdas[1]*x0
        hcur=F.rms_norm(x,(D,)); v=a.c_v(hcur).view(B,T,NH,HD); pat=attn_pat(a,hcur,B,T,cosb,sinb,mask)
        yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        acc[li]+=torch.einsum('nhd,nhe->hde',yh.reshape(-1,NH,HD).double(),yh.reshape(-1,NH,HD).double())
        x=x+a.c_proj(yh.reshape(B,T,-1)); x=x+b.mlp(F.rms_norm(x,(D,)))
for i in range(0,64,8): collect(COOC[i:i+8].to(DEV)[:,:128])
def bases(kind,seed=0):
    Q=[]; g=torch.Generator(device=DEV).manual_seed(seed)
    for li in range(NL):
        cw=m.transformer.h[li].attn.c_proj.weight.detach().float(); cs=[]
        for hh in range(NH):
            if kind=='pca':
                ev,evec=torch.linalg.eigh(acc[li][hh]); cs.append(cw[:,hh*HD:(hh+1)*HD]@evec[:,ev.argsort(descending=True)[:PER]].float())
            else:
                Rh,_=torch.linalg.qr(torch.randn(HD,PER,generator=g,device=DEV)); cs.append(cw[:,hh*HD:(hh+1)*HD]@Rh)
        Qx,_=torch.linalg.qr(torch.cat(cs,1)); Q.append(Qx)
    return Q
QB={'pca':bases('pca'),'null':bases('null',7)}
print("bases ready", flush=True)

@torch.no_grad()
def fwd(idx,Q):
    B,T=idx.shape; x0=F.rms_norm(m.transformer.wte(idx),(D,)); x=None
    cos,sin=rope_tables(T,HD,DEV,x0.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    for li in range(NL):
        b=m.transformer.h[li]; a=b.attn
        x=(b.lambdas[0]+b.lambdas[1])*x0 if li==0 else b.lambdas[0]*x+b.lambdas[1]*x0
        hcur=F.rms_norm(x,(D,)); v=a.c_v(hcur).view(B,T,NH,HD); pat=attn_pat(a,hcur,B,T,cosb,sinb,mask)
        aout=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        if Q is not None:
            fa=aout.reshape(-1,D); aout=((fa@Q[li])@Q[li].T).view(B,T,D).to(aout.dtype)
        x=x+aout; x=x+b.mlp(F.rms_norm(x,(D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30).float()
@torch.no_grad()
def ptok(Q):
    ces=[]
    for i in range(0,len(HELD),4):
        bb=HELD[i:i+4].to(DEV); lg=fwd(bb[:,:-1],Q)
        ces.append(F.cross_entropy(lg.reshape(-1,V),bb[:,1:].reshape(-1),reduction='none').cpu())
    return torch.cat(ces)
cr=ptok(None); base=float(cr.mean())
def rep(c): d=c-cr; return {'dCE':round(float(d.mean()),5),'SE':round(float(d.std()/np.sqrt(d.numel())),6)}
res={'model':'bilinsm12','base_CE':round(base,4),'gate1_mlp_gauge_relerr':g1,
     'pca_bottleneck':rep(ptok(QB['pca'])),'headspan_null':rep(ptok(QB['null']))}
print(f"bilinsm12 PCA/head bottleneck: dCE +{res['pca_bottleneck']['dCE']} (SE {res['pca_bottleneck']['SE']}) | null +{res['headspan_null']['dCE']}", flush=True)
json.dump(res,open(f'{QK}/qk_bilinsm12_composition.json','w'),indent=2)
print("QK BILIN12 COMPOSITION DONE", flush=True)
