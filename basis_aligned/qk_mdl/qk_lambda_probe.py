"""What do the 36 stream-mixing scalars buy for CE? Direct weight intervention, no mean-ablation.
Forward verbatim from qk_hub_threshold.py. Held FW[448:600,:128], paired per-token standard errors."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
DEV='cuda'; m,cfg=load_elriggs('bilin18')
NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']; V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD=FW[448:600,:128].to(DEV); B0=6

L0=np.array([m.transformer.h[i].lambdas[0].item() for i in range(NL)])
L1=np.array([m.transformer.h[i].lambdas[1].item() for i in range(NL)])
print("exact lambda0:", np.round(L0,4).tolist())
print("exact lambda1:", np.round(L1,4).tolist())
print("lambda1 unique values:", sorted(set(np.round(L1,4).tolist())))

@torch.no_grad()
def ce(l0=None, l1=None):
    """l0,l1: optional length-NL overrides for the carry/reinject scalars."""
    o0=[m.transformer.h[i].lambdas[0].item() for i in range(NL)]
    o1=[m.transformer.h[i].lambdas[1].item() for i in range(NL)]
    per=[]
    for s in range(0,HELD.shape[0],B0):
        idx=HELD[s:s+B0]; B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(NL):
            blk=m.transformer.h[li]
            a0=o0[li] if l0 is None else l0[li]; a1=o1[li] if l1 is None else l1[li]
            x=a0*x+a1*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
            def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
            x=x+a.c_proj(yh.reshape(B,T,-1)); x=x+blk.mlp(F.rms_norm(x,(D,)))
        lg=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
        per.append(F.cross_entropy(lg[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1).mean(0).cpu())
    return torch.cat([p for p in per])  # not aligned per-seq; use pooled below

@torch.no_grad()
def ce_flat(l0=None,l1=None):
    o0=[m.transformer.h[i].lambdas[0].item() for i in range(NL)]
    o1=[m.transformer.h[i].lambdas[1].item() for i in range(NL)]
    outs=[]
    for s in range(0,HELD.shape[0],B0):
        idx=HELD[s:s+B0]; B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(NL):
            blk=m.transformer.h[li]
            a0=o0[li] if l0 is None else l0[li]; a1=o1[li] if l1 is None else l1[li]
            x=a0*x+a1*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
            def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
            x=x+a.c_proj(yh.reshape(B,T,-1)); x=x+blk.mlp(F.rms_norm(x,(D,)))
        lg=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
        outs.append(F.cross_entropy(lg[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1).cpu())
    return torch.cat(outs,0).reshape(-1)  # per-token

base=ce_flat()
bm=base.mean().item()
print(f"\nBASELINE CE = {bm:.4f}  (n={base.numel()} tokens)")
def dCE(v):
    d=v-base; return d.mean().item(), d.std().item()/d.numel()**0.5

# GLOBAL scenarios
scen={
 'all lambda0 -> 1 (kill the leak/reset schedule, vanilla carry)': (np.ones(NL), None),
 'all lambda1 -> 0 (no embedding re-injection anywhere)': (None, np.zeros(NL)),
 'VANILLA residual (lambda0=1, lambda1=0)': (np.ones(NL), np.zeros(NL)),
 'freeze lambda0 at its geo-mean 0.63': (np.full(NL,0.63), None),
 'all lambda1 -> 8 (uniform max re-injection)': (None, np.full(NL,8.0)),
}
print("\n=== GLOBAL interventions (dCE +- standard error) ===")
for name,(a,b) in scen.items():
    a2=a.tolist() if a is not None else None; b2=b.tolist() if b is not None else None
    d,se=dCE(ce_flat(a2,b2)); print(f"  {name:60s} +{d:.4f} +- {se:.4f}")

# PER-BLOCK: set lambda0[i] -> 1  and  lambda1[i] -> 0
print("\n=== per-block lambda0 -> 1 (remove that block's carry-reset) ===")
r0=[]
for i in range(NL):
    a=L0.copy(); a[i]=1.0; d,se=dCE(ce_flat(a.tolist(),None)); r0.append((i,d,se))
for i,d,se in sorted(r0,key=lambda t:-abs(t[1]))[:8]:
    print(f"  block {i:2d}: true lambda0={L0[i]:.3f} -> 1 : dCE {d:+.4f} +- {se:.4f}")
print("=== per-block lambda1 -> 0 (remove that block's embedding re-injection) ===")
r1=[]
for i in range(NL):
    b=L1.copy(); b[i]=0.0; d,se=dCE(ce_flat(None,b.tolist())); r1.append((i,d,se))
for i,d,se in sorted(r1,key=lambda t:-abs(t[1]))[:8]:
    print(f"  block {i:2d}: true lambda1={L1[i]:.3f} -> 0 : dCE {d:+.4f} +- {se:.4f}")
np.savez('qk_lambda_probe.npz', L0=L0,L1=L1, base=bm,
         perblock_l0=np.array([(d,se) for _,d,se in r0]),
         perblock_l1=np.array([(d,se) for _,d,se in r1]))
print("\nsaved qk_lambda_probe.npz")
