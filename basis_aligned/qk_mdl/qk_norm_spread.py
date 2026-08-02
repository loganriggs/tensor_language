"""Per-token spread of the magnitudes RMSNorm erases vs what block 0 restores.
Read-only. Forward verbatim from qk_hub_threshold.py."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
DEV='cuda'; m,cfg=load_elriggs('bilin18')
NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD=FW[448:600,:128].to(DEV)
wte=m.transformer.wte.weight.detach().float()
# raw embedding norm spread across the vocabulary and across held tokens
vn=wte.norm(dim=-1)
print(f"raw wte norm over FULL vocab: mean {vn.mean():.0f} std {vn.std():.0f} min {vn.min():.0f} max {vn.max():.0f}  (coeff of variation {vn.std()/vn.mean():.2f})")
# block-0 write norm per (token) on held data, and whether it's ~constant per token id
mns=[]; tks=[]
with torch.no_grad():
    for i in range(0,60,6):
        idx=HELD[i:i+6]; B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        blk=m.transformer.h[0]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD); v1=v; v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        x=x+a.c_proj(yh.reshape(B,T,-1)); mo=blk.mlp(F.rms_norm(x,(D,)))
        mns.append(mo.norm(dim=-1).reshape(-1).float().cpu()); tks.append(idx.reshape(-1).cpu())
mns=torch.cat(mns); tks=torch.cat(tks)
print(f"block-0 write norm across held positions: mean {mns.mean():.0f} std {mns.std():.0f} (coeff of variation {mns.std()/mns.mean():.2f})")
# is the write norm token-determined? compare within-token std to overall std
import collections
byt=collections.defaultdict(list)
for n,t in zip(mns.tolist(),tks.tolist()): byt[t].append(n)
within=[np.std(v) for v in byt.values() if len(v)>=5]
means=[np.mean(v) for v in byt.values() if len(v)>=5]
print(f"tokens with >=5 occ: {len(within)}; mean WITHIN-token std {np.mean(within):.0f} vs BETWEEN-token std {np.std(means):.0f}")
print(f"  -> fraction of write-norm variance that is token-determined ~ {np.var(means)/(np.var(means)+np.mean([np.var(v) for v in byt.values() if len(v)>=5])):.2f}")
# a couple concrete tokens
dec=lambda t: repr(m.enc.decode([t])) if hasattr(m,'enc') else str(t)
srt=sorted([(np.mean(v),t) for t,v in byt.items() if len(v)>=5])
print("lowest write-norm tokens:", [(round(a),t) for a,t in srt[:3]])
print("highest write-norm tokens:", [(round(a),t) for a,t in srt[-3:]])
