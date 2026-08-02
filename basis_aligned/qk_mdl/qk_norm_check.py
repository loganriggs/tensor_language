"""Quick measurement for Logan: norms of the token embedding vs block 0's feed-forward output,
raw and as-consumed at block 1's input; plus the stream lambdas that set the leak.
Forward verbatim from qk_hub_threshold.py. Read-only."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
DEV='cuda'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd']//cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV)
wte = m.transformer.wte.weight.detach().float()
toks = HELD.reshape(-1)
print(f"raw wte row norm (held tokens): mean {wte[toks].norm(dim=-1).mean():.1f}")
print(f"rms-normed embedding (as injected): norm sqrt(D) = {D**0.5:.2f} exactly, every token")
en, mn, an = [], [], []
with torch.no_grad():
    for i in range(0, 48, 6):
        idx = HELD[i:i+6]; B,T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin = rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb = cos[None,:,None,:],sin[None,:,None,:]
        mask = torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(2):
            blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
            def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
            ao=a.c_proj(yh.reshape(B,T,-1)); x=x+ao; mo=blk.mlp(F.rms_norm(x,(D,)))
            if li==0:
                mn.append(mo.norm(dim=-1).reshape(-1).float()); an.append(ao.norm(dim=-1).reshape(-1).float())
                # what block 1 receives from each, after the lambda mix at block 1's entry:
                b1=m.transformer.h[1]
                # embedding piece at block1 input: lam0*lam-mixed... measure directly next iter
            x=x+mo
b1 = m.transformer.h[1]; b0 = m.transformer.h[0]
mo_n = torch.cat(mn); ao_n = torch.cat(an)
print(f"block-0 feed-forward output norm (raw, at its own layer): mean {mo_n.mean():.1f}  median {mo_n.median():.1f}")
print(f"block-0 attention output norm (raw): mean {ao_n.mean():.1f}")
l0b1, l1b1 = b1.lambdas[0].item(), b1.lambdas[1].item()
lamE = (b1.lambdas[0]*(b0.lambdas[0]+b0.lambdas[1])+b1.lambdas[1]).item()
print(f"block-1 entry mix: lambda0 (carry stream, incl. block-0 write) = {l0b1:.4f}; lambda1 (fresh embedding re-inject) = {l1b1:.4f}")
print(f"  -> embedding effective coefficient at block-1 input (lamE) = {lamE:.3f}; block-0-write coefficient = {l0b1:.4f}")
print(f"  -> as-consumed norms at block 1: embedding {lamE*D**0.5:.1f} vs block-0 write {l0b1*mo_n.mean():.1f} vs attention-0 {l0b1*ao_n.mean():.1f}")
lams = [(h.lambdas[0].item(), h.lambdas[1].item()) for h in m.transformer.h]
p=1.0
for a_,_ in lams: p*=a_
print("per-block (lambda0 carry, lambda1 emb-reinject):", [(round(a_,3),round(b_,3)) for a_,b_ in lams])
print(f"product of lambda0 over {NL} blocks (block-0 write's direct-path survival to readout): {p:.1e}")
