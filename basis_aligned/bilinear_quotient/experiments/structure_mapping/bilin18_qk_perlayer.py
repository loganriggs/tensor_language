"""Closing the QK-code question: if sharing is within-layer (section 88, LOLO 0.18),
each layer's 9 heads should compress strongly on their own. REGISTERED: (a) each
of layers 2-4 has within-layer eff-rank <= 5 of 9 (strong per-layer code; random
9 gives ~8.8); (b) the per-layer codes are DISTINCT: principal angles between
layer subspaces (top-4 each) have median cos <= 0.4 (else the family was global
after all and the LOLO failure was a fitting artifact)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH, HD, D, K = 9, 128, 1152, 48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_qk_perlayer_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    acc=[]; fwd(FW[0:36,:513].to(DEV), collect=1, acc=acc)
    Y1c=acc[0]-acc[0].mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    caps={}
    def run_capture(idx):
        B_,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(5):
            blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            a=blk.attn; hcur=F.rms_norm(x,(D,))
            def qk(l):
                z=F.rms_norm(l(hcur).view(B_,T,NH,HD),(HD,))
                return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B_,T,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            if li in (2,3,4):
                for h in range(NH):
                    kk=torch.einsum('btd,bte->de',k1_[:,:,h,:],k2[:,:,h,:])
                    caps[(li,h)]=caps.get((li,h),0)+kk
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
            s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B_,T,-1))
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
    for i in range(0,24,6):
        run_capture(FW[i:i+6,:257].to(DEV))
    spans={}; ers={}
    for li in (2,3,4):
        a=m.transformer.h[li].attn
        Wq=(a.c_q.weight.detach().float()@V).view(NH,HD,K)
        Wq2=(a.c_q2.weight.detach().float()@V).view(NH,HD,K)
        rows=[]
        for h in range(NH):
            Ek=caps[(li,h)]/24.0
            M=torch.einsum('da,de,eb->ab',Wq[h],Ek,Wq2[h])
            Ms=0.5*(M+M.T)
            rows.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
        X=torch.stack(rows)
        sv=torch.linalg.svdvals(X); e=sv**2
        ers[li]=float(e.sum()**2/(e**2).sum())
        _,_,W=torch.linalg.svd(X, full_matrices=False)
        spans[li]=W[:4].T
        print(f'L{li} within-layer eff-rank: {ers[li]:.1f}/9')
    coss=[]
    for a_,b_ in ((2,3),(2,4),(3,4)):
        s=torch.linalg.svdvals(spans[a_].T@spans[b_])
        coss+=[float(x) for x in s]
    medc=sorted(coss)[len(coss)//2]
    pa=all(ers[li]<=5 for li in ers)
    pb=medc<=0.4
    out={'effranks':{str(k):v for k,v in ers.items()},
         'median_principal_cos':medc,'pred_a_perlayer':bool(pa),
         'pred_b_distinct':bool(pb)}
    print(f'median principal cos between layer codes: {medc:.2f}')
    print(f"(a) per-layer compact (<=5): {'HELD' if pa else 'FAILED'}")
    print(f"(b) codes distinct (<=0.4): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
