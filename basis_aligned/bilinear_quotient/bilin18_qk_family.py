"""Does the QK code have its own vocabulary? Section 84: QK query-side couplings
are disjoint from the MLP code. Question: do the 27 heads (layers 2-4) at least
share among THEMSELVES, the way MLP readers do (80 dims for 240 functionals)?

REGISTERED PREDICTIONS: (a) sharing -- the 27 unit-normalized QK couplings have
eff-rank <= 15 (a compact QK vocabulary); (b) leave-one-layer-out: heads of a
held-out layer reconstruct from the other two layers' span at median R^2 >= 0.3
(the code is cross-layer, not per-layer). Null: 27 random symmetric matrices
(eff-rank should be ~26, LOLO R^2 ~<= 0)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH, HD, D, K = 9, 128, 1152, 48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_qk_family_results.json')

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
    mats={}
    for li in (2,3,4):
        a=m.transformer.h[li].attn
        Wq=(a.c_q.weight.detach().float()@V).view(NH,HD,K)
        Wq2=(a.c_q2.weight.detach().float()@V).view(NH,HD,K)
        for h in range(NH):
            Ek=caps[(li,h)]/24.0
            M=torch.einsum('da,de,eb->ab',Wq[h],Ek,Wq2[h])
            Ms=0.5*(M+M.T)
            mats[(li,h)]=(Ms/Ms.norm().clamp_min(1e-12)).flatten()
    X=torch.stack(list(mats.values()))
    sv=torch.linalg.svdvals(X); e=sv**2
    er=float(e.sum()**2/(e**2).sum())
    g=torch.Generator(device=DEV).manual_seed(0)
    Xr=[]
    for _ in range(27):
        A=torch.randn(K,K,device=DEV,generator=g); A=0.5*(A+A.T)
        Xr.append((A/A.norm()).flatten())
    Xr=torch.stack(Xr)
    svr=torch.linalg.svdvals(Xr); er_r=float((svr**2).sum()**2/((svr**2).sum(0)**0+((svr**2)**2).sum()))
    er_r=float((svr**2).sum()**2/((svr**2)**2).sum())
    lolo=[]
    for hold in (2,3,4):
        tr=torch.stack([v for (li,h),v in mats.items() if li!=hold])
        te=torch.stack([v for (li,h),v in mats.items() if li==hold])
        _,_,W=torch.linalg.svd(tr, full_matrices=False)
        Bb=W[:12]
        rec=(te@Bb.T)@Bb
        r2=[1-float(((te[i]-rec[i])**2).sum()) for i in range(te.shape[0])]
        lolo+= r2
    med=sorted(lolo)[len(lolo)//2]
    pa=er<=15; pb=med>=0.3
    out={'qk_effrank':er,'random_effrank':er_r,'lolo_median_r2':med,
         'pred_a_compact':bool(pa),'pred_b_crosslayer':bool(pb)}
    print(f'QK family eff-rank: {er:.1f} (random 27: {er_r:.1f})')
    print(f'leave-one-layer-out median R^2: {med:.2f}')
    print(f"(a) compact (<=15): {'HELD' if pa else 'FAILED'}")
    print(f"(b) cross-layer (>=0.3): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
