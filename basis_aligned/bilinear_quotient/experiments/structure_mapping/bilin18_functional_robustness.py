"""The user's robustness hypothesis: is dense support noise-protective?

Hypothesis (2026-08-17): spreading each functional over the full support makes reader
coefficients robust -- deleting or noising any single L1 direction barely moves any
coefficient, whereas sparse-support functionals would be fragile. Test on the trained
model: for 12 reader coefficients, (i) worst-case single-direction deletion damage
(max over the 48 basis dirs of |Delta c| / sigma_c), (ii) coefficient SNR under
isotropic noise on L1's output; both compared against matched SPARSE controls
(synthetic coefficients built from rank-2, 4-support-direction forms with matched
natural variance, read through the same downstream measurement). REGISTERED
PREDICTIONS: (a) the real coefficients' worst single-direction damage is <= 1/2 the
sparse controls'; (b) under isotropic noise the real/sparse robustness ratio is ~1
(dense support does NOT help against isotropic noise -- it helps against targeted/
sparse corruption only; unit-norm functionals pass isotropic noise equally). Bar (b)
is a null prediction that disciplines the hypothesis: if dense support helped
isotropic noise too, the robustness story would be suspiciously unfalsifiable."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
READERS=(2,5,13)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_functional_robustness_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    y=(Y1c@V)[:20000]                        # (n, K)
    mats=[]
    for j in READERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in (0,2,5,8):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            mats.append(0.5*(M+M.T))
    g=torch.Generator(device=DEV).manual_seed(0)
    # v2 control: sparse in the WHITENED basis (covariance-matched), isolating
    # support density from data-covariance alignment (v1's confound, §74)
    Cy=(y.T@y/y.shape[0]).double()
    evy,Uy=torch.linalg.eigh(Cy)
    Wh=(Uy*evy.clamp_min(1e-9).sqrt())@Uy.T
    def sparse_control(M):
        # rank-2 form supported on 4 random directions, scaled to match c-variance
        idx=torch.randperm(K,generator=g,device=DEV)[:4]
        A=torch.zeros(K,K,device=DEV)
        v1=torch.zeros(K,device=DEV); v1[idx[:2]]=torch.randn(2,device=DEV,generator=g)
        v2=torch.zeros(K,device=DEV); v2[idx[2:]]=torch.randn(2,device=DEV,generator=g)
        A=torch.outer(v1,v1)-torch.outer(v2,v2)
        A=(Wh.float()@A@Wh.float())      # place the sparse support in whitened coords
        c=torch.einsum('na,ab,nb->n',y,A,y)
        cm=torch.einsum('na,ab,nb->n',y,M,y)
        A=A*(cm.std()/c.std().clamp_min(1e-9))
        return A
    def worst_del(M):
        c0=torch.einsum('na,ab,nb->n',y,M,y); s=c0.std().clamp_min(1e-9)
        worst=0.0
        for k in range(K):
            yk=y.clone(); yk[:,k]=y[:,k].mean()
            ck=torch.einsum('na,ab,nb->n',yk,M,yk)
            worst=max(worst,float((ck-c0).abs().mean()/s))
        return worst
    def noise_snr(M):
        c0=torch.einsum('na,ab,nb->n',y,M,y)
        eps=torch.randn_like(y)*y.std()*0.3
        c1=torch.einsum('na,ab,nb->n',y+eps,M,y+eps)
        return float(c0.std()/ (c1-c0).std().clamp_min(1e-9))
    real_w=[]; sp_w=[]; real_s=[]; sp_s=[]
    for M in mats:
        A=sparse_control(M)
        real_w.append(worst_del(M)); sp_w.append(worst_del(A))
        real_s.append(noise_snr(M)); sp_s.append(noise_snr(A))
    rw=sum(real_w)/len(real_w); sw=sum(sp_w)/len(sp_w)
    rs=sum(real_s)/len(real_s); ss=sum(sp_s)/len(sp_s)
    out={'worst_del_real':rw,'worst_del_sparse':sw,
         'noise_snr_real':rs,'noise_snr_sparse':ss}
    pa=rw<=0.5*sw
    ratio=rs/max(ss,1e-9)
    pb=0.7<=ratio<=1.4
    out['pred_a_targeted_robust']=bool(pa)
    out['pred_b_isotropic_null']=bool(pb)
    print(f'worst single-direction damage: real {rw:.2f}s vs sparse {sw:.2f}s')
    print(f'isotropic-noise SNR: real {rs:.2f} vs sparse {ss:.2f} (ratio {ratio:.2f})')
    print(f"\n(a) dense support protects against targeted damage (<=1/2): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) no isotropic advantage (null discipline): "
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
