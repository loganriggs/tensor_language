"""Are the negative-deletion spans real? Section 96 found five tail layers where
deleting an 8-dim span IMPROVES held-out CE (rows 300-380). Before building on
them: replicate on DISJOINT rows with fresh random-control seeds.

REGISTERED PREDICTIONS: (a) the two strongest negatives (L9 PCA-span -0.0156,
L15 PCA-span -0.0107) keep their sign on disjoint rows 384-452; (b) magnitude
survives at >= 40% of the original; (c) fresh random-8 spans at the same layers
stay within +-0.002 (the improvement is span-specific, not site-generic)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_negative_stability_results.json')

@torch.no_grad()
def ce_eval(patches, lo=384, hi=452):
    hs=[]
    for li,(Q,cbar) in patches.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    tot,n=0.0,0
    for i in range(lo,hi,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    ORIG={9:-0.0156, 15:-0.0107}
    base=ce_eval({})
    print(f'baseline CE (disjoint rows) {base:.4f}\n',flush=True)
    out={'base':base,'layers':{}}
    g=torch.Generator(device=DEV).manual_seed(7)
    okA=okB=okC=0
    for li in (9,15):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Qp=orth(Vh[:8].T)
        d_pca=ce_eval({li:(Qp,Ybar@Qp)})-base
        Qr=orth(torch.randn(D,8,device=DEV,generator=g))
        d_rnd=ce_eval({li:(Qr,Ybar@Qr)})-base
        out['layers'][str(li)]={'pca':d_pca,'random':d_rnd,'orig':ORIG[li]}
        print(f'L{li}: pca-span {d_pca:+.4f} (orig {ORIG[li]:+.4f}) | '
              f'fresh random {d_rnd:+.4f}',flush=True)
        if d_pca<0: okA+=1
        if d_pca<=0.4*ORIG[li]: okB+=1
        if abs(d_rnd)<=0.002: okC+=1
    pa=okA==2; pb=okB==2; pc=okC==2
    out['pred_a_sign']=bool(pa); out['pred_b_magnitude']=bool(pb)
    out['ctrl_c']=bool(pc)
    print(f"\n(a) both negatives keep sign: {'HELD' if pa else 'FAILED'} ({okA}/2)")
    print(f"(b) magnitude >=40%: {'HELD' if pb else 'FAILED'} ({okB}/2)")
    print(f"(c) fresh randoms inert: {'HELD' if pc else 'VIOLATED'} ({okC}/2)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
