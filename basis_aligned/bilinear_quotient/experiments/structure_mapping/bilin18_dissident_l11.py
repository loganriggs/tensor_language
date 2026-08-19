"""Why is L11 the dissident? Section 84: L11's functionals reconstruct from the
six-reader basis at R^2 -0.10 -- the only reader of 16 speaking none of the shared
code. Two hypotheses: (D1) disengagement -- L11 barely reads L1 at all, so its
coupling matrices are near-zero noise and R^2 on unit-normalized noise is
meaningless; (D2) foreign language -- L11 reads L1 with normal strength but with
functionals outside the shared span.

REGISTERED PREDICTIONS: (a) D1: L11's raw (un-normalized) coupling-matrix norms
are bottom-2 of the 16 readers (median over its 40 forms); (b) if instead norms
are middling (rank 5-12), D2 stands and L11's own family should still be
internally coherent: eff-rank of its 40 functionals <= 25 (a real code, just a
foreign one). Controls: norms for all 16 readers reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH, HD, D, K, NF = 9, 128, 1152, 48, 40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_dissident_l11_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    def collect(li):
        outs=[]
        h=m.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    norms={}; fam11=None
    for j in range(2,18):
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        ns=[]; rows=[]
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T)
            ns.append(float(Ms.norm()))
            if j==11: rows.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
        norms[j]=sorted(ns)[NF//2]
        if j==11: fam11=torch.stack(rows)
    rank11=sorted(norms, key=norms.get).index(11)+1  # 1 = smallest
    sv=torch.linalg.svdvals(fam11); e=sv**2
    er11=float(e.sum()**2/(e**2).sum())
    out={'median_norms':{str(k):v for k,v in norms.items()},
         'l11_norm_rank_ascending':rank11,'l11_family_effrank':er11}
    pa=rank11<=2
    pb=(3<=rank11<=13) and er11<=25
    out['pred_a_disengaged']=bool(pa); out['pred_b_foreign_code']=bool(pb)
    print('median coupling norm by reader:')
    for k in sorted(norms): print(f'  L{k:2d}: {norms[k]:.4f}'
                                  + ('   <-- dissident' if k==11 else ''))
    print(f'\nL11 norm rank (ascending): {rank11}/16 | own-family eff-rank {er11:.1f}')
    print(f"(a) disengaged (bottom-2 norm): {'HELD' if pa else 'FAILED'}")
    print(f"(b) foreign but coherent code: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
