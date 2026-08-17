"""The functional family's spectrum: quantify 'overcomplete and diverse' vs the
rank-few envelope.

§57: the front of the model reads L1 through near-orthogonal signed quadratic
functionals sharing one magnitude envelope. Two spectra make that quantitative:
sample 240 functionals (40 forms x 6 readers, coupling matrices in L1's top-48 basis,
unit-normalised), stack as vectors, SVD. REGISTERED PREDICTIONS:
  (a) SIGNED family: effective rank >= 100 of 240 (genuinely diverse -- no small
      functional basis compresses the readers);
  (b) ENVELOPE family (|abs| of the same matrices): effective rank <= 5 (one shared
      template plus noise);
  (c) the signed family's top principal functional carries <= 10% of total mass (no
      dominant shared signed component)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_functional_spectrum_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    V=orth(Vh[:K].T)
    rows=[]
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
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            M=0.5*(M+M.T)
            rows.append((M/M.norm().clamp_min(1e-12)).flatten())
    X=torch.stack(rows)                        # (240, K*K)
    def effrank(A):
        sv=torch.linalg.svdvals(A)
        e=sv**2
        return float(e.sum()**2/(e**2).sum()), sv
    er_s,sv_s=effrank(X)
    er_a,sv_a=effrank(X.abs())
    top1=float(sv_s[0]**2/(sv_s**2).sum())
    out={'n_functionals':X.shape[0],'effrank_signed':er_s,'effrank_envelope':er_a,
         'top1_share_signed':top1}
    pa=er_s>=100; pb=er_a<=5; pc=top1<=0.10
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'signed family: eff-rank {er_s:.0f} of {X.shape[0]} | top-1 share {top1:.2f}')
    print(f'envelope family: eff-rank {er_a:.1f}')
    print(f"\n(a) signed eff-rank >= 100: {'HELD' if pa else 'FAILED'} | "
          f"(b) envelope <= 5: {'HELD' if pb else 'FAILED'} | "
          f"(c) top-1 <= 10%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
