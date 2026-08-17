"""The coverage gap: where does the causal mass outside the top-32 spans live?

Layer 1's full quadratic deletion costs ~5.65 nats; its top-32 output span costs 0.43.
Layer 0: 1.80 vs 0.13. The program's direction-level machinery has therefore been
working inside <10% of these layers' causal effect. This measures the coverage curve --
deletion cost of the top-k span as k grows to full rank -- in two bases:

    output-PCA   (variance ordering, what the batteries used)
    G_lam        (weights + input second moment; the causal-leader predictor)

REGISTERED PREDICTIONS:
  (a) coverage is heavy-tailed at layer 1: even 128 directions carry under half the
      full deletion cost -- cost(128)/cost(full) < 0.5.
  (b) the G_lam ordering dominates output-PCA at every k on both layers (it predicted
      causal leaders better than variance did; the same should hold for cumulative
      causal coverage).
Gate: k = 1152 in either basis must equal the full deletion cost (same operator).
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
KS=(32,64,128,256,512,1152)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_coverage_curve_results.json')

@torch.no_grad()
def collect(li, what):
    ins,outs=[],[]
    def hook(mod,inp,o):
        if what!='out': ins.append(inp[0].detach().reshape(-1,D).float())
        if what!='in': outs.append(o.detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    return (torch.cat(ins) if what=='in' else torch.cat(outs))

def main():
    t0=time.time()
    base=held(); out={'layers':{}}
    for li in (1,0):
        Y=collect(li,'out'); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        X=collect(li,'in'); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        G=Dw@((L@S@L.T)*(R@S@R.T))@Dw.T
        ev,U=torch.linalg.eigh(G)
        Ulam=U[:,ev.argsort(descending=True)]
        def val(Q):
            PATCH[li]=(Q,Ybar@Q)
            try: return float((held()-base).mean())
            finally: PATCH.pop(li)
        rows={}
        print(f'layer {li}:')
        print(f"  {'k':>5} {'PCA cost':>9} {'G_lam cost':>11}")
        for k in KS:
            cp=val(orth(Vh[:k].T))
            cl=val(orth(Ulam[:,:k]))
            rows[k]={'pca':cp,'lam':cl}
            print(f"  {k:>5} {cp:>+9.4f} {cl:>+11.4f}",flush=True)
        out['layers'][li]=rows
    r1=out['layers'][1]
    pa = r1[128]['pca']/max(r1[1152]['pca'],1e-9) < 0.5
    pb = all(out['layers'][li][k]['lam'] >= out['layers'][li][k]['pca']
             for li in (0,1) for k in KS[:-1])
    out['pred_a_heavy_tail']=bool(pa); out['pred_b_lam_dominates']=bool(pb)
    print(f"\n(a) layer-1 cost(128)/cost(full) = "
          f"{r1[128]['pca']/r1[1152]['pca']:.2f} -> "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) G_lam >= PCA at every k: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
