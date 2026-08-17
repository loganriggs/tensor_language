"""Empirical structure of C_edge(3->4): which L3 bands couple hardest?

§42 found coupling per unit damage varies ~5x with which L3 directions are damaged;
§43 found the weight-side operator K does not predict it. So measure it directly:
damage L3 along disjoint PCA bands (equal counts where possible), L4 damage fixed
(PCA-32), compute c = excess/(d3*d4) per band. REGISTERED PREDICTIONS:
  (a) from §42's grid (ka=32 rows c~25-30 vs ka=8 c~13): the band [8,32) has
      c > 2x the band [0,8).
  (b) c declines beyond rank 64: band [64,128) below band [8,32) by > 2x.
Control: a random 24-dim span (c expected between the extremes, since random mixes
bands)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,8),(8,32),(32,64),(64,128),(128,256))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_coupling_bands_results.json')

@torch.no_grad()
def collect(li):
    outs=[]
    def hook(mod,inp,o): outs.append(o.detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    Y=torch.cat(outs); return Y.mean(0), Y

def main():
    t0=time.time()
    base=held()
    Yb3,Y3=collect(3); _,_,Vh3=torch.linalg.svd((Y3-Yb3).float(), full_matrices=False)
    Yb4,Y4=collect(4); _,_,Vh4=torch.linalg.svd((Y4-Yb4).float(), full_matrices=False)
    span4=orth(Vh4[:32].T)
    def val(patches):
        for li,Q,Yb in patches: PATCH[li]=(Q,Yb@Q)
        try: return float((held()-base).mean())
        finally:
            for li,_,_ in patches: PATCH.pop(li)
    d4=val([(4,span4,Yb4)])
    out={'d4':d4,'bands':{}}
    print(f'd4 = {d4:+.4f}')
    print(f"  {'band':>10} {'d3':>8} {'excess':>8} {'c':>7}")
    for lo,hi in BANDS:
        Q=orth(Vh3[lo:hi].T)
        d3=val([(3,Q,Yb3)])
        j=val([(3,Q,Yb3),(4,span4,Yb4)])
        exc=j-d3-d4; c=exc/max(d3*d4,1e-9)
        out['bands'][f'{lo}-{hi}']={'d3':d3,'excess':exc,'c':c}
        print(f"  {f'[{lo},{hi})':>10} {d3:>+8.4f} {exc:>+8.4f} {c:>+7.1f}",flush=True)
    g=torch.Generator(device=DEV).manual_seed(0)
    Qr=orth(torch.randn(D,24,device=DEV,generator=g))
    d3r=val([(3,Qr,Yb3)]); jr=val([(3,Qr,Yb3),(4,span4,Yb4)])
    cr=(jr-d3r-d4)/max(d3r*d4,1e-9)
    out['random24']={'d3':d3r,'c':cr}
    print(f"  {'random-24':>10} {d3r:>+8.4f} {jr-d3r-d4:>+8.4f} {cr:>+7.1f}")
    ca=out['bands']['8-32']['c']; c0=out['bands']['0-8']['c']; c64=out['bands']['64-128']['c']
    out['pred_a']=bool(ca>2*c0); out['pred_b']=bool(c64<ca/2)
    print(f"\n(a) c[8,32) > 2x c[0,8): {'HELD' if out['pred_a'] else 'FAILED'}")
    print(f"(b) c[64,128) < c[8,32)/2: {'HELD' if out['pred_b'] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
