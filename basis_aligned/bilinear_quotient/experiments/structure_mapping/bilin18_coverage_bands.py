"""The mid-spectrum causal peak at layer 1, measured band-wise (not cumulatively).

§39's cumulative curve implied per-direction deletion cost peaks in ranks 256-512.
Cumulative differences conflate interference between bands; disjoint band deletions
measure it cleanly. REGISTERED PREDICTIONS at layer 1:
  (a) per-dimension cost of band [256,512) exceeds band [0,32) -- i.e. a mid-spectrum
      direction carries more causal weight than a top-variance direction on average;
  (b) the same holds for band [128,256) vs [0,32).
Control: a random 128-dim span drawn from the tail (ranks 512+), predicted lowest
per-dim cost."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_coverage_bands_results.json')

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
    Yb,Y=collect(1); _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
    def val(Q):
        PATCH[1]=(Q,Yb@Q)
        try: return float((held()-base).mean())
        finally: PATCH.pop(1)
    out={'bands':{}}
    print(f"  {'band':>12} {'cost':>8} {'cost/dim':>10}")
    for lo,hi in BANDS:
        Q=orth(Vh[lo:hi].T)
        c=val(Q); per=c/(hi-lo)
        out['bands'][f'{lo}-{hi}']={'cost':c,'per_dim':per}
        print(f"  {f'[{lo},{hi})':>12} {c:>+8.4f} {per:>10.5f}",flush=True)
    pa=out['bands']['256-512']['per_dim']>out['bands']['0-32']['per_dim']
    pb=out['bands']['128-256']['per_dim']>out['bands']['0-32']['per_dim']
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) per-dim [256,512) > [0,32): {'HELD' if pa else 'FAILED'}")
    print(f"(b) per-dim [128,256) > [0,32): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
