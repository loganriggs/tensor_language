"""The 20x within-layer superadditivity: pairwise band interactions at layer 1.

§44.2: disjoint band deletions sum to 0.24 nats vs 4.90 for the full span -- the causal
mass is in cross-band interactions. Is it pairwise, or higher-order? Measure all pairs
among the five bands. REGISTERED PREDICTION: pairwise excesses sum to well under half
of the missing 4.66 nats -- i.e. the interaction is dominated by HIGHER-ORDER terms
(consistent with a quadratic-of-quadratics stack where k-band deletions interact at
all orders). Bar: sum of pairwise excesses < 2.0 nats."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_band_interactions_results.json')

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
    spans={b: orth(Vh[b[0]:b[1]].T) for b in BANDS}
    def val(bs):
        Q=torch.cat([spans[b] for b in bs],1)
        PATCH[1]=(Q,Yb@Q)
        try: return float((held()-base).mean())
        finally: PATCH.pop(1)
    solo={b: val([b]) for b in BANDS}
    out={'solo':{str(b):v for b,v in solo.items()},'pairs':{}}
    tot_pair=0.0
    print(f"  {'pair':>22} {'joint':>8} {'excess':>8}")
    for a,b in itertools.combinations(BANDS,2):
        j=val([a,b]); exc=j-solo[a]-solo[b]; tot_pair+=exc
        out['pairs'][f'{a}x{b}']={'joint':j,'excess':exc}
        print(f"  {str(a)+'x'+str(b):>22} {j:>+8.4f} {exc:>+8.4f}",flush=True)
    full=val(list(BANDS))
    ssum=sum(solo.values())
    out['full']=full; out['solo_sum']=ssum; out['pair_excess_sum']=tot_pair
    higher=full-ssum-tot_pair
    out['higher_order']=higher
    print(f'\nsolo sum {ssum:+.4f} | pairwise excess sum {tot_pair:+.4f} | '
          f'full {full:+.4f}')
    print(f'higher-order (3+) share: {higher:+.4f} '
          f'({100*higher/max(full-ssum,1e-9):.0f}% of the interaction)')
    held_=tot_pair<2.0
    out['pred_held']=bool(held_)
    print(f"registered (pairwise sum < 2.0): {'HELD' if held_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
