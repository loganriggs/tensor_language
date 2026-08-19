"""Contrast case: is the compressible layer's interaction hierarchy shallow?

Layer 1 (uncompressible) is graded to all orders: solo 5% / pairs 19% / order-3 35% /
order-4+ 41% of its full-span deletion cost. Layer 16 is the two-direction layer.
REGISTERED PREDICTION: layer 16's interaction is shallow -- solo + pairwise capture
>= 70% of its full-span cost (compressibility and interaction depth are the same
property seen from two sides). Same five-band Mobius machinery."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
LI=16
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_band_orders_l16_results.json')

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
    Yb,Y=collect(LI); _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
    spans={b: orth(Vh[b[0]:b[1]].T) for b in BANDS}
    def val(bs):
        Q=torch.cat([spans[b] for b in bs],1)
        PATCH[LI]=(Q,Yb@Q)
        try: return float((held()-base).mean())
        finally: PATCH.pop(LI)
    solo={b: val([b]) for b in BANDS}
    psum=0.0
    for a,b in itertools.combinations(BANDS,2):
        psum+=val([a,b])-solo[a]-solo[b]
    full=val(list(BANDS))
    ssum=sum(solo.values())
    out={'solo_sum':ssum,'pair_excess_sum':psum,'full':full,
         'solo':{str(b):v for b,v in solo.items()}}
    share=(ssum+psum)/max(full,1e-9)
    out['solo_plus_pair_share']=share
    print(f'layer {LI}: solo sum {ssum:+.4f} | pairwise excess {psum:+.4f} | '
          f'full {full:+.4f}')
    print(f'solo+pair share of full: {100*share:.0f}%  '
          f'(layer 1 was 24%)')
    held_=share>=0.70
    out['pred_held']=bool(held_)
    print(f"registered (>=70%): {'HELD' if held_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
