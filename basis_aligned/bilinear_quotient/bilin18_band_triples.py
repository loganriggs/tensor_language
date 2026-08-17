"""Does the interaction hierarchy converge at order 3, or is layer 1 full-order?

§45: solo bands 0.24, pairwise excess 0.94, total 4.90 -- 80% of the interaction is
order >= 3. Measure all ten triples; Mobius gives the pure order-3 term. REGISTERED
PREDICTION: pure order-3 contributions sum to under half of the remaining 3.72 nats --
i.e. order-4+ still dominates and the layer is effectively full-order (holistic all
the way up). Bar: order-3 sum < 1.86."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_band_triples_results.json')

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
    pair={}
    for a,b in itertools.combinations(BANDS,2):
        pair[(a,b)]=val([a,b])
    o3sum=0.0
    out={'triples':{}}
    print(f"  {'triple':>14} {'joint':>8} {'pure order-3':>13}")
    for tri in itertools.combinations(BANDS,3):
        j=val(list(tri))
        m3=j
        for b in tri: m3-=solo[b]
        for a,b in itertools.combinations(tri,2):
            m3-=(pair[(a,b)]-solo[a]-solo[b])
        o3sum+=m3
        out['triples'][str(tuple(BANDS.index(b) for b in tri))]={'joint':j,'order3':m3}
        print(f"  {str(tuple(BANDS.index(b) for b in tri)):>14} {j:>+8.4f} "
              f"{m3:>+13.4f}",flush=True)
    out['order3_sum']=o3sum
    held_=o3sum<1.86
    out['pred_held']=bool(held_)
    print(f'\npure order-3 sum: {o3sum:+.4f} of the 3.72 remaining')
    print(f"registered (order-3 sum < 1.86 -> order-4+ dominates): "
          f"{'HELD' if held_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
