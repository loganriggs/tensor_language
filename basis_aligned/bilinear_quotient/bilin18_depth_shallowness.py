"""Is interaction shallowness the same property as compressibility, across the model?

§47: solo+pairwise share of the full-span deletion cost is 99% at layer 16 (the
compressible layer) and 24% at layer 1 (the uncompressible one). Measure the same
statistic at layers 0, 2, 3, 17. REGISTERED PREDICTION: across the six layers, the
solo+pair share correlates with the known leader-surrogate repair fraction
(L0 66%, L1 92%*, L2 3.5%, L3 68%, L16 ~100%, L17 whole-layer ~90%) at Spearman
>= 0.6 -- and specifically L2 and L3 land BELOW 50% share while L17 lands ABOVE 70%.
(*L1's repair is high but its layer-wide structure is deep; the cleaner pairing is
share vs the layer's own whole-layer compressibility from section 9-10: L0 partial,
L2/L3 none, L16/L17 yes. The ordinal prediction: L16, L17, L0 above; L1, L2, L3
below.)"""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
LAYERS=(0,2,3,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_depth_shallowness_results.json')

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
    out={'layers':{}}
    known={1:0.24,16:0.99}
    for li in LAYERS:
        Yb,Y=collect(li); _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans={b: orth(Vh[b[0]:b[1]].T) for b in BANDS}
        def val(bs):
            Q=torch.cat([spans[b] for b in bs],1)
            PATCH[li]=(Q,Yb@Q)
            try: return float((held()-base).mean())
            finally: PATCH.pop(li)
        solo={b: val([b]) for b in BANDS}
        psum=sum(val([a,b])-solo[a]-solo[b]
                 for a,b in itertools.combinations(BANDS,2))
        full=val(list(BANDS))
        share=(sum(solo.values())+psum)/max(abs(full),1e-9)
        out['layers'][li]={'solo_sum':sum(solo.values()),'pair_excess':psum,
                           'full':full,'share':share}
        print(f'layer {li:2d}: solo {sum(solo.values()):+.4f} pair {psum:+.4f} '
              f'full {full:+.4f} -> solo+pair share {100*share:.0f}%',flush=True)
    allsh={**{li:v['share'] for li,v in out['layers'].items()},**known}
    print(f"\nall layers: {dict(sorted((k,round(v,2)) for k,v in allsh.items()))}")
    above={li:allsh[li] for li in (16,17,0) if li in allsh}
    below={li:allsh[li] for li in (1,2,3) if li in allsh}
    pred=min(above.values())>max(below.values())
    out['ordinal_prediction_held']=bool(pred)
    print(f"ordinal prediction (16,17,0 all above 1,2,3): "
          f"{'HELD' if pred else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
