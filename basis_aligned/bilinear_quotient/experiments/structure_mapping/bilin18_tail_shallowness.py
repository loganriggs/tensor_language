"""Complete the shallowness map: the same statistic for tail layers 5-15.

§48: solo+pair share orders the six profiled layers exactly as compressibility does
(L1 24% ... L17 100%). The tail (5-15) has small individual effects and known
shift-fragility. REGISTERED PREDICTION: every tail layer's solo+pair share lands
ABOVE layer 3's 57% -- the deep-interaction regime is exclusive to layers 1-3, and
depth of interaction peaks where the causal mass does. (Shares for layers with tiny
full-span costs (<0.02) are noise-dominated and will be reported but excluded from
the prediction: expected exclusions around layers 6, 14, 15.)"""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
BANDS=((0,32),(32,128),(128,256),(256,512),(512,1152))
LAYERS=tuple(range(5,16))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_tail_shallowness_results.json')

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
        share=(sum(solo.values())+psum)/full if abs(full)>1e-9 else float('nan')
        out['layers'][li]={'solo_sum':sum(solo.values()),'pair_excess':psum,
                           'full':full,'share':share}
        print(f'layer {li:2d}: full {full:+.4f} -> solo+pair share '
              f'{100*share:.0f}%{" (noise-dominated)" if abs(full)<0.02 else ""}',
              flush=True)
    ok=[li for li,v in out['layers'].items()
        if abs(v['full'])>=0.02 and v['share']>0.57]
    tested=[li for li,v in out['layers'].items() if abs(v['full'])>=0.02]
    out['tested']=tested; out['above_57']=ok
    pred=len(ok)==len(tested) and len(tested)>0
    out['pred_held']=bool(pred)
    print(f"\ntested (|full|>=0.02): {tested}; above 57%: {ok}")
    print(f"registered (all tested tail layers above layer 3's 57%): "
          f"{'HELD' if pred else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
