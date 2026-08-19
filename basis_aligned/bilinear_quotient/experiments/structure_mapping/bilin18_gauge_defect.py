"""Balanced-gauge defect survey for bilin18 (balanced_gauge_spec.md, deliverable
1.5, restricted to the bilin18 use case). Each MLP is Down((Left x) o (Right x))
== C((Ax) o (Bx)); per hidden unit i the gauge-invariant mass is
m_i = (|a_i||b_i||c_i|)^(1/3) and the balancedness defect is
delta_i = std(log|a_i|, log|b_i|, log|c_i|). Weight decay drives delta to zero
(Kempf-Ness / conservation law), so a trained-with-wd model should be
near-balanced -- which would certify that any raw-weight reading of bilin18 was
taken in a sane gauge. Program audit note: our published statistics are
T-level/activation-level/function-level (gauge-invariant); this survey is
certification, not correction.

REGISTERED PREDICTIONS: (a) near-balanced: m-weighted mean delta <= 0.15 at
every layer; alternative: any layer > 0.5 means that layer's raw weights are in
an arbitrary gauge (flag loudly per spec). (b) dead units (< 1e-10 relative
mass) < 1% everywhere. (c) sanity per spec test 4: balancing changes no m_i
(checked to 1e-5)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_gauge_defect_results.json')

def main():
    t0=time.time()
    out={'layers':{}}
    worst=0.0; worst_li=None; dead_max=0.0; m_ok=True
    for li,blk in enumerate(m.transformer.h):
        A=blk.mlp.Left.weight.detach().float()   # h x d  rows a_i
        B=blk.mlp.Right.weight.detach().float()  # h x d
        C=blk.mlp.Down.weight.detach().float()   # d x h  columns c_i
        na=A.norm(dim=1); nb=B.norm(dim=1); nc=C.norm(dim=0)
        mi=(na*nb*nc).clamp_min(1e-30)**(1/3)
        scale=float(mi.max())
        dead=(mi<=1e-10*scale)
        logs=torch.stack([na.clamp_min(1e-30).log(),
                          nb.clamp_min(1e-30).log(),
                          nc.clamp_min(1e-30).log()])
        delta=logs.std(dim=0,unbiased=False)
        w=mi**3
        wmean=float((delta*w).sum()/w.sum())
        dmax=float(delta[~dead].max()) if (~dead).any() else 0.0
        # spec test 4: rescale to balanced and confirm m_i unchanged
        with torch.no_grad():
            a2=A*(mi/na.clamp_min(1e-30))[:,None]
            b2=B*(mi/nb.clamp_min(1e-30))[:,None]
            c2=C*(mi/nc.clamp_min(1e-30))[None,:]
            mi2=(a2.norm(dim=1)*b2.norm(dim=1)*c2.norm(dim=0))**(1/3)
            m_ok &= bool(torch.allclose(mi,mi2,rtol=1e-5,atol=1e-8))
        out['layers'][str(li)]={'weighted_mean_delta':wmean,'max_delta':dmax,
                                'dead_frac':float(dead.float().mean()),
                                'median_m':float(mi.median())}
        if wmean>worst: worst=wmean; worst_li=li
        dead_max=max(dead_max,float(dead.float().mean()))
        print(f'L{li:2d}: weighted-mean delta {wmean:.4f} | max {dmax:.3f} | '
              f'dead {float(dead.float().mean()):.4%} | median m {float(mi.median()):.3f}',
              flush=True)
    pa=all(v['weighted_mean_delta']<=0.15 for v in out['layers'].values())
    flag=any(v['weighted_mean_delta']>0.5 for v in out['layers'].values())
    pb=dead_max<0.01; pc=m_ok
    out['pred_a']=bool(pa); out['flag_arbitrary_gauge']=bool(flag)
    out['pred_b']=bool(pb); out['sanity_c']=bool(pc)
    out['worst_layer']=worst_li
    print(f"\n(a) near-balanced everywhere (<=0.15): {'HELD' if pa else 'FAILED'} "
          f"(worst L{worst_li} at {worst:.3f})")
    if flag: print('ALTERNATIVE: some layer > 0.5 -- raw weights in arbitrary gauge')
    print(f"(b) dead units <1%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) m_i invariance sanity: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
