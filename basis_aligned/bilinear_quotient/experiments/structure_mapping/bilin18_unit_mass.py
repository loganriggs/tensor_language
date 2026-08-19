"""Unit-mass spectrum -- the gauge-invariant unit-importance question the program
never asked. Per spec, m_i = (|a_i||b_i||c_i|)^(1/3) is the invariant mass of
hidden unit i; its cube weights the unit's rank-1 term in T. Per layer: the
effective number of active units, PR = (sum m_i^3)^2 / sum m_i^6, out of h=4608.

REGISTERED PREDICTIONS (continuing the diffuseness theme): (a) no unit
concentration -- PR >= 30% of h at every layer; alternative: any layer with PR
< 10% would be the program's first unit-level concentration. (b) depth
uniformity: max/min PR across layers <= 2 (unit usage is homogeneous, like
everything else within component types)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import m
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_unit_mass_results.json')

def main():
    t0=time.time()
    out={'layers':{}}; prs=[]
    for li,blk in enumerate(m.transformer.h):
        A=blk.mlp.Left.weight.detach().float()
        B=blk.mlp.Right.weight.detach().float()
        C=blk.mlp.Down.weight.detach().float()
        mi=(A.norm(dim=1)*B.norm(dim=1)*C.norm(dim=0))**(1/3)
        w=mi**3
        pr=float(w.sum()**2/(w**2).sum())
        h=w.numel()
        out['layers'][str(li)]={'pr':pr,'frac':pr/h}
        prs.append(pr)
        print(f'L{li:2d}: effective units {pr:７.0f} / {h} ({pr/h:.0%})'
              .replace('７','7'),flush=True)
    fr=[p/4608 for p in prs]
    pa=min(fr)>=0.30; conc=min(fr)<0.10
    pb=max(prs)/min(prs)<=2
    out['pred_a']=bool(pa); out['concentration_flag']=bool(conc)
    out['pred_b']=bool(pb)
    print(f"\n(a) no unit concentration (>=30% everywhere): {'HELD' if pa else 'FAILED'} "
          f"(min {min(fr):.0%})")
    if conc: print('ALTERNATIVE: unit-level concentration found')
    print(f"(b) depth-uniform (max/min <=2): {'HELD' if pb else 'FAILED'} "
          f"({max(prs)/min(prs):.2f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
