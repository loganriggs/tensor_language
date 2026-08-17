"""Scope the corrected claim (section 209: behavioral vocabulary sharing is
writer-general, L0 0.70 / L1 0.64 / L9 0.54). Those are the three STRONG
writers. Does sharing extend to quiet middle writers (L6, L12 -- solo deletion
costs near zero), or does it track writer power? Same instrument: activation-
weighted LORO over the writer's top-48 output coords, readers (2,3,5,9,13,17)
minus any reader equal to the writer, fresh evaluation rows 384-448.

REGISTERED PREDICTIONS: (a) both weak writers >= 0.40 (sharing is a property
of the reader population, fully writer-general); (b) random basis <= 0.1.
Alternative if (a) fails: sharing tracks writer strength -- a scoping note on
section 209, not a correction (it claimed strong writers only)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_behavioral_writers import loro

OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_weak_writers_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    res={}
    for Wl,readers in ((6,(2,3,5,9,13,17)),(12,(2,3,5,9,13,17))):
        med,rnd=loro(Wl,readers)
        res[Wl]=(med,rnd)
        print(f'writer L{Wl}: behavioral LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    pa=all(v[0]>=0.40 for v in res.values())
    pb=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]} for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\n(a) both weak writers >=0.40: {'HELD -- fully writer-general' if pa else 'FAILED -- sharing tracks writer strength'}")
    print(f"(b) randoms <=0.1: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
