"""Is bilin18's privacy notch one layer wide, like bilin12's? bilin12's full
profile shows a sharp notch (L3 +0.32, L4 -0.08, L5 +0.38). bilin18's
measured writers are 0,1,6,9,12 (0.70/0.64/0.16/0.54/0.51) -- the private
writer's neighbors L5 and L7 (and L3 for shape) were never scanned.
Behavioral LORO, same instrument as bilin18_behavioral_writers (readers
2,3,5,9,13,17 minus self; the L5 fold set swaps 5->15 to avoid overlap).

REGISTERED PREDICTIONS: (a) L5 and L7 both >= 0.40 (the notch is sharp in
bilin18 too); (b) both exceed L6's 0.16 by >= 0.20; (c) the measured
random-basis null (per the section-221 rule, reported not assumed) <= 0.1
for every writer -- the 80/1176-dim regime has a low floor, unlike the
rank-18/36-dim regime that produced ledger #17."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_behavioral_writers import loro

OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_notch_profile_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    res={}
    for Wl,readers in ((3,(2,5,9,13,17,15)),(5,(2,3,9,13,17,15)),
                       (7,(2,3,5,9,13,17))):
        med,rnd=loro(Wl,readers)
        res[Wl]=(med,rnd)
        print(f'writer L{Wl}: behavioral LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    pa=res[5][0]>=0.40 and res[7][0]>=0.40
    pb=res[5][0]>=0.36 and res[7][0]>=0.36
    pc=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]} for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) L5,L7 >=0.40 sharp notch: {'HELD' if pa else 'FAILED'}")
    print(f"(b) both exceed L6+0.20: {'HELD' if pb else 'FAILED'}")
    print(f"(c) measured nulls <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
