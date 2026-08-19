"""Completing the functional-nonlinearity map at the front. Section 106: L2 is the
most functionally nonlinear layer measured (+0.109); L0, L1, L3 were never
linearized. L1 is the program's hardest-read layer (densest interactions, source
of the functional vocabulary).

Consistent-protocol individual costs for L0, L1, L3 (fit and apply inside the
same forward). REGISTERED PREDICTIONS: (a) L1 >= 0.15 (the hardest-read layer is
the most functionally nonlinear); (b) the full front map is unimodal with its
peak at L1 or L2 (L0 < peak > L4, no second local max); (c) all front costs
exceed the mid/tail median +0.036."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_front_map_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    PR.LINS={}
    base=PR.ce_eval()
    print(f'base {base:.4f}\n',flush=True)
    costs={0:None,1:None,3:None}
    for li in (0,1,3):
        PR.LINS={li:PR.fit_layer(li)}
        costs[li]=PR.ce_eval()-base
        PR.LINS={}
        print(f'L{li}: consistent cost +{costs[li]:.4f}',flush=True)
    known={2:0.1091,4:0.0542}
    full={**{k:v for k,v in costs.items()},**known}
    seq=[full[k] for k in sorted(full)]
    peak=max(full,key=full.get)
    unimodal=all(seq[i]<=seq[i+1] for i in range(sorted(full).index(peak))) and \
             all(seq[i]>=seq[i+1] for i in range(sorted(full).index(peak),len(seq)-1))
    pa=costs[1]>=0.15
    pb=peak in (1,2) and unimodal
    pc=all(v>0.036 for v in costs.values())
    out={'costs':{str(k):v for k,v in full.items()},'peak':peak,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\nfront map: '+' | '.join(f'L{k} +{full[k]:.3f}' for k in sorted(full)))
    print(f"(a) L1 >= 0.15: {'HELD' if pa else 'FAILED'}")
    print(f"(b) unimodal peak at L1/L2: {'HELD' if pb else 'FAILED'} (peak L{peak})")
    print(f"(c) front all above mid/tail median: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
