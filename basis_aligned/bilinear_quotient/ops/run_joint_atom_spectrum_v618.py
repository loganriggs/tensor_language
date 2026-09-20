#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fp64_energy pred_b_positive_atom_gram pred_c_half_budget_obstruction
"""FP64 atom-spectrum bound: any native subset, permitting scalar refitting.

Frozen gates: FP64 energy/diagonal-energy ratio agrees with v617 to 2e-5; unit-atom Gram
minimum eigenvalue >1e-6; spectral lower bound at2304 retained atoms >.10.
No training/data/forward calls. PRICE0 forwards,0 fits,0 native backwards.
This is a floating-point numerical bound, not an interval certificate. It
constrains this fixed dictionary only, not new directions or vector writers.
"""
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/joint_atom_spectrum_v618_result.json'
PREDICTIONS={'pred_a_fp64_energy':'relative <=2e-5', 'pred_b_positive_atom_gram':'>1e-6',
             'pred_c_half_budget_obstruction':'>.10 at2304'}


def main():
    plan=dict(retained=[64,512,1024,2048,2304,3072,4096,4608],forwards_max=0,
              model_backwards=0,model_updates=0,fit_parameters=0,execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    from run_joint_atom_pruning_v617 import SNAP
    sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
    from joint_atom_pruning import subset_error_bound
    tic=time.perf_counter();torch.set_num_threads(8);torch.set_grad_enabled(False)
    sd=torch.load(SNAP,map_location='cpu',weights_only=True,mmap=True)
    def w(name):return sd[name].double().cuda()
    U=w('lm_head.weight');D=w('transformer.h.17.mlp.Down.weight')
    L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight')
    Dprev=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0]
    O=w('transformer.h.17.attn.c_proj.weight')
    metric=torch.eye(L.shape[1],dtype=L.dtype,device=L.device)+Dprev@Dprev.T+O@O.T
    output=D.T@(U.T@U)@D
    aa=L@metric@L.T;bb=R@metric@R.T;ab=L@metric@R.T
    gram=output*(aa*bb+ab*ab.T)/2
    gram=(gram+gram.T)/2
    energy=gram.sum();ratio=float(energy/gram.diag().sum())
    previous=json.loads(OUT.with_name('joint_atom_pruning_v617_result.json').read_text())
    disagreement=abs(ratio/previous['atom_cancellation_ratio']-1)
    print('FP64 Gram built; diagonal-normalized energy ratio',ratio,flush=True)
    eigenvalues,bounds=subset_error_bound(gram,plan['retained'])
    # Relax the numerical eigenvalue by a stated margin; not an interval proof.
    minimum=float(eigenvalues[0]); margin=1e-6
    relaxed={k:v*(max(0.,minimum-margin)/max(minimum,1e-300))**.5 for k,v in bounds.items()}
    result=dict(plan=plan,minimum_unit_atom_eigenvalue=minimum,maximum_unit_atom_eigenvalue=float(eigenvalues[-1]),
        atom_cancellation_ratio_fp64=ratio,relative_energy_ratio_disagreement=disagreement,
        relative_error_lower_bounds=relaxed,eigenvalue_margin=margin,
        predictions={'pred_a_fp64_energy':disagreement<=2e-5,
            'pred_b_positive_atom_gram':minimum>1e-6,'pred_c_half_budget_obstruction':relaxed[2304]>.1},
        scope='numerical bound for all supports and arbitrary scalar coefficients in fixed native-atom dictionary; not new basis/writers or behavioral fidelity',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000,label='v618 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
