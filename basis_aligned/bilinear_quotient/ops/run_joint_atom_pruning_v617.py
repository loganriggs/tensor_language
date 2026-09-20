#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_gram_replay pred_b_full_identity pred_c_half_budget_fidelity
"""Native wider sparse bilinear family, weights-only folded tensor screen.

Keep original bilinear atoms, with all output directions. Compare greedy
minimum incremental removal error, largest atom norm and three seeded random
orders. No fitting. Gram includes every cancellation between removed atoms.
Frozen gates: selected explicit tensor Gram replay <=2e-5; full representation
has zero removal error; at <=2304 retained atoms one method achieves <=.10
relative full-tensor error. Last gate is a discovery hypothesis, not expected
to pass. Negative result does not rule out basis changes or shared DAGs.
PRICE 0 native forwards, 0 native backwards/updates, 0 fits.
"""
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'basis_aligned/bilinear_quotient/circuits/followups/joint_atom_pruning_v617_result.json'
SNAP = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
PREDICTIONS = {'pred_a_gram_replay': '<=2e-5', 'pred_b_full_identity': 'zero removal error',
               'pred_c_half_budget_fidelity': '<=.10 at <=2304 retained atoms'}


def main():
    plan = dict(retained=[64,128,256,512,1024,2048,2304,3072,4096,4608],
                forwards_max=0, model_backwards=0, model_updates=0, fit_parameters=0,
                random_seeds=[0,1,2], execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan)); return
    import torch
    import disk_guard
    sys.path.insert(0, str(ROOT/'basis_aligned/polynomial_causal'))
    from joint_atom_pruning import atom_gram, greedy_removal_order, removal_energy
    tic=time.perf_counter()
    torch.set_grad_enabled(False); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(SNAP,map_location='cpu',weights_only=True,mmap=True)
    def w(name): return sd[name].float().cuda()
    U=w('lm_head.weight'); D=w('transformer.h.17.mlp.Down.weight')
    L=w('transformer.h.17.mlp.Left.weight'); R=w('transformer.h.17.mlp.Right.weight')
    Dprev=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0]
    O=w('transformer.h.17.attn.c_proj.weight')
    E=torch.cat([torch.eye(L.shape[1],device='cuda'),Dprev,O],1)
    Qu,Ru=torch.linalg.qr(U); Qe,Re=torch.linalg.qr(E.T)
    C,A,B=Ru@D,L@Re.T,R@Re.T
    gram=atom_gram(C,A,B).double()
    gram=(gram+gram.T)/2
    total=gram.sum()
    # Independent FP64 selected-atom contraction in original residual factors;
    # this checks both ambient reduction and Gram construction, not self-replay.
    ids=torch.tensor([0,17,511,2047,4607],device='cuda')
    cc=U.double()@D[:,ids].double()
    aa=L[ids].double()@E.double(); bb=R[ids].double()@E.double()
    reference=atom_gram(cc,aa,bb)
    replay=float((gram[ids][:,ids]-reference).norm()/reference.norm())
    del cc,aa,bb,Qu,Qe
    orders={'greedy':greedy_removal_order(gram), 'atom_norm':gram.diag().argsort()}
    for seed in plan['random_seeds']:
        orders[f'random_{seed}']=torch.randperm(len(gram),generator=torch.Generator().manual_seed(seed)).cuda()
    rows=[]
    for method,order in orders.items():
        for k in plan['retained']:
            kept=order[-k:]
            energy=removal_energy(gram,kept)
            if float(energy)<-1e-6*float(total): raise ValueError('negative squared error')
            rows.append(dict(method=method,retained=k,
                relative_tensor_error=float((energy.clamp_min(0)/total).sqrt()),
                folded_factor_values=k*(U.shape[0]+2*E.shape[1]),
                native_factor_values=U.numel()+E.numel()+k*(3*L.shape[1]),
                quadratic_products=k))
    half=[r for r in rows if r['retained']<=2304]
    best=min(half,key=lambda r:r['relative_tensor_error'])
    result=dict(schema='joint_atom_pruning_v617',plan=plan,rows=rows,
        selected_gram_fp64_relative_error=replay,
        atom_cancellation_ratio=float(total/gram.diag().sum()),
        best_half_budget=best, removal_orders={k:v.cpu().tolist() for k,v in orders.items() if not k.startswith('random')},
        predictions={'pred_a_gram_replay':replay<=2e-5,
                     'pred_b_full_identity':all(r['relative_tensor_error']==0 for r in rows if r['retained']==4608),
                     'pred_c_half_budget_fidelity':best['relative_tensor_error']<=.1},
        price_scope='two alternative storage grammars; native factor price includes U and E but not generation of z or normalization',
        scope='fixed native dictionary; no basis discovery, OOD, extraction, removal selectivity or reuse claim',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(1000000,label='v617 JSON')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('removal_orders','rows')},indent=2))
    print(json.dumps(rows))


if __name__=='__main__': main()
