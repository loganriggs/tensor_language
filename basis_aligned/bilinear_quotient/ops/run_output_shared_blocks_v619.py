#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_block_energy pred_b_component_replay pred_c_native_product_budget
"""Learn signed-square features in full output-sharing quadratic blocks.

All1152 output modes; chunk8 scalar quadratics, each diagonalized exactly.
No optimizer/stopping budget: closed-form spectral dictionary, global optimal
square allocation within that fixed dictionary. No claim of optimal rotation.
Frozen gates: summed block energy matches implicit tensor energy <=3e-5;
first block eigen-reconstruction replay <=3e-5;4608 new square products reach
<=.10 relative global tensor error. Save strongest output component with99%
quadratic energy for subsequent normalized-model tests; not a circuit claim.
PRICE0 model forwards/backwards/updates,0 iterative fits.
"""
import os,json,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/output_shared_blocks_v619_result.json'
ARTIFACT=OUT.with_name('output_shared_blocks_v619_component.pt')
PREDICTIONS={'pred_a_block_energy':'<=3e-5','pred_b_component_replay':'<=3e-5',
             'pred_c_native_product_budget':'<=.10 at4608 squares'}


def main():
    plan=dict(budgets=[64,512,2048,4608,16384,65536],chunk=8,forwards_max=0,
        model_backwards=0,model_updates=0,fit_parameters=0,execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    from run_joint_atom_pruning_v617 import SNAP
    sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
    from joint_folded_tucker import mode_grams
    from output_shared_quadratics import quadratic_blocks,allocate_squares
    tic=time.perf_counter();torch.set_num_threads(8);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(SNAP,map_location='cpu',weights_only=True,mmap=True)
    def w(n):return sd[n].float().cuda()
    U=w('lm_head.weight');D=w('transformer.h.17.mlp.Down.weight')
    L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight')
    E=torch.cat([torch.eye(1152,device='cuda'),
        w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],
        w('transformer.h.17.attn.c_proj.weight')],1)
    Qu,Ru=torch.linalg.qr(U);Qe,Re=torch.linalg.qr(E.T)
    C,A,B=Ru@D,L@Re.T,R@Re.T
    go,_=mode_grams(C,A,B);eo,W=torch.linalg.eigh(go)
    W=W.flip(1);eo=eo.flip(0);total=go.trace().double()
    values=[];component=None;replay=None
    for start in range(0,1152,plan['chunk']):
        blocks=quadratic_blocks(C,A,B,W[:,start:start+plan['chunk']])
        if start==0:
            vals,vecs=torch.linalg.eigh(blocks[0])
            reconstructed=(vecs*vals)@vecs.T
            replay=float((blocks[0]-reconstructed).norm()/blocks[0].norm())
            order=vals.square().argsort(descending=True)
            energy=vals[order].double().square().cumsum(0)
            rank=int(torch.searchsorted(energy,.99*energy[-1]))+1
            ix=order[:rank]
            component=dict(readers=(Qe@vecs[:,ix]).cpu(),coefficients=vals[ix].cpu(),
                writer=(Qu@W[:,0]).cpu(),input_port='z=[r,m16,a17]',
                normalization='divide by mean(h**2)+native_eps with actual h=Ez',
                scope='strongest output-mode component only; upstream states still required',
                retained_block_energy=float(energy[rank-1]/energy[-1]),
                full_tensor_energy_fraction=float(energy[rank-1]/total))
        values.append(torch.linalg.eigvalsh(blocks))
        if start%128==0:print('output blocks',start,'of1152',flush=True)
    values=torch.cat(values)
    mismatch=float((values.double().square().sum()-total).abs()/total)
    rows=[]
    for budget in plan['budgets']:
        ids,active,energy=allocate_squares(values,budget)
        rows.append(dict(squares=budget,active_output_features=len(active),
            relative_tensor_error=float((1-energy/total).clamp_min(0).sqrt()),
            stored_values=E.shape[1]*budget+U.shape[0]*len(active)+budget,
            alternative_factored_values=E.numel()+U.numel()+1152*budget+1152*len(active)+budget))
    disk_guard.guard_torch_save(component,str(ARTIFACT),label='v619 single output component')
    result=dict(plan=plan,rows=rows,energy_relative_mismatch=mismatch,component_replay=replay,
        component={k:v for k,v in component.items() if not torch.is_tensor(v)},
        component_squares=component['coefficients'].numel(),artifact_bytes=ARTIFACT.stat().st_size,
        artifact_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
        predictions={'pred_a_block_energy':mismatch<=3e-5,'pred_b_component_replay':replay<=3e-5,
            'pred_c_native_product_budget':next(r for r in rows if r['squares']==4608)['relative_tensor_error']<=.1},
        scope='new features in fixed output spectral frame; no input sharing across blocks optimized; no behavioral/circuit claim',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000,label='v619 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
