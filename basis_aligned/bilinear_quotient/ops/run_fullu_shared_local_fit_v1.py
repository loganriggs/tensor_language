#!/usr/bin/env python3
# BQGATE:0bodyforwards;0texttokens;2capacities;10starts;3promoted;3900seconds.
"""Weight-only shared/private output graph:10 starts per capacity,3 promoted.
pred_a native reused-encoder replay <=1e-8 and compiled FP32 function error <=1e-5.
pred_b >=2 promoted starts per capacity locally converged: intrinsic gradient<=1e-6,
and zero changed assignments over the final two sweeps. No global guarantee.
pred_c each capacity reduces squared full coefficient error >=10% versus optimal
global factorization at same literal byte budget. Null: weak or unconverged fit.
Per capacity1800seconds,3screen sweeps/start,240additional sweeps/promoted start.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P))
import torch
import shared_local_subspaces_v1 as solver
from shared_local_gram_bank_v1 import gram_bank
from shared_local_reused_encode_v1 import reused_encode
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross
from fullu_shared_local_feasibility_v1 import price
STEM='FULLU_SHARED_LOCAL_FIT_V1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(f)==sha for f,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0forwards0tokens;2capacities10starts3screensweeps3promoted;1800seconds/capacity;3900hard');return
    assert not (P/(STEM+'_RESULT.json')).exists()
    signal.alarm(3900);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].to(device='cuda',dtype=torch.float64)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].to(device='cuda',dtype=torch.float64)
              for key in ['Left','Right','Down']]
    metric=down@product_cross(l,r,l,r)@down.T
    root=torch.linalg.cholesky((metric+metric.T)/2);mean=u.mean(0);x=(u-mean)@root
    energy=float(x.square().sum());mean_energy=float(len(u)*(mean@metric@mean))
    reference=json.loads((P/'FULLU_SHARED_LOCAL_FEASIBILITY_V1_RESULT.json').read_text())
    assert abs((energy+mean_energy)/reference['coefficient_energy']-1)<=1e-8
    x=x/energy**.5;fraction=energy/(energy+mean_energy)
    del u,l,r,down,metric,sd
    eigen=torch.linalg.eigvalsh(x.T@x).flip(0).clamp_min(0)
    solver.bank=gram_bank
    original=solver.initialize(x,64,32,8,9518)
    improved=reused_encode(x,original['global_bank'],original['local_banks'])
    replay=float((sum(solver.parts(original))-sum(solver.parts(improved))).norm())
    assert replay<=1e-8
    del original,improved
    solver.encode=reused_encode
    all_records=[]
    for g,groups,local in [(64,32,8),(128,64,16)]:
        begin=time.perf_counter();states=[];screens=[]
        def advance(state,limit):
            history=[];stable=0;gradient=None;reason='sweep_limit'
            previous=solver.objective(x,state)
            for step in range(limit):
                if time.perf_counter()-begin>=1800:
                    reason='time_limit';break
                old_labels=state['labels']
                updated=solver.sweep(x,state);value=solver.objective(x,updated)
                assert value<=previous+1e-8
                changed=int((old_labels!=updated['labels']).sum())
                stable=stable+1 if changed==0 else 0
                state=updated;previous=value
                history.append(dict(loss=value,changed_assignments=changed))
                if (step+1)%5==0 or stable>=2 or step==limit-1:
                    gradient=solver.stationarity(x,state)
                    if gradient['maximum_intrinsic_gradient']<=1e-6 and stable>=2:
                        reason='local_convergence';break
                if (step+1)%10==0:
                    print(json.dumps(dict(phase='refine_progress',g=g,sweeps=step+1,**history[-1])),flush=True)
            if gradient is None or reason!='local_convergence':gradient=solver.stationarity(x,state)
            return state,dict(loss=previous,sweeps=len(history),history=history,
                              **gradient,converged=reason=='local_convergence',terminal=reason)
        for seed in range(9518,9528):
            state=solver.initialize(x,g,groups,local,seed)
            initial=solver.objective(x,state)
            state,report=advance(state,3)
            states.append(state);screens.append(dict(seed=seed,initial_loss=initial,**report))
            print(json.dumps(dict(phase='screen',g=g,seed=seed,loss=report['loss'])),flush=True)
        chosen=sorted(range(10),key=lambda i:screens[i]['loss'])[:3]
        promoted=[]
        for i in chosen:
            states[i],report=advance(states[i],240)
            promoted.append(dict(start=i,seed=screens[i]['seed'],**report))
            print(json.dumps(dict(phase='promoted',g=g,start=i,**{k:v for k,v in report.items() if k!='history'})),flush=True)
        best=min(range(10),key=lambda i:solver.objective(x,states[i]));state=states[best]
        loss=solver.objective(x,state);costs=price(len(x),x.shape[1],g,groups,local)
        baseline=float(eigen[costs['matched_global_rank']:].sum())*fraction
        def reader(bank):
            return (torch.linalg.solve_triangular(root.T,bank.T,upper=True).T*energy**.5).float()
        program=dict(mean=mean.float(),global_reader=reader(state['global_bank']),
                     local_readers=[reader(b) for b in state['local_banks']],
                     global_codes=state['global_codes'].float(),local_codes=state['local_codes'].float(),
                     labels=state['labels'].to(torch.int32))
        restored=dict(global_bank=program['global_reader'].double()@root/energy**.5,
                      local_banks=[b.double()@root/energy**.5 for b in program['local_readers']],
                      global_codes=program['global_codes'].double(),local_codes=program['local_codes'].double(),
                      labels=program['labels'].long())
        error=sum(solver.parts(restored))-sum(solver.parts(state))
        error+=(program['mean'].double()-mean)@root/energy**.5
        compiled_error=float(error.norm())*fraction**.5
        assert compiled_error<=1e-5
        def cpu(value):
            if isinstance(value,torch.Tensor):return value.cpu()
            return [cpu(v) for v in value]
        torch.save({k:cpu(v) for k,v in program.items()},P/(STEM+f'_G{g}_PROGRAM.pt'))
        functions=[sum(solver.parts(states[i])).flatten() for i in chosen]
        cosines=[float(torch.dot(functions[i],functions[j])/(functions[i].norm()*functions[j].norm()))
                 for i in range(3) for j in range(i)]
        record=dict(global_width=g,groups=groups,local_width=local,screens=screens,promoted=promoted,
                    best_start=best,full_coefficient_squared_error=loss*fraction,
                    full_coefficient_relative_error=(loss*fraction)**.5,matched_global_squared_error=baseline,
                    relative_squared_gain=1-loss*fraction/baseline,compiled_fp32_error=compiled_error,
                    promoted_function_cosines=cosines,**costs,wall_seconds=time.perf_counter()-begin)
        write_json(P/(STEM+f'_G{g}_RESULT.json'),record);all_records.append(record)
        del states,state,functions,restored,program,error
    result={'pred_a':replay<=1e-8 and all(z['compiled_fp32_error']<=1e-5 for z in all_records),
            'pred_b':all(sum(q['converged'] for q in z['promoted'])>=2 for z in all_records),
            'pred_c':all(z['relative_squared_gain']>=.1 for z in all_records),
            'native_reused_encoder_replay':replay,'configurations':all_records,
            'wall_seconds':time.perf_counter()-started,'body_forwards':0,
            'scope':'Weights-only output graph; local stationarity not global recovery; no native behavior claim.'}
    write_json(P/(STEM+'_RESULT.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='configurations'}),flush=True)


if __name__=='__main__':main()
