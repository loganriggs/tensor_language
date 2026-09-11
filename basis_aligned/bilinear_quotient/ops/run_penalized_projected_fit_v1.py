#!/usr/bin/env python3
# BQGATE: 0forwards0seq; complete weight-only objective, no text.
"""pred_a numerical; pred_b joint convergence; pred_c penalized gain;
pred_d capture/component budget; pred_e whole-function stability.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import numpy as np
import torch
from native_reader_msp_generalization_v1 import CK
from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram
from structured_branch_amplitudes_v1 import inner
from folded_sparse_dictionary_v1 import decode
from penalized_projected_sparse_fit_v1 import PenalizedObjective,fit
PREFIX='PENALIZED_PROJECTED_FIT_V1';PENALTY=.0018823354889938328


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    for name in ('PENALIZED_PROJECTED_SPARSE_V1_CONTROL','PENALIZED_SPARSE_FD_V1_AUDIT','PENALIZED_PROJECTED_FIT_V1_CONTROL'):
        assert all(json.loads((P/f'{name}.json').read_text())['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            seeds=[0,937],fit_seconds_per_seed=1200,iterations=2500,features=2304,sparsity=128,
            penalty=PENALTY,parent='Final original unpenalized parents; original supports',
            objective='Full coefficient error plus summed component energies')));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(3600)
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    parent_path=P/'PROJECTED_SPARSE_DICTIONARY_FIT_V1_RESULT.json'
    parent=json.loads(parent_path.read_text());parent_sha=digest(parent_path)
    assert parent['predictions']['pred_a_instrument']
    assert parent['binding']==json.loads((P/'PROJECTED_SPARSE_DICTIONARY_FIT_V1_BINDING.json').read_text())
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda()
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double().cuda()
    wh=torch.linalg.cholesky(metric).T;native=(l,r,wh@down)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total'];rows=[];functions=[]
    for seed in (0,937):
        baseline=next(a for a in parent['arms'] if a['seed']==seed);source=baseline['cache']
        assert digest(source['path'])==source['sha256']
        saved=torch.load(source['path'],weights_only=True,map_location='cpu')
        raw=saved['analysis_basis'].double().cuda();ids=saved['code_indices'].long().cuda()
        values=saved['code_values'].double().cuda();values/=values.norm(dim=1,keepdim=True)
        assert raw.shape==(2304,1152) and ids.shape==values.shape==(9216,128)
        objective=PenalizedObjective((l,r,down),wh,raw,ids,values,total,PENALTY)
        tic=time.perf_counter();initial,g=objective.evaluate(objective.initial);initial_seconds=time.perf_counter()-tic
        initial_details=objective.last['details']
        def components(point,writer):
            rb,rv=objective.unpack(point);cv=rv/rv.norm(dim=1,keepdim=True)
            readers,basis,_,_=decode(rb,ids,cv,torch.ones(len(ids),device='cuda'))
            a,b=readers.chunk(2);w=wh@writer;cp=(a,b,w)
            residual=float((inner(cp,cp)-2*inner(native,cp)+total)/total)
            component=float((w.square().sum(0)*.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())).sum()/total)
            return a,b,basis,cv,residual,component
        *_,ires,icomp=components(objective.initial,objective.last['writer'])
        initial_replay=abs(initial-ires-PENALTY*icomp)
        budget_replay=0.
        if seed==0:
            budget=json.loads((P/'OUTPUT_COMPONENT_BUDGET_V1_AUDIT.json').read_text())
            assert abs(budget['penalty']-PENALTY)<=1e-15
            budget_replay=abs(initial-(1-budget['capture']+PENALTY*budget['component_energy']))
        direction=-g/max(np.linalg.norm(g),1e-30);slope=float(g@direction);h=1e-6
        fd=abs((objective.evaluate(objective.initial+h*direction)[0]-objective.evaluate(objective.initial-h*direction)[0])/(2*h)-slope)/max(1.,abs(slope))
        preflight=dict(seed=seed,fd_error=fd,initial_cp_replay=initial_replay,budget_replay=budget_replay,
            initial_penalized_objective=initial,initial_capture=1-ires,initial_component_energy=icomp,
            initial_gradient_seconds=initial_seconds,parent_capture=baseline['final_capture'],parent_converged=baseline['optimization']['converged'])
        write(P/f'{PREFIX}_SEED_{seed}_PREFLIGHT.json',preflight);print(json.dumps(preflight),flush=True)
        assert fd<=1e-6 and max(initial_replay,budget_replay)<=1e-8
        cache=Path(f'/dev/shm/bilin18_penalized_projected_fit_v1_s{seed}.pt');assert not cache.exists()
        def checkpoint(point,current,history):
            rb,rv=objective.unpack(point);cv=rv/rv.norm(dim=1,keepdim=True)
            basis=rb/rb.norm(dim=1,keepdim=True);tmp=cache.with_suffix('.tmp')
            torch.save(dict(analysis_basis=basis.cpu(),raw_basis=rb.cpu(),raw_values=rv.cpu(),
                code_indices=ids.to(torch.int16).cpu(),code_values=cv.cpu(),down=current['writer'].cpu(),
                retained_bias_key='transformer.h.17.mlp.Down_bias',source=source,binding=binding,
                penalty=PENALTY,history=history,loss=current['value'],details=current['details'],
                all_native_weight_rows_in_objective=True,lbfgs_history_restorable=False),tmp)
            os.replace(tmp,cache)
        point,writer,optimization=fit(objective,seconds=1200,max_iterations=2500,checkpoint=checkpoint)
        fitpath=P/f'{PREFIX}_SEED_{seed}_FIT.json';write(fitpath,optimization)
        a,b,basis,cv,residual,component=components(point,writer)
        replay=abs(residual+PENALTY*component-optimization['final_loss'])
        program=RectangularSparseReaderProgram(basis,ids,cv,writer,bias)
        x=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(924))
        direct=((x@a.T)*(x@b.T))@writer.T+bias
        execution=float((program(x)-direct).norm()/direct.norm())
        uniterror=max(float((basis.norm(dim=1)-1).abs().max()),float((cv.norm(dim=1)-1).abs().max()))
        normal=max([initial_details['output_solve']['normal_residual'],optimization['final_details']['output_solve']['normal_residual']]+
                   [h['details']['output_solve']['normal_residual'] for h in optimization['history']])
        numeric=max(replay,execution,normal,initial_replay,budget_replay)<=1e-8 and uniterror<=1e-6 and fd<=1e-6 and optimization['maximum_accepted_loss_increase']<=1e-10
        row=dict(seed=seed,instrument_passed=numeric,preflight=preflight,converged=optimization['converged'],
            fit_receipt=dict(path=str(fitpath),sha256=digest(fitpath)),final_capture=1-residual,
            final_component_energy=component,final_objective=optimization['final_loss'],
            joint_objective_gain=initial-optimization['final_loss'],parent_capture=baseline['final_capture'],
            independent_cp_replay=replay,executor_replay=execution,unit_error=uniterror,normal_residual=normal,
            price=program.price(),source=source,cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size))
        write(P/f'{PREFIX}_SEED_{seed}.json',row);rows.append(row);print(json.dumps(row),flush=True)
        functions.append((a.clone(),b.clone(),(wh@writer).clone()))
        del objective,program,saved,optimization,a,b,basis,cv,raw,values
    energies=[inner(f,f) for f in functions];cosine=float(inner(*functions)/(energies[0]*energies[1]).sqrt())
    predictions={'pred_a_instrument':all(r['instrument_passed'] for r in rows),
        'pred_b_convergence':all(r['converged'] for r in rows),
        'pred_c_joint_gain':all(r['joint_objective_gain']>=1e-4 for r in rows),
        'pred_d_capture_and_energy':all(r['final_capture']>=r['parent_capture']-.001 and r['final_component_energy']<=1 for r in rows),
        'pred_e_stability':cosine>=.9}
    write(out,dict(predictions=predictions,arms=rows,function_cosine=cosine,penalty=PENALTY,binding=binding,
        parent_result_sha256=parent_sha,body_forwards=0,corpus_access=False,wall_seconds=time.perf_counter()-started,
        scope='Joint penalized full folded fit, original fixed graphs. No global optimum, fresh-weight or circuit claim.'))
    print(json.dumps(dict(predictions=predictions,function_cosine=cosine)),flush=True)


if __name__=='__main__':main()
