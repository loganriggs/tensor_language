#!/usr/bin/env python3
# BQGATE: 0forwards0seq; weight-only joint optimizer repair, no text.
"""pred_a numerical; pred_b joint convergence; pred_c gradient reduction;
pred_d full function gain; pred_e reader gain; pred_f function stability.
"""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import numpy as np
import torch
from run_overcomplete_ols_reencode_v1 import CK,digest,encode,RectangularSparseReaderProgram,inner
from active_orthant_dictionary_v1 import OrthantObjective,polish
from quadratic_token_dictionary_v1 import conditional,diagnostics
PREFIX='NATIVE_COUPLED_L1_POLISH_V1'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def preflight():
    gen=torch.Generator().manual_seed(920)
    x=torch.randn(12,8,generator=gen);b=torch.randn(16,8,generator=gen);b/=b.norm(dim=1,keepdim=True)
    z=torch.randn(12,16,generator=gen);z[z.abs()<.5]=0
    cpu=OrthantObjective(x,z,b,.05);gpu=OrthantObjective(x.cuda(),z.cuda(),b.cuda(),.05)
    cv,cg=cpu.value_gradient(cpu.initial);gv,gg=gpu.value_gradient(cpu.initial)
    error=abs(cv-gv)/max(abs(cv),1e-30);gradient=float(np.linalg.norm(cg-gg)/np.linalg.norm(cg))
    result=dict(objective_error=error,gradient_error=gradient,passed=max(error,gradient)<=1e-9)
    assert result['passed'];return result


def fit(x,parent,seed,binding):
    source=parent['cache'];assert digest(source['path'])==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu')
    assert saved['seed']==seed and saved['penalty']==.05
    b=saved['dictionary'].double().cuda();z=saved['codes'].double().cuda()
    assert b.shape==(2304,1152) and z.shape==(6144,2304)
    initial=diagnostics(x,z,b,.05);replay=abs(initial['objective']-parent['final']['objective'])
    assert replay<=1e-10
    history=[];checks=[];converged=False;path=Path(f'/dev/shm/bilin18_native_coupled_l1_polish_v1_s{seed}_fit.pt')
    assert not path.exists();started=time.perf_counter();del saved
    for iteration in range(1,21):
        remaining=600-(time.perf_counter()-started)
        if remaining<=0:break
        before=initial if not checks else checks[-1]
        z,b,report=polish(x,z,b,.05,max_iterations=200,seconds=min(60.,remaining))
        z,refresh=conditional(z,b@b.T,x@b.T,.05,'codes',1000,1e-7)
        check=diagnostics(x,z,b,.05);checks.append(check)
        change=None if len(checks)<2 else abs(checks[-1]['objective']-checks[-2]['objective'])/max(abs(checks[-1]['objective']),1e-30)
        converged=len(checks)>=2 and all(c['relative_stationarity']<=1e-5 for c in checks[-2:]) and change<=1e-8
        numeric=(check['objective']<=before['objective']+1e-10 and check['atom_norm_max']<=1+1e-6
                 and report['canonicalization']['reconstruction_error']<=1e-10
                 and all(np.isfinite(v) for v in check.values()))
        row=dict(round=iteration,seconds=time.perf_counter()-started,polish=report,code_refresh=refresh,
                 check=check,relative_objective_change=change,instrument_passed=bool(numeric))
        history.append(row)
        temp=path.with_suffix('.tmp');torch.save(dict(dictionary=b.cpu(),codes=z.cpu(),seed=seed,penalty=.05,
            binding=binding,source=source,history=history,initial=initial),temp);os.replace(temp,path)
        print(json.dumps(dict(seed=seed,phase='fit',round=iteration,seconds=row['seconds'],check=check,
            converged=converged,evaluations=report['evaluations'],polish_seconds=report['seconds'])),flush=True)
        assert numeric
        if converged:break
    final=checks[-1] if checks else initial
    result=dict(seed=seed,initial=initial,final=final,converged=converged,
        stop='converged' if converged else 'time_or_round_limit',seconds=time.perf_counter()-started,
        history=history,parent_objective_replay=replay,
        stationarity_reduction=initial['relative_stationarity']/max(final['relative_stationarity'],1e-30),
        cache=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size))
    write(P/f'{PREFIX}_SEED_{seed}_FIT.json',result);return b,result


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    assert all(json.loads((P/'ACTIVE_ORTHANT_DICTIONARY_V1_CONTROL.json').read_text())['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            seeds=[0,937],fit_seconds_per_seed=600,features=2304,penalty=.05,sparsity=128,
            optimizer='coupled L-BFGS-B plus global proximal refresh',new_structural_assumption=False)));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(2100)
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter();check=preflight()
    write(P/f'{PREFIX}_PREFLIGHT.json',check)
    parent_path=P/'OVERCOMPLETE_L1_READER_V1_RESULT.json';parent=json.loads(parent_path.read_text())
    repaired_path=P/'OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json';repaired=json.loads(repaired_path.read_text())
    assert repaired['predictions']['pred_a_instrument']
    assert repaired['parent_result_sha256']==digest(parent_path)
    expected=json.loads((P/'OVERCOMPLETE_L1_READER_V1_BINDING.json').read_text())
    assert parent['binding']==expected
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda();readers=torch.cat((l,r));norm=readers.norm(dim=1)
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train=torch.cat((order[:3072],order[:3072]+4608)).cuda();test=torch.cat((order[3072:],order[3072:]+4608)).cuda()
    x=(readers/norm[:,None])[train]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    writer=torch.linalg.cholesky(metric).T@down;native=(l,r,writer)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    results=[];functions=[]
    for seed in (0,937):
        parent_fit=next(f for f in parent['fits'] if f['seed']==seed)
        baseline=next(a for a in repaired['arms'] if a['seed']==seed and a['name']=='learned')
        source=baseline['cache'];assert digest(source['path'])==source['sha256']
        old=RectangularSparseReaderProgram.from_artifact(torch.load(source['path'],weights_only=True,map_location='cpu'),down,bias)
        old_read=torch.sparse.mm(old.codes,old.analysis_basis);old_function=(*old_read.split(4608),writer)
        parent_capture=1-float((inner(old_function,old_function)-2*inner(native,old_function)+total)/total)
        folded_replay=abs(parent_capture-baseline['coefficient_capture']);assert folded_replay<=1e-8
        del old,old_read,old_function
        basis,optimization=fit(x,parent_fit,seed,binding)
        tic=time.perf_counter();ids,values,encoding=encode(basis,readers,128,.05,batch_size=64)
        encoding_seconds=time.perf_counter()-tic
        cache=Path(f'/dev/shm/bilin18_native_coupled_l1_polish_v1_s{seed}.pt');assert not cache.exists()
        torch.save(dict(analysis_basis=basis.cpu(),code_indices=ids.to(torch.int16).cpu(),code_values=values.cpu(),
            order=order,encoding=encoding,binding=binding,fit_cache=optimization['cache'],
            retained_down_key='transformer.h.17.mlp.Down.weight',retained_bias_key='transformer.h.17.mlp.Down_bias'),cache)
        program=RectangularSparseReaderProgram(basis,ids,values,down,bias)
        fitted=torch.sparse.mm(program.codes,basis);a,b=fitted.split(4608);proposal=(a,b,writer)
        error=((fitted-readers)/norm[:,None]).square().sum(1)
        capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
        sample=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(920))
        direct=((sample@a.T)*(sample@b.T))@down.T+bias
        execution=float((program(sample)-direct).norm()/direct.norm())
        instrument=(encoding['converged'] and encoding['maximum_code_kkt']<=1e-5 and max(
            encoding['support_ls_normal_residual'],execution,folded_replay)<=1e-8 and
            bool(torch.isfinite(error).all()) and np.isfinite(capture))
        row=dict(seed=seed,optimization=optimization,encoding=encoding,encoding_seconds=encoding_seconds,
            coefficient_capture=capture,coefficient_gain=capture-parent_capture,
            train_reader_capture=1-float(error[train].mean()),historical_test_reader_capture=1-float(error[test].mean()),
            historical_test_gain=1-float(error[test].mean())-baseline['historical_test_reader_capture'],
            instrument_passed=bool(instrument),executor_replay=execution,parent_folded_replay=folded_replay,
            baseline=baseline,cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size),price=program.price())
        write(P/f'{PREFIX}_SEED_{seed}.json',row);results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('optimization','encoding','baseline')}),flush=True)
        assert instrument;functions.append((a.clone(),b.clone(),writer))
        del program,fitted,a,b,proposal,ids,values,basis
    energies=[inner(f,f) for f in functions];cosine=float(inner(*functions)/(energies[0]*energies[1]).sqrt())
    predictions={'pred_a_instrument':all(r['instrument_passed'] for r in results),
        'pred_b_joint_convergence':all(r['optimization']['converged'] for r in results),
        'pred_c_stationarity_reduction':all(r['optimization']['stationarity_reduction']>=10 for r in results),
        'pred_d_tensor_gain':all(r['coefficient_gain']>=.01 for r in results),
        'pred_e_reader_gain':all(r['historical_test_gain']>=.01 for r in results),'pred_f_function_stability':cosine>=.9}
    result=dict(predictions=predictions,arms=results,function_cosine=cosine,preflight=check,
        parent_sha256=digest(parent_path),repaired_parent_sha256=digest(repaired_path),binding=binding,
        body_forwards=0,corpus_access=False,wall_seconds=time.perf_counter()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),scope='Same weight-only L1 objective, new coupled optimizer; historical split, no circuit identification or global recovery guarantee.')
    write(out,result);print(json.dumps(dict(predictions=predictions,function_cosine=cosine)),flush=True)


if __name__=='__main__':main()
