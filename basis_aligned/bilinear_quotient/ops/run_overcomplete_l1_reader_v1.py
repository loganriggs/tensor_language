#!/usr/bin/env python3
"""pred_a execution; pred_b convergence; pred_c heldout reader gain; pred_d tensor.
BQGATE:0forwards0seq. Full-rank overcomplete L1 reader discovery from weights.
"""
import json
import os
from pathlib import Path
import signal
import time
import torch
from run_structured_bilinear_native_v2 import P,CK,digest
from quadratic_token_dictionary_v1 import cycle,diagnostics
from lasso_reader_encoding_v1 import encode
from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram
from structured_branch_amplitudes_v1 import inner

PREFIX='OVERCOMPLETE_L1_READER_V1'

def atomic_save(value,path):
    temp=path.with_suffix('.tmp')
    torch.save(value,temp);os.replace(temp,path)

@torch.no_grad()
def preflight():
    b=torch.eye(8,device='cuda')
    gen=torch.Generator(device='cuda').manual_seed(816)
    readers=torch.randn(20,8,device='cuda',generator=gen)
    ids,values,report=encode(b,readers,3)
    down=torch.randn(7,10,device='cuda',generator=gen)
    program=RectangularSparseReaderProgram(b,ids,values,down)
    dense=torch.zeros_like(readers).scatter_(1,ids,values)
    x=torch.randn(5,8,device='cuda',generator=gen)
    direct=((x@dense[:10].T)*(x@dense[10:].T))@down.T
    error=float((program(x)-direct).norm()/direct.norm())
    assert report['converged'] and max(error,report['support_ls_normal_residual'])<=1e-8
    return dict(encoder=report,execution_error=error,body_forwards=0)

@torch.no_grad()
def fit(x64,initial,seed,binding):
    path=Path(f'/dev/shm/bilin18_overcomplete_l1_reader_v1_s{seed}_fit.pt')
    assert not path.exists()
    x=x64.float();b=initial.float();z=torch.zeros(len(x),len(b),device='cuda',dtype=x.dtype)
    history=[];checks=[];precision='float32';converged=False;stop='step_limit'
    started=time.perf_counter();saved_at=-60.
    def checkpoint():
        atomic_save(dict(dictionary=b.cpu(),codes=z.cpu(),initial_dictionary=initial.cpu(),
            history=history,checks=checks,precision=precision,seed=seed,penalty=.05,binding=binding),path)
    checkpoint()
    for iteration in range(1,2001):
        z,b,conditional=cycle(x,z,b,.05,300,1e-6)
        elapsed=time.perf_counter()-started
        verified=None
        numerical_stop=any(v['terminal']=='nondecreasing_step' for v in conditional.values())
        if iteration%5==0:
            verified=diagnostics(x64,z.double(),b.double(),.05)
            checks.append(dict(iteration=iteration,**verified))
        if precision=='float32' and (elapsed>=900 or numerical_stop or
                (verified is not None and verified['relative_stationarity']<=1e-4)):
            x=x64;z=z.double();b=b.double();precision='float64'
        change=None
        if verified is not None and len(checks)>=2:
            change=abs(checks[-1]['objective']-checks[-2]['objective'])/max(abs(checks[-1]['objective']),1e-30)
            converged=precision=='float64' and all(v['relative_stationarity']<=1e-5 for v in checks[-2:]) and change<=1e-8
        row=dict(iteration=iteration,seconds=elapsed,precision=precision,conditional=conditional,
            fp64_check=verified,check_relative_objective_change=change)
        history.append(row)
        if iteration==1 or iteration%5==0 or converged:print(json.dumps(dict(seed=seed,phase='fit',**row)),flush=True)
        if elapsed-saved_at>=60 or converged:
            checkpoint();saved_at=elapsed
        if converged:stop='converged';break
        if elapsed>=1800:stop='time_limit';break
    final=diagnostics(x64,z.double(),b.double(),.05)
    checkpoint()
    report=dict(seed=seed,converged=converged,stop=stop,seconds=time.perf_counter()-started,
        iterations=iteration,precision=precision,final=final,checks=checks,history=history,
        cache=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size))
    with (P/f'{PREFIX}_SEED_{seed}_FIT.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    return b.double(),report

@torch.no_grad()
def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            features=2304,input_dimension=1152,seeds=[0,937],fit_seconds_per_seed=1800,
            coefficients=9142272,sparse_indices=1179648,penalty=.05)))
        return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists()
    signal.alarm(7200);torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter();check=preflight()
    with (P/f'{PREFIX}_PREFLIGHT.json').open('x') as f:json.dump(check,f,indent=2);f.write('\n')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda()
    readers=torch.cat((l,r));norms=readers.norm(dim=1)
    unit=readers/norms[:,None]
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train=torch.cat((order[:3072],order[:3072]+4608)).cuda()
    test=torch.cat((order[3072:],order[3072:]+4608)).cuda()
    metric_receipt=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    metric=torch.load(metric_receipt['cache']['path'],weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    whitener=torch.linalg.cholesky(metric).T
    native=(l,r,whitener@down);total=metric_receipt['native_total']
    results=[];fits=[]
    for seed in (0,937):
        selected=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed))[:2304].cuda()
        initial=unit[train[selected]].clone()
        learned,optimization=fit(unit[train],initial,seed,binding);fits.append(optimization)
        for name,basis in [('untrained',initial),('learned',learned)]:
            ids,values,encoding=encode(basis,readers)
            cache=Path(f'/dev/shm/bilin18_overcomplete_l1_reader_v1_{name}_s{seed}.pt')
            assert not cache.exists()
            torch.save(dict(analysis_basis=basis.cpu(),code_indices=ids.to(torch.int16).cpu(),code_values=values.cpu(),
                seed=seed,name=name,order=order,encoding=encoding,binding=binding,
                retained_down_key='transformer.h.17.mlp.Down.weight',retained_bias_key='transformer.h.17.mlp.Down_bias'),cache)
            saved=torch.load(cache,weights_only=True,map_location='cpu')
            program=RectangularSparseReaderProgram.from_artifact(saved,down,bias)
            reconstructed=torch.sparse.mm(program.codes,program.analysis_basis)
            errors=((reconstructed-readers)/norms[:,None]).square().sum(1)
            proposal=(reconstructed[:4608],reconstructed[4608:],native[2])
            capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
            gen=torch.Generator(device='cuda').manual_seed(816)
            x=torch.randn(8,1152,device='cuda',generator=gen)
            direct=((x@proposal[0].T)*(x@proposal[1].T))@down.T+bias
            execution=float((program(x)-direct).norm()/direct.norm())
            normmax=float(basis.norm(dim=1).max())
            scores=dict(train_capture=1-float(errors[train].mean()),heldout_capture=1-float(errors[test].mean()),
                coefficient_capture=capture,executor_replay=execution,atom_norm_max=normmax)
            instrument=all(torch.isfinite(torch.tensor(v)).item() for v in scores.values()) and normmax<=1+1e-6 and max(execution,encoding['support_ls_normal_residual'])<=1e-8
            report=dict(seed=seed,name=name,scores=scores,encoding=encoding,instrument_passed=instrument,
                cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size))
            with (P/f'{PREFIX}_{name}_SEED_{seed}.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
            print(json.dumps(report),flush=True);results.append(report)
            assert instrument
            del saved,program,reconstructed,proposal,ids,values
    baselines=[r for r in results if r['name']=='untrained'];learned_rows=[r for r in results if r['name']=='learned']
    reader_best=max(r['scores']['heldout_capture'] for r in baselines)
    tensor_best=max(r['scores']['coefficient_capture'] for r in baselines)
    predictions={'pred_a_instrument':all(r['instrument_passed'] for r in results),
        'pred_b_convergence':all(f['converged'] for f in fits) and all(r['encoding']['converged'] and r['encoding']['maximum_code_kkt']<=1e-5 for r in results),
        'pred_c_heldout_gain':all(r['scores']['heldout_capture']>=reader_best+.02 for r in learned_rows),
        'pred_d_tensor_gain':all(r['scores']['coefficient_capture']>=tensor_best+.01 for r in learned_rows)}
    result=dict(predictions=predictions,eligible_for_validation=all(predictions.values()),
        fits=fits,arms=results,binding=binding,preflight=check,wall_seconds=time.perf_counter()-started,
        coefficients=9142272,sparse_indices=1179648,body_forwards=0,corpus_access=False,
        heldout_weights_used_for_basis_fit=False,scope='Overcomplete L1 reader discovery with retained native product pairings and Down; no circuit identification.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(predictions),flush=True)

if __name__=='__main__':main()
