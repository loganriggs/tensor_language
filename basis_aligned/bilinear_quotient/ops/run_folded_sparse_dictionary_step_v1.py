#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact weight tensor gradient, no model body or text.
"""pred_a numerical; pred_b descent; pred_c tensor gain; pred_d gradient cost."""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import torch
from run_overcomplete_ols_reencode_v1 import CK,digest,RectangularSparseReaderProgram,inner
from folded_sparse_dictionary_v1 import decode,loss_gradient
PREFIX='FOLDED_SPARSE_DICTIONARY_STEP_V1'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    control=json.loads((P/'FOLDED_SPARSE_DICTIONARY_V1_CONTROL.json').read_text())['predictions']
    assert control['pred_a_dense_gradient'] and control['pred_b_finite_and_gauge']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            seeds=[0,937],accepted_steps_per_seed=1,features=2304,sparsity=128,
            objective='Full folded coefficient error, all weights; no heldout reader claim')));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(600)
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    parent=json.loads((P/'OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json').read_text())
    assert parent['predictions']['pred_a_instrument']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda();original=torch.cat((l,r));scale=original.norm(dim=1)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    writer=torch.linalg.cholesky(metric).T@down;native=(l,r,writer)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total'];results=[]
    for seed in (0,937):
        baseline=next(a for a in parent['arms'] if a['seed']==seed and a['name']=='learned')
        source=baseline['cache'];assert digest(source['path'])==source['sha256']
        saved=torch.load(source['path'],weights_only=True,map_location='cpu')
        raw=saved['analysis_basis'].double().cuda();ids=saved['code_indices'].long().cuda()
        values=saved['code_values'].double().cuda()/scale[:,None]
        assert raw.shape==(2304,1152) and values.shape==ids.shape==(9216,128)
        torch.cuda.synchronize();tic=time.perf_counter()
        initial,(gb,gc),details=loss_gradient(native,writer,raw,ids,values,scale,total)
        torch.cuda.synchronize();gradient_seconds=time.perf_counter()-tic
        parent_replay=abs(1-float(initial)-baseline['coefficient_capture'])
        direction_b=-gb*raw.norm()/gb.norm().clamp_min(1e-30)
        direction_c=-gc*values.norm()/gc.norm().clamp_min(1e-30)
        slope=float((gb*direction_b).sum()+(gc*direction_c).sum());assert slope<0
        h=1e-5
        plus=loss_gradient(native,writer,raw+h*direction_b,ids,values+h*direction_c,scale,total)[0]
        minus=loss_gradient(native,writer,raw-h*direction_b,ids,values-h*direction_c,scale,total)[0]
        fd=abs(float((plus-minus)/(2*h))-slope)/max(1.,abs(slope))
        assert fd<=1e-6 and parent_replay<=1e-8
        trials=[];accepted=False;new_raw=raw;new_values=values;final=initial
        step=.05
        for attempt in range(12):
            rb=raw+step*direction_b;cv=values+step*direction_c
            proposed,_,_=loss_gradient(native,writer,rb,ids,cv,scale,total)
            armijo=float(proposed)<=float(initial)+1e-4*step*slope
            trials.append(dict(step=step,loss=float(proposed),armijo=armijo))
            if armijo:new_raw,new_values,final,accepted=rb,cv,proposed,True;break
            step*=.5
        fitted,basis,_,_=decode(new_raw,ids,new_values,scale)
        a,b=fitted.chunk(2);proposal=(a,b,writer)
        independent=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
        cp_replay=abs(independent-(1-float(final)))
        physical_values=new_values*scale[:,None]
        program=RectangularSparseReaderProgram(basis,ids,physical_values,down,bias)
        sample=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(922))
        direct=((sample@a.T)*(sample@b.T))@down.T+bias
        execution=float((program(sample)-direct).norm()/direct.norm())
        before=decode(raw,ids,values,scale)[0]
        reader_before=float(((before-original)/scale[:,None]).square().sum(1).mean())
        reader_after=float(((fitted-original)/scale[:,None]).square().sum(1).mean())
        norms=float((basis.norm(dim=1)-1).abs().max())
        numeric=max(parent_replay,cp_replay,execution)<=1e-8 and fd<=1e-6 and norms<=1e-6 and bool(
            torch.isfinite(gb).all() and torch.isfinite(gc).all() and torch.isfinite(final))
        cache=Path(f'/dev/shm/bilin18_folded_sparse_dictionary_step_v1_s{seed}.pt');assert not cache.exists()
        torch.save(dict(analysis_basis=basis.cpu(),code_indices=ids.to(torch.int16).cpu(),code_values=physical_values.cpu(),
            source=source,binding=binding,retained_down_key='transformer.h.17.mlp.Down.weight',
            retained_bias_key='transformer.h.17.mlp.Down_bias',all_native_weight_rows_in_objective=True),cache)
        row=dict(seed=seed,instrument_passed=bool(numeric),accepted=accepted,initial_capture=1-float(initial),
            final_capture=1-float(final),capture_gain=float(initial-final),gradient_seconds=gradient_seconds,
            parent_replay=parent_replay,independent_cp_replay=cp_replay,executor_replay=execution,
            fd_error=fd,analytic_directional_derivative=slope,gradient_norms=[float(gb.norm()),float(gc.norm())],
            initial_diagnostics=details,reader_squared_error_before=reader_before,reader_squared_error_after=reader_after,
            trials=trials,maximum_unit_norm_error=norms,price=program.price(),source=source,
            cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size))
        write(P/f'{PREFIX}_SEED_{seed}.json',row);results.append(row);print(json.dumps(row),flush=True);assert numeric
    predictions={'pred_a_instrument':all(r['instrument_passed'] for r in results),
        'pred_b_descent':all(r['accepted'] and r['capture_gain']>=-1e-10 for r in results),
        'pred_c_gain':all(r['capture_gain']>=.005 for r in results),
        'pred_d_cost':all(r['gradient_seconds']<=30 for r in results)}
    result=dict(predictions=predictions,arms=results,binding=binding,wall_seconds=time.perf_counter()-started,
        body_forwards=0,corpus_access=False,scope='Single full-tensor step through fixed sparse supports/shared dictionary; all native weights in objective, no convergence or circuit claim.')
    write(out,result);print(json.dumps(predictions),flush=True)


if __name__=='__main__':main()
