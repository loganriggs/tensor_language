#!/usr/bin/env python3
# BQGATE: 0forwards0seq; sustained full weight-tensor fitting, no text.
"""pred_a numerical; pred_b convergence; pred_c joint gain; pred_d capture;
pred_e complete-function stability. Output solved exactly every evaluation.
"""
import os,sys,json,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import numpy as np
import torch
from run_overcomplete_ols_reencode_v1 import CK,digest,RectangularSparseReaderProgram,inner
from folded_sparse_dictionary_v1 import decode
from projected_sparse_dictionary_fit_v1 import ProjectedObjective,fit
PREFIX='PROJECTED_SPARSE_DICTIONARY_FIT_V1'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    assert all(json.loads((P/'PROJECTED_SPARSE_DICTIONARY_V1_CONTROL.json').read_text())['predictions'].values())
    assert all(json.loads((P/'FOLDED_SPARSE_DICTIONARY_STEP_V1_RESULT.json').read_text())['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            seeds=[0,937],fit_seconds_per_seed=3600,iterations=2500,features=2304,sparsity=128,
            objective='Full coefficient error with exact eliminated output weights',historical_weights_are_heldout=False)));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(9000)
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    parent=json.loads((P/'OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json').read_text())
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda();scale=torch.cat((l,r)).norm(dim=1)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    whitener=torch.linalg.cholesky(metric).T;native=(l,r,whitener@down)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total'];results=[];functions=[]
    for seed in (0,937):
        baseline=next(a for a in parent['arms'] if a['seed']==seed and a['name']=='learned')
        source=baseline['cache'];assert digest(source['path'])==source['sha256']
        saved=torch.load(source['path'],weights_only=True,map_location='cpu')
        raw=saved['analysis_basis'].double().cuda();ids=saved['code_indices'].long().cuda()
        values=saved['code_values'].double().cuda()/scale[:,None]
        assert raw.shape==(2304,1152) and ids.shape==values.shape==(9216,128)
        objective=ProjectedObjective((l,r,down),whitener,raw,ids,values,scale,total)
        initial,gradient=objective.evaluate(objective.initial);initial_writer=objective.last['writer'].clone()
        initial_readers=decode(raw,ids,values,scale)[0];initial_cp=(*initial_readers.chunk(2),whitener@initial_writer)
        initial_cp_loss=float((inner(initial_cp,initial_cp)-2*inner(native,initial_cp)+total)/total)
        initial_replay=abs(initial-initial_cp_loss)
        direction=-gradient/max(np.linalg.norm(gradient),1e-30);slope=float(gradient@direction);h=1e-5
        plus=objective.evaluate(objective.initial+h*direction)[0]
        minus=objective.evaluate(objective.initial-h*direction)[0]
        fd=abs((plus-minus)/(2*h)-slope)/max(1.,abs(slope));assert fd<=1e-6 and initial_replay<=1e-8
        preflight=dict(seed=seed,packed_fd_error=fd,initial_cp_replay=initial_replay,
            original_writer_capture=baseline['coefficient_capture'],initial_optimal_writer_capture=1-initial,
            initial_writer_norm_ratio=float(initial_writer.norm()/down.norm()))
        write(P/f'{PREFIX}_SEED_{seed}_PREFLIGHT.json',preflight);print(json.dumps(preflight),flush=True)
        cache=Path(f'/dev/shm/bilin18_projected_sparse_dictionary_fit_v1_s{seed}.pt');assert not cache.exists()
        def checkpoint(point,current,history):
            rr,cc=objective.unpack(point);_,bb,_,_=decode(rr,ids,cc,scale)
            temp=cache.with_suffix('.tmp')
            torch.save(dict(analysis_basis=bb.cpu(),raw_basis=rr.cpu(),normalized_values=cc.cpu(),
                code_indices=ids.to(torch.int16).cpu(),code_values=(cc*scale[:,None]).cpu(),down=current['writer'].cpu(),
                retained_bias_key='transformer.h.17.mlp.Down_bias',source=source,binding=binding,
                history=history,loss=current['value'],details=current['details'],
                all_native_weight_rows_in_objective=True,lbfgs_history_restorable=False),temp)
            os.replace(temp,cache)
        point,writer,optimization=fit(objective,seconds=3600,max_iterations=2500,checkpoint=checkpoint)
        write(P/f'{PREFIX}_SEED_{seed}_FIT.json',optimization)
        rr,cc=objective.unpack(point);fitted,basis,_,_=decode(rr,ids,cc,scale)
        a,b=fitted.chunk(2);proposal=(a,b,whitener@writer)
        capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
        replay=abs(capture-(1-optimization['final_loss']))
        program=RectangularSparseReaderProgram(basis,ids,cc*scale[:,None],writer,bias)
        sample=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(924))
        direct=((sample@a.T)*(sample@b.T))@writer.T+bias
        execution=float((program(sample)-direct).norm()/direct.norm())
        normerror=float((basis.norm(dim=1)-1).abs().max())
        ranks=[optimization['initial_details']['writer_solve']['rank']]+[h['details']['writer_solve']['rank'] for h in optimization['history']]
        smooth=all(rank==4608 for rank in ranks)
        numeric=max(replay,execution,initial_replay)<=1e-8 and normerror<=1e-6 and fd<=1e-6 and smooth and optimization['maximum_accepted_loss_increase']<=1e-10
        row=dict(seed=seed,instrument_passed=numeric,preflight=preflight,optimization=optimization,
            final_capture=capture,joint_capture_gain=capture-(1-initial),initial_optimal_writer_capture=1-initial,
            original_writer_capture=baseline['coefficient_capture'],independent_cp_replay=replay,
            executor_replay=execution,maximum_unit_norm_error=normerror,accepted_product_ranks=sorted(set(ranks)),
            writer_norm_ratio=float(writer.norm()/down.norm()),price=program.price(),source=source,
            cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size))
        write(P/f'{PREFIX}_SEED_{seed}.json',row);results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='optimization'}),flush=True)
        functions.append((a.clone(),b.clone(),(whitener@writer).clone()))
        del objective,program,fitted,a,b,proposal,saved,raw,values,initial_readers,initial_cp,initial_writer
    energies=[inner(f,f) for f in functions];cosine=float(inner(*functions)/(energies[0]*energies[1]).sqrt())
    predictions={'pred_a_instrument':all(r['instrument_passed'] for r in results),
        'pred_b_convergence':all(r['optimization']['converged'] for r in results),
        'pred_c_joint_gain':all(r['joint_capture_gain']>=.05 for r in results),
        'pred_d_capture':all(r['final_capture']>=.70 for r in results),'pred_e_stability':cosine>=.9}
    result=dict(predictions=predictions,arms=results,function_cosine=cosine,binding=binding,
        body_forwards=0,corpus_access=False,wall_seconds=time.perf_counter()-started,
        scope='Full folded weight objective, fixed support graph and product pairing, learned shared features/codes/output. No global optimum, heldout weight discovery or behavioral circuit claim.')
    write(out,result);print(json.dumps(dict(predictions=predictions,function_cosine=cosine)),flush=True)


if __name__=='__main__':main()
