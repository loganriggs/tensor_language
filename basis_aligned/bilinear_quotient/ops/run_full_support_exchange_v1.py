#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact full folded weights, no text.
"""pred_a numerical/execution; pred_b gain; pred_c control advantage; pred_d cost.

Two final projected-fit parents, refit-only versus one exact support exchange
per reader; unchanged features, actual fitted output, capacity and pairing.
Full predictions and price: FULL_SUPPORT_EXCHANGE_V1_PREREGISTRATION.md.
"""
import os, sys, json, time, signal
from pathlib import Path
RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]; P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import torch
from run_overcomplete_ols_reencode_v1 import CK, digest, RectangularSparseReaderProgram, inner
from folded_sparse_dictionary_v1 import decode
from folded_support_exchange_v1 import quadratic, best_exchange
PREFIX='FULL_SUPPORT_EXCHANGE_V1'


def write(path, value):
    with path.open('x') as f: json.dump(value,f,indent=2); f.write('\n')


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            seeds=[0,937],arms=['refit','exchange'],readers_per_arm=9216,support_size=128,
            dependency='PROJECTED_SPARSE_DICTIONARY_FIT_V1_RESULT.json',alarm_seconds=5400))); return
    out=P/f'{PREFIX}_RESULT.json'; assert not out.exists(); signal.alarm(5400)
    parent=json.loads((P/'PROJECTED_SPARSE_DICTIONARY_FIT_V1_RESULT.json').read_text())
    assert parent['predictions']['pred_a_instrument'], 'Parent instrument failed; no silent fallback'
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda()
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double().cuda()
    wh=torch.linalg.cholesky(metric).T; native=(l,r,wh@down)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    results=[]; functions={}; started=time.perf_counter()
    for par in parent['arms']:
        seed=par['seed']; source=par['cache']; assert digest(source['path'])==source['sha256']
        saved=torch.load(source['path'],weights_only=True,map_location='cpu')
        basis=saved['analysis_basis'].double().cuda(); rawids=saved['code_indices'].long().cuda()
        rawvalues=saved['code_values'].double().cuda(); physical_down=saved['down'].double().cuda()
        initial,basis,_,_=decode(basis,rawids,rawvalues,torch.ones(len(rawids),device='cuda'))
        w=wh@physical_down; original=(*initial.chunk(2),w)
        initial_capture=1-float((inner(original,original)-2*inner(native,original)+total)/total)
        initial_replay=abs(initial_capture-par['final_capture']); assert initial_replay<=1e-8
        bg=basis@basis.T
        for mode in ('refit','exchange'):
            readers=initial.clone(); a,b=readers.chunk(2); ids=rawids.clone(); values=rawvalues.clone()
            cache=Path(f'/dev/shm/bilin18_full_support_exchange_v1_{mode}_s{seed}.pt'); assert not cache.exists()
            rows=[]; clock=time.perf_counter(); count=0
            def checkpoint():
                temp=cache.with_suffix('.tmp')
                torch.save(dict(analysis_basis=basis.cpu(),code_indices=ids.to(torch.int16).cpu(),
                    code_values=values.cpu(),down=physical_down.cpu(),retained_bias_key=saved['retained_bias_key'],
                    source=source,binding=binding,mode=mode,readers_completed=count,
                    all_native_weight_rows_in_objective=True),temp)
                os.replace(temp,cache)
            for side in ('Left','Right'):
                aa,bb=(a,b) if side=='Left' else (b,a)
                nn=native if side=='Left' else (r,l,native[2])
                for j in range(4608):
                    index=j if side=='Left' else j+4608; support=ids[index].tolist()
                    gram,rhs=quadratic(nn,aa,bb,w,j,basis,bg)
                    ss=gram[support][:,support]; old=values[index]
                    old_energy=old@ss@old-2*old@rhs[support]
                    if mode=='exchange':
                        new_ids,new_values,report=best_exchange(gram,rhs,support)
                        changed=report['changed']; swap_gain=report['gain']/total
                    else:
                        new_ids=support; chol=torch.linalg.cholesky((ss+ss.T)/2)
                        new_values=torch.cholesky_solve(rhs[support,None],chol).flatten()
                        changed=False; swap_gain=0.
                    new_energy=new_values@gram[new_ids][:,new_ids]@new_values-2*new_values@rhs[new_ids]
                    gain=float((old_energy-new_energy)/total)
                    assert torch.isfinite(new_values).all() and len(set(new_ids))==128
                    aa[j]=new_values@basis[new_ids]; ids[index]=torch.tensor(new_ids,device='cuda')
                    values[index]=new_values; count+=1
                    rows.append(dict(side=side,product=j,total_gain=gain,swap_gain=swap_gain,changed=changed))
                    if count%512==0:
                        checkpoint()
                        print(json.dumps(dict(seed=seed,mode=mode,readers=count,seconds=time.perf_counter()-clock,
                            conditional_gain=sum(q['total_gain'] for q in rows))),flush=True)
            proposal=(a,b,w)
            capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
            gain=capture-initial_capture; replay=abs(gain-sum(q['total_gain'] for q in rows))
            program=RectangularSparseReaderProgram(basis,ids,values,physical_down,bias)
            x=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(621))
            direct=((x@a.T)*(x@b.T))@physical_down.T+bias
            execution=float((program(x)-direct).norm()/direct.norm())
            seconds=time.perf_counter()-clock; checkpoint()
            numeric=max(initial_replay,replay,execution)<=1e-8 and min(q['total_gain'] for q in rows)>=-1e-10
            row=dict(seed=seed,mode=mode,instrument_passed=numeric,initial_capture=initial_capture,
                final_capture=capture,capture_gain=gain,conditional_cp_replay=replay,executor_replay=execution,
                compute_seconds=seconds,changed_readers=sum(q['changed'] for q in rows),rows=rows,
                parent_converged=par['optimization']['converged'],source=source,price=program.price(),
                cache=dict(path=str(cache),sha256=digest(cache)))
            write(P/f'{PREFIX}_{mode}_SEED_{seed}.json',row); results.append(row)
            functions[(seed,mode)]=(a.clone(),b.clone(),w.clone())
            print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
    comparisons=[]
    for seed in (0,937):
        refit=next(q for q in results if q['seed']==seed and q['mode']=='refit')
        exchange=next(q for q in results if q['seed']==seed and q['mode']=='exchange')
        comparisons.append(dict(seed=seed,exchange_gain=exchange['capture_gain'],
            advantage=exchange['final_capture']-refit['final_capture']))
    stability={}
    for mode in ('refit','exchange'):
        f,g=functions[(0,mode)],functions[(937,mode)]
        stability[mode]=float(inner(f,g)/(inner(f,f)*inner(g,g)).sqrt())
    predictions={'pred_a_instrument':all(q['instrument_passed'] for q in results),
        'pred_b_gain':all(q['exchange_gain']>=.001 for q in comparisons),
        'pred_c_advantage':all(q['advantage']>=.0005 for q in comparisons),
        'pred_d_cost':all(q['compute_seconds']<=900 for q in results)}
    result=dict(predictions=predictions,comparisons=comparisons,function_cosines=stability,
        arms=[{k:v for k,v in q.items() if k!='rows'} for q in results],binding=binding,
        wall_seconds=time.perf_counter()-started,body_forwards=0,corpus_access=False,
        scope='One full conditional support sweep, fixed features/output/pairing; no graph convergence or circuit claim.')
    write(out,result); print(json.dumps(dict(predictions=predictions,comparisons=comparisons)),flush=True)


if __name__=='__main__': main()
