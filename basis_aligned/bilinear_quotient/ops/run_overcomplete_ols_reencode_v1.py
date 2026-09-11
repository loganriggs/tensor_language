#!/usr/bin/env python3
# BQGATE: 0forwards0seq; four frozen weight dictionaries, no text or new basis fit.
"""pred_a numerical; pred_b nonworsening; pred_c learned encoding gain;
pred_d heldout reader gain; pred_e full tensor gain; pred_f function stability.
"""
import os,sys,json,time,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import torch
from native_reader_msp_generalization_v1 import CK
from batched_lasso_ols_completion_v1 import encode
from batched_lasso_ols_completion_v1_control import compare
from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram
from structured_branch_amplitudes_v1 import inner
PREFIX='OVERCOMPLETE_OLS_REENCODE_V1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    assert json.loads((P/'BATCHED_LASSO_OLS_COMPLETION_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
            arms=4,features=2304,readers=9216,sparsity=128,penalty=.05,batch_size=64,
            new_basis_fit=False,parent_result_required_at_execution='OVERCOMPLETE_L1_READER_V1_RESULT.json')));return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists();signal.alarm(1800)
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    preflight=compare('cuda');assert preflight['passed']
    with (P/f'{PREFIX}_PREFLIGHT.json').open('x') as f:json.dump(preflight,f,indent=2);f.write('\n')
    parent_path=P/'OVERCOMPLETE_L1_READER_V1_RESULT.json';parent_sha=digest(parent_path)
    parent=json.loads(parent_path.read_text());rows=parent['arms'];assert len(rows)==4
    assert {(r['seed'],r['name']) for r in rows}=={(s,n) for s in (0,937) for n in ('untrained','learned')}
    expected_parent_binding=json.loads((P/'OVERCOMPLETE_L1_READER_V1_BINDING.json').read_text())
    assert parent['binding']==expected_parent_binding
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda();readers=torch.cat((l,r));norms=readers.norm(dim=1)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    writer=torch.linalg.cholesky(metric).T@down;native=(l,r,writer)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    results=[];learned_functions=[]
    for row in rows:
        seed,name=row['seed'],row['name'];source=row['cache'];assert digest(source['path'])==source['sha256']
        artifact=torch.load(source['path'],weights_only=True,map_location='cpu')
        basis=artifact['analysis_basis'].cuda();assert basis.shape==(2304,1152)
        old_program=RectangularSparseReaderProgram.from_artifact(artifact,down,bias)
        old_read=torch.sparse.mm(old_program.codes,old_program.analysis_basis)
        old=(old_read[:4608],old_read[4608:],writer)
        old_capture=1-float((inner(old,old)-2*inner(native,old)+total)/total)
        parent_replay=abs(old_capture-row['scores']['coefficient_capture'])
        del old_program,old_read,old
        tic=time.perf_counter();ids,values,encoding=encode(basis,readers,128,.05,batch_size=64)
        encoding_seconds=time.perf_counter()-tic
        cache=Path(f'/dev/shm/bilin18_overcomplete_ols_reencode_v1_{name}_s{seed}.pt');assert not cache.exists()
        torch.save(dict(analysis_basis=basis.cpu(),code_indices=ids.to(torch.int16).cpu(),code_values=values.cpu(),
            order=artifact['order'],source=source,encoding=encoding,binding=binding,
            retained_down_key='transformer.h.17.mlp.Down.weight',retained_bias_key='transformer.h.17.mlp.Down_bias'),cache)
        program=RectangularSparseReaderProgram(basis,ids,values,down,bias)
        fitted=torch.sparse.mm(program.codes,basis);a,b=fitted.split(4608);proposal=(a,b,writer)
        error=((fitted-readers)/norms[:,None]).square().sum(1)
        order=artifact['order'].cuda();train=torch.cat((order[:3072],order[:3072]+4608));test=torch.cat((order[3072:],order[3072:]+4608))
        capture=1-float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
        x=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(819))
        direct=((x@a.T)*(x@b.T))@down.T+bias
        execution=float((program(x)-direct).norm()/direct.norm());maximum_norm=float(basis.norm(dim=1).max())
        instrument=encoding['converged'] and encoding['maximum_code_kkt']<=1e-5 and maximum_norm<=1+1e-6 and max(
            encoding['support_ls_normal_residual'],execution,parent_replay)<=1e-8 and bool(torch.isfinite(error).all()) and bool(torch.isfinite(torch.tensor(capture)))
        result=dict(seed=seed,name=name,instrument_passed=instrument,encoding=encoding,encoding_seconds=encoding_seconds,
            coefficient_capture_before=old_capture,coefficient_capture=capture,coefficient_gain=capture-old_capture,
            train_reader_capture=1-float(error[train].mean()),historical_test_reader_capture=1-float(error[test].mean()),
            parent_capture_replay=parent_replay,executor_replay=execution,maximum_atom_norm=maximum_norm,
            source=source,cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size),price=program.price())
        with (P/f'{PREFIX}_{name}_SEED_{seed}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in result.items() if k!='encoding'}),flush=True)
        results.append(result);assert instrument
        if name=='learned':learned_functions.append((a.clone(),b.clone(),writer))
        del artifact,program,fitted,a,b,proposal,ids,values,basis
    base=[r for r in results if r['name']=='untrained'];learned=[r for r in results if r['name']=='learned']
    best_reader=max(r['historical_test_reader_capture'] for r in base);best_tensor=max(r['coefficient_capture'] for r in base)
    energies=[inner(f,f) for f in learned_functions]
    cosine=float(inner(*learned_functions)/(energies[0]*energies[1]).sqrt())
    predictions={'pred_a_instrument':all(r['instrument_passed'] for r in results),
        'pred_b_nonworsening':all(r['coefficient_gain']>=-1e-8 for r in results),
        'pred_c_encoding_gain':all(r['coefficient_gain']>=.02 for r in learned),
        'pred_d_historical_reader_gain':all(r['historical_test_reader_capture']>=best_reader+.02 for r in learned),
        'pred_e_tensor_gain':all(r['coefficient_capture']>=best_tensor+.01 for r in learned),
        'pred_f_function_stability':cosine>=.9}
    result=dict(predictions=predictions,arms=results,learned_function_cosine=cosine,
        parent_convergence={str(f['seed']):f['converged'] for f in parent['fits']},parent_result_sha256=parent_sha,
        binding_sha256=digest(P/f'{PREFIX}_BINDING.json'),body_forwards=0,corpus_access=False,
        wall_seconds=time.perf_counter()-started,preflight=preflight,
        scope='Four frozen weight dictionaries, same inference repair and price, historical split. Does not repair parent joint convergence or establish circuits.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=predictions,learned_function_cosine=cosine)),flush=True)


if __name__=='__main__':main()
