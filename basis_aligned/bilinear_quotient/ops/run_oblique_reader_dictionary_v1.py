#!/usr/bin/env python3
"""pred_a exact execution; pred_b converged; pred_c heldoutgain; pred_d tensor.
BQGATE:0forwards0seq. Native weight vectors only, four900second fits.
"""
import json
import os
from pathlib import Path
import signal
import time
from run_structured_bilinear_native_v2 import P,CK,digest
import torch
from scipy.optimize import linear_sum_assignment
from orthogonal_reader_msp_v1 import fit
from tyler_reader_shape_v1 import roots,normalize_columns
from oblique_sparse_reader_v1 import canonical_maps,encode
from sparse_reader_program_v1 import SparseReaderProgram
from structured_branch_amplitudes_v1 import inner

PREFIX='OBLIQUE_READER_DICTIONARY_V1'


@torch.no_grad()
def gpu_preflight():
    identity=torch.eye(8,device='cuda')
    readers=torch.arange(96,device='cuda',dtype=torch.float64).reshape(12,8)/97
    indices,values,encoding=encode(identity,identity,readers,3,chunk=4)
    down=torch.arange(30,device='cuda',dtype=torch.float64).reshape(5,6)/31
    program=SparseReaderProgram(identity,indices,values,down)
    codes=torch.zeros_like(readers).scatter_(1,indices,values)
    x=torch.arange(24,device='cuda',dtype=torch.float64).reshape(3,8)/25
    direct=((x@codes[:6].T)*(x@codes[6:].T))@down.T
    error=float((program(x)-direct).norm()/direct.norm())
    assert max(error,encoding['normal_equation_relative_residual'])<=1e-8
    return dict(sparse_execution_relative_error=error,encoding=encoding,
                body_forwards=0,synthetic_inputs=3)


@torch.no_grad()
def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            methods=['ordinary_covariance','tyler'],seeds=[0,937],seconds_per_fit=900,
            coefficients=7815168,sparse_indices=1179648,heldout_basis_fitting=False)))
        return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists()
    baseline_path=P/'FULL_READER_DICTIONARY_MSP_V1_RESULT.json'
    baseline=json.loads(baseline_path.read_text())
    assert baseline['predictions']['pred_a_basis_replay']
    assert all(binding.get(path)==sha for path,sha in baseline['binding'].items())
    reference=list(baseline['baselines'].values())+[r['scores'] for r in baseline['starts']]
    best_reader=max(r['test_top128_energy'] for r in reference)
    best_tensor=max(r['coefficient_capture'] for r in reference)
    signal.alarm(4800)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    preflight=gpu_preflight()
    with (P/f'{PREFIX}_PREFLIGHT.json').open('x') as f:
        json.dump(preflight,f,indent=2);f.write('\n')
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda()
    readers=torch.cat((l,r));norms=readers.norm(dim=1)
    assert float(norms.min())>1e-8
    y=(readers/norms[:,None]).T
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train_ids=torch.cat((order[:3072],order[:3072]+4608)).cuda()
    test_ids=torch.cat((order[3072:],order[3072:]+4608)).cuda()
    shape_receipt=json.loads((P/'NATIVE_READER_SHAPE_V1_AUDIT.json').read_text())
    saved_shapes=torch.load(shape_receipt['cache']['path'],map_location='cpu',weights_only=True)
    assert torch.equal(order[:3072],saved_shapes['training_products'])
    metric=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    saved_metric=torch.load(metric['cache']['path'],map_location='cpu',weights_only=True)
    whitener=torch.linalg.cholesky(saved_metric['unembedding_gram'].cuda()).T
    native=l,r,whitener@d
    total=metric['native_total']
    results=[];features_by_method={}
    for method,key in [('ordinary_covariance','ordinary_shape'),('tyler','tyler_shape')]:
        shape=saved_shapes[key].cuda()
        _,inverse=roots(shape)
        training=normalize_columns(inverse@y[:,train_ids])
        features_by_method[method]=[]
        for seed in (0,937):
            rotation,optimization=fit(training,seed,max_steps=20000,max_seconds=900,
                callback=lambda row:print(json.dumps(dict(method=method,seed=seed,phase='fit',**row)),flush=True))
            rotation_cache=Path(f'/dev/shm/bilin18_oblique_reader_v1_{method}_s{seed}_rotation.pt')
            assert not rotation_cache.exists()
            torch.save(dict(rotation=rotation.cpu(),method=method,seed=seed,
                optimization=optimization,binding=binding),rotation_cache)
            rotation_receipt=dict(path=str(rotation_cache),sha256=digest(rotation_cache),bytes=rotation_cache.stat().st_size)
            with (P/f'{PREFIX}_{method}_SEED_{seed}_FIT.json').open('x') as f:
                json.dump(dict(method=method,seed=seed,optimization=optimization,cache=rotation_receipt),f,indent=2);f.write('\n')
            maps=canonical_maps(rotation,shape)
            indices,values,encoding=encode(maps['dictionary'],maps['weight_coder'],readers,128,chunk=64)
            program=SparseReaderProgram(maps['input_features'],indices,values,d,bias)
            reconstructed=torch.sparse.mm(program.codes,program.analysis_basis)
            reader_errors=((reconstructed-readers)/norms[:,None]).square().sum(1)
            proposal=reconstructed[:4608],reconstructed[4608:],native[2]
            residual=float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
            full_coordinates=readers@maps['weight_coder'].T
            full_replay=float((full_coordinates@maps['input_features']-readers).norm()/readers.norm())
            generator=torch.Generator(device='cuda').manual_seed(739)
            x=torch.randn(16,1152,device='cuda',generator=generator)
            direct=((x@proposal[0].T)*(x@proposal[1].T))@d.T+bias
            sparse_replay=float((program(x)-direct).norm()/direct.norm())
            scores=dict(train_normalized_reader_capture=1-float(reader_errors[train_ids].mean()),
                test_normalized_reader_capture=1-float(reader_errors[test_ids].mean()),
                coefficient_capture=1-residual,coefficient_residual=residual,
                full_coordinate_replay=full_replay,sparse_executor_replay=sparse_replay)
            finite=all(torch.isfinite(torch.tensor(v)).item() for v in scores.values())
            instrument=finite and max(full_replay,sparse_replay,encoding['normal_equation_relative_residual'])<=1e-8
            cache=Path(f'/dev/shm/bilin18_oblique_reader_v1_{method}_s{seed}.pt')
            assert not cache.exists()
            torch.save(dict(analysis_basis=maps['input_features'].cpu(),code_indices=indices.to(torch.int16).cpu(),
                code_values=values.cpu(),rotation=rotation.cpu(),optimization_only_keys=['rotation'],
                retained_down_key='transformer.h.17.mlp.Down.weight',retained_bias_key='transformer.h.17.mlp.Down_bias',
                order=order,seed=seed,method=method,history=optimization['history'],binding=binding,sparsity=128),cache)
            report=dict(method=method,seed=seed,optimization=optimization,encoding=encoding,scores=scores,
                instrument_passed=instrument,rotation_cache=rotation_receipt,
                cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size))
            with (P/f'{PREFIX}_{method}_SEED_{seed}.json').open('x') as f:
                json.dump(report,f,indent=2);f.write('\n')
            print(json.dumps(dict(method=method,seed=seed,scores=scores,instrument=instrument,
                converged=optimization['converged'])),flush=True)
            assert instrument
            results.append(report)
            features_by_method[method].append(maps['input_features'].cpu())
            del program,indices,values,reconstructed,proposal,maps,full_coordinates
    families={}
    for method,features in features_by_method.items():
        selected=[r for r in results if r['method']==method]
        similarity=(features[0]@features[1].T).abs().numpy()
        i,j=linear_sum_assignment(-similarity)
        reader_gain=all(r['scores']['test_normalized_reader_capture']>=best_reader+.02 for r in selected)
        tensor_gain=all(r['scores']['coefficient_capture']>=best_tensor+.01 for r in selected)
        converged=all(r['optimization']['converged'] for r in selected)
        families[method]=dict(reader_gain_both=reader_gain,tensor_gain_both=tensor_gain,
            combined_quality=reader_gain and tensor_gain,converged_both=converged,
            eligible_for_validation=reader_gain and tensor_gain and converged,
            mean_atom_alignment=float(similarity[i,j].mean()))
    result=dict(predictions={
        'pred_a_instrument':all(r['instrument_passed'] for r in results),
        'pred_b_all_converged':all(r['optimization']['converged'] for r in results),
        'pred_c_heldout_reader_gain':any(f['reader_gain_both'] for f in families.values()),
        'pred_d_same_family_tensor_gain':any(f['combined_quality'] for f in families.values())},
        families=families,starts=results,binding=binding,preflight=preflight,baseline_result_sha256=digest(baseline_path),
        baseline_best_reader_capture=best_reader,baseline_best_tensor_capture=best_tensor,
        coefficients=7815168,sparse_indices=1179648,wall_seconds=time.perf_counter()-started,
        body_forwards=0,corpus_access=False,heldout_weights_used_for_basis_fit=False,
        scope='Weight-only oblique reader dictionaries; encoder support remains heuristic, convergence is local, circuit properties untested.')
    with out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],families=families)),flush=True)


if __name__=='__main__':main()
