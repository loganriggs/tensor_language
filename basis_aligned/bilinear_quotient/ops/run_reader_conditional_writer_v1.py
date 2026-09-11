#!/usr/bin/env python3
"""pred_a numerical optimum; pred_b no worse; pred_c gain; pred_d baseline gain.
BQGATE:0forwards0seq. Eight frozen weight-only dictionaries, no reader refitting.
"""
import json
import os
from pathlib import Path
import signal
import time
import torch
from run_structured_bilinear_native_v2 import P,CK,digest
from conditional_writer_spectral_v1 import solve
from joint_quadratic_fit_v1 import product_cross,implicit_squared_error
from sparse_reader_program_v1 import SparseReaderProgram
from structured_branch_amplitudes_v1 import inner

PREFIX='READER_CONDITIONAL_WRITER_V1'


@torch.no_grad()
def main():
    binding=json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            arms=8,reader_refitting=False,coefficient_count=7815168,seconds=1800)))
        return
    out=P/f'{PREFIX}_RESULT.json';assert not out.exists()
    parents={}
    for family,prefix in [('orthogonal','FULL_READER_DICTIONARY_MSP_V1'),('oblique','OBLIQUE_READER_DICTIONARY_V1')]:
        path=P/f'{prefix}_RESULT.json'
        report=json.loads(path.read_text())
        assert all(binding.get(path)==sha for path,sha in report['binding'].items())
        parents[family]=dict(report=report,path=str(path),sha256=digest(path))
    assert parents['orthogonal']['report']['predictions']['pred_a_basis_replay']
    assert parents['oblique']['report']['predictions']['pred_a_instrument']
    signal.alarm(1800)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double().cuda()
    readers=torch.cat((l,r))
    metric_receipt=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    metric=torch.load(metric_receipt['cache']['path'],weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    whitener=torch.linalg.cholesky(metric).T
    native=(l,r,whitener@d)
    total=metric_receipt['native_total']
    native_gram=product_cross(l,r,l,r)
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train_ids=torch.cat((order[:3072],order[:3072]+4608)).cuda()
    unit=readers/readers.norm(dim=1,keepdim=True)
    train=unit[train_ids].T
    _,evec=torch.linalg.eigh(train@train.T)
    arms=[dict(family='baseline',seed=name,basis=basis) for name,basis in
          [('identity',torch.eye(1152,device='cuda')),('pca',evec.T)]]
    for report in parents['orthogonal']['report']['starts']:
        arms.append(dict(family='orthogonal',seed=report['seed'],report=report))
    for report in parents['oblique']['report']['starts']:
        arms.append(dict(family=report['method'],seed=report['seed'],report=report))
    results=[]
    for arm in arms:
        family,seed=arm['family'],arm['seed'];label=f'{family}_{seed}'
        if family=='baseline':
            basis=arm['basis'];coordinates=readers@basis.T
            indices=coordinates.abs().topk(128,dim=1).indices
            values=coordinates.gather(1,indices)
            artifact=dict(analysis_basis=basis.cpu(),code_indices=indices.to(torch.int16).cpu(),code_values=values.cpu())
            path=Path(f'/dev/shm/bilin18_reader_writer_v1_{label}_readers.pt')
            assert not path.exists();torch.save(artifact,path)
            source=dict(path=str(path),sha256=digest(path));converged=None
            expected_capture=parents['orthogonal']['report']['baselines'][seed]['coefficient_capture']
        else:
            source=arm['report']['cache'];assert digest(source['path'])==source['sha256']
            artifact=torch.load(source['path'],weights_only=True,map_location='cpu')
            converged=arm['report']['optimization']['converged']
            expected_capture=arm['report']['scores']['coefficient_capture']
        program=SparseReaderProgram.from_artifact(artifact,d,bias)
        reconstructed=torch.sparse.mm(program.codes,program.analysis_basis)
        a,b=reconstructed.split(4608)
        before=(a,b,native[2])
        before_capture=1-float((inner(before,before)-2*inner(native,before)+total)/total)
        parent_replay=abs(before_capture-expected_capture)
        assert parent_replay<=1e-8
        t=time.perf_counter()
        w,cross,gram,diagnostics=solve(d,l,r,a,b)
        solve_seconds=time.perf_counter()-t
        cache=Path(f'/dev/shm/bilin18_reader_writer_v1_{label}_down.pt')
        assert not cache.exists()
        torch.save(dict(down=w.cpu(),reader_artifact=source,diagnostics=diagnostics,binding=binding),cache)
        receipt=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size)
        with (P/f'{PREFIX}_{label}_SOLVE.json').open('x') as f:
            json.dump(dict(cache=receipt,source=source,diagnostics=diagnostics,solve_seconds=solve_seconds),f,indent=2);f.write('\n')
        proposal=(a,b,whitener@w)
        after_energy=inner(proposal,proposal)
        chunked=(after_energy-2*inner(native,proposal)+total)/total
        implicit=implicit_squared_error(metric,d,native_gram,w,cross,gram)/total
        capture=1-float(chunked)
        loss_replay=float(abs(chunked-implicit))
        cosine=float(inner(before,proposal)/(inner(before,before)*after_energy).sqrt())
        program=SparseReaderProgram.from_artifact(artifact,w,bias)
        x=torch.randn(8,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(755))
        direct=((x@a.T)*(x@b.T))@w.T+bias
        executor_replay=float((program(x)-direct).norm()/direct.norm())
        instrument=all(torch.isfinite(torch.tensor(value)).item() for value in
            [capture,cosine,loss_replay,executor_replay,diagnostics['normal_residual']]) and max(
            parent_replay,loss_replay,executor_replay,diagnostics['normal_residual'])<=1e-8
        result=dict(family=family,seed=seed,parent_converged=converged,source=source,cache=receipt,
            diagnostics=diagnostics,solve_seconds=solve_seconds,parent_capture_replay=parent_replay,
            before_capture=before_capture,after_capture=capture,gain=capture-before_capture,
            loss_replay=loss_replay,executor_replay=executor_replay,function_cosine=cosine,
            writer_norm_ratio=float(w.norm()/d.norm()),instrument_passed=instrument)
        with (P/f'{PREFIX}_{label}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result),flush=True);results.append(result)
        assert instrument
        del program,reconstructed,a,b,w,cross,gram,proposal,artifact
    best=max(row['after_capture'] for row in results if row['family']=='baseline')
    families={}
    for family in ('orthogonal','ordinary_covariance','tyler'):
        rows=[row for row in results if row['family']==family]
        c=all(row['gain']>=.01 for row in rows)
        dd=all(row['after_capture']>=best+.01 for row in rows)
        families[family]=dict(gain_both=c,baseline_gain_both=dd,combined=c and dd,
            parent_converged_both=all(row['parent_converged'] for row in rows))
    predictions={'pred_a_instrument':all(row['instrument_passed'] for row in results),
        'pred_b_no_worsening':all(row['gain']>=-1e-8 for row in results),
        'pred_c_gain':any(row['gain_both'] for row in families.values()),
        'pred_d_baseline_gain':any(row['baseline_gain_both'] for row in families.values())}
    result=dict(predictions=predictions,arms=results,families=families,binding=binding,
        parent_results={k:{key:value for key,value in v.items() if key!='report'} for k,v in parents.items()},
        wall_seconds=time.perf_counter()-started,body_forwards=0,corpus_access=False,
        full_weights_used_for_writer_fit=True,coefficients=7815168,sparse_indices=1179648,
        scope='Conditional coefficient optimum of frozen readers; no global reader optimum or circuit validation.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=predictions,families=families)),flush=True)


if __name__=='__main__':main()
