#!/usr/bin/env python3
"""pred_a exact basis; pred_b bothconverged; pred_c heldoutweightgain.
BQGATE:0forwards0seq. Complete input dictionary; no corpus or activations.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
from run_structured_bilinear_native_v2 import P, CK, digest
import torch
from scipy.optimize import linear_sum_assignment
from orthogonal_reader_msp_v1 import fit, topk_energy
from structured_branch_amplitudes_v1 import inner

PREFIX = 'FULL_READER_DICTIONARY_MSP_V1'


@torch.no_grad()
def main():
    binding = json.loads((P/f'{PREFIX}_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
            dimension=1152,training_weight_vectors=6144,heldout_weight_vectors=3072,
            seeds=[0,937],seconds_per_seed=900,coefficients=7815168,sparse_indices=1179648)))
        return
    out=P/f'{PREFIX}_RESULT.json'
    assert not out.exists()
    signal.alarm(2400)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    readers=torch.cat((l,r))
    norms=readers.norm(dim=1)
    assert float(norms.min())>1e-8
    y=(readers/norms[:,None]).T
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))
    train_ids=torch.cat((order[:3072],order[:3072]+4608)).cuda()
    test_ids=torch.cat((order[3072:],order[3072:]+4608)).cuda()
    train,test=y[:,train_ids],y[:,test_ids]
    assert len(set(train_ids.tolist()).intersection(test_ids.tolist()))==0
    _,vectors=torch.linalg.eigh(train@train.T)
    bases={'identity':torch.eye(1152,device='cuda'),'pca':vectors.T}
    u=sd['lm_head.weight'].double().cuda()
    whitener=torch.linalg.cholesky(u.T@u).T
    del u
    native=(l,r,whitener@d)
    total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']

    def score(a):
        coordinates=readers@a.T
        ids=coordinates.abs().topk(128,dim=1).indices
        sparse=torch.zeros_like(coordinates).scatter_(1,ids,coordinates.gather(1,ids))
        reconstructed=sparse@a
        proposal=(reconstructed[:4608],reconstructed[4608:],native[2])
        residual=float((inner(proposal,proposal)-2*inner(native,proposal)+total)/total)
        return dict(train_top128_energy=topk_energy(a,train,128),
            test_top128_energy=topk_energy(a,test,128),coefficient_capture=1-residual,
            coefficient_residual=residual,
            orthogonality_error=float((a@a.T-torch.eye(1152,device='cuda')).norm()),
            full_coordinate_reader_replay=float(((readers@a.T)@a-readers).norm()/readers.norm()))

    baseline={name:score(a) for name,a in bases.items()}
    print(json.dumps(dict(baselines=baseline)),flush=True)
    results=[]
    learned=[]
    for seed in (0,937):
        a,optimization=fit(train,seed,max_steps=20000,max_seconds=900,
            callback=lambda row:print(json.dumps(dict(seed=seed,phase='fit',**row)),flush=True))
        cache=Path(f'/dev/shm/bilin18_full_reader_msp_v1_s{seed}.pt')
        assert not cache.exists()
        coordinates=readers@a.T
        code_indices=coordinates.abs().topk(128,dim=1).indices
        code_values=coordinates.gather(1,code_indices)
        torch.save(dict(analysis_basis=a.cpu(),order=order,seed=seed,
            code_indices=code_indices.to(torch.int16).cpu(),code_values=code_values.cpu(),
            retained_down_key='transformer.h.17.mlp.Down.weight',
            history=optimization['history'],binding=binding,sparsity=128),cache)
        scores=score(a)
        report=dict(seed=seed,optimization=optimization,scores=scores,
            cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size))
        with (P/f'{PREFIX}_SEED_{seed}.json').open('x') as f:
            json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps(dict(seed=seed,scores=scores,converged=optimization['converged'])),flush=True)
        results.append(report)
        learned.append(a)
    similarity=(learned[0]@learned[1].T).abs().cpu().numpy()
    i,j=linear_sum_assignment(-similarity)
    alignment=dict(mean=float(similarity[i,j].mean()),minimum=float(similarity[i,j].min()))
    best=max(r['test_top128_energy'] for r in baseline.values())
    all_scores=list(baseline.values())+[r['scores'] for r in results]
    result=dict(predictions={
        'pred_a_basis_replay':all(max(s['orthogonality_error'],s['full_coordinate_reader_replay'])<=1e-10 for s in all_scores),
        'pred_b_both_converged':all(r['optimization']['converged'] for r in results),
        'pred_c_heldout_weight_gain':all(r['scores']['test_top128_energy']>=best+.05 for r in results)},
        baselines=baseline,starts=results,atom_alignment=alignment,binding=binding,
        wall_seconds=time.perf_counter()-started,body_forwards=0,corpus_access=False,
        coefficients=7815168,sparse_indices=1179648,
        scope='Full-rank shared reader organization of native products; weight holdout, not behavioral/OOD evidence or joint tensor optimum.')
    with out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['predictions']),flush=True)


if __name__=='__main__':
    main()
