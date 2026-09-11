#!/usr/bin/env python3
# BQGATE: 768 cached tail rows, zero transformer body forwards; no fitting.
"""Extract frozen amplitudes and sensitivities, then score on CPU.
pred_a: cached native CE <=1e-5 and previous four-donor linear replay <=1e-7.
pred_b: each single all-donor linear mean >=.001 on both panels.
pred_c: each paired bootstrap lower95>0 on both panels.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

RUNNER=Path(__file__).resolve()
ROOT=RUNNER.parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
STEM='BRANCH_ALL_DONOR_ALIGNMENT_V1'
sys.path.insert(0, str(RUNNER.parent))


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8<<20), b''):h.update(c)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())
    assert all(digest(path)==sha for path,sha in binding['files'].items())
    from run_branch_context_interchange_v2 import tail, loss_gradient, permutations
    saved=torch.load(P/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',weights_only=True,map_location='cpu')
    panels={}
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        panel=torch.load(P/f'SHARED_NODE_PARENT1_{suffix}_ROWS.pt',weights_only=True,map_location='cpu')
        cache=torch.load(P/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt',weights_only=True,map_location='cpu')
        assert cache['ports']['input'].shape==(384,1152) and len(panel['documents'])==128
        panels[label]=(panel,cache,permutations(panel))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,tail_rows=768,fitting=False)));return
    output=P/(STEM+'_SCALARS.pt'); receipt=P/(STEM+'_EXTRACTION.json')
    assert not output.exists() and not receipt.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(binding['checkpoint'],weights_only=True,mmap=True,map_location='cpu')
    unembedding=checkpoint['lm_head.weight'].float().cuda()
    node=saved['nodes'][1]
    writers=torch.linalg.solve_triangular(saved['output_whitener'].double(),node['writers'].double(),upper=True).cuda()
    old=json.loads((P/'BRANCH_CONTEXT_INTERCHANGE_V2_RESULT.json').read_text())
    results={};errors=[];replays=[]
    for label,(panel,cache,donors) in panels.items():
        x=cache['ports']['input'].double()
        amp=(x@node['reader'].double())[:,None]*(x@node['partners'].double())
        sensitivities=[]
        for off in range(0,384,8):
            h=(cache['ports']['pre'][off:off+8]+cache['ports']['native_output'][off:off+8]).cuda()
            targets=panel['rows'][off:off+8,-1].cuda()
            ce=F.cross_entropy(tail(h,unembedding).double(),targets,reduction='none').cpu()
            errors.append(float((ce-cache['scores'].reshape(384,4)[off:off+8,0]).abs().max()))
            sensitivities.append((loss_gradient(h,targets,unembedding).double()@writers).cpu())
        sensitivity=torch.cat(sensitivities)
        effects=(amp[donors]-amp[None])*sensitivity[None]
        replay=effects.reshape(4,3,128,2).mean((0,1))
        replays.append(float((replay-torch.tensor(old['panels'][label]['linear_document_effects'],dtype=torch.float64)[:,:2]).abs().max()))
        assert bool(torch.isfinite(amp).all() and torch.isfinite(sensitivity).all())
        results[label]=dict(amplitudes=amp.reshape(3,128,2),sensitivities=sensitivity.reshape(3,128,2),
            documents=panel['documents'],domain_labels=panel.get('domain_labels',['all']*128))
    torch.save(results,output)
    result=dict(pred_a=max(errors)<=1e-5 and max(replays)<=1e-7,
        max_cached_ce_error=max(errors),four_donor_replay_errors=replays,
        scalar_sha256=digest(output),binding_sha256=digest(P/(STEM+'_BINDING.json')),
        scope='Extraction only; B/C are pending CPU all-donor covariance analysis.')
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    assert result['pred_a']
    sys.path.insert(0,str(P))
    from branch_all_donor_alignment_v1 import main as score_cpu
    score_cpu()
    scored=json.loads((P/(STEM+'_RESULT.json')).read_text())
    print(json.dumps({key:scored[key] for key in ('pred_a','pred_b','pred_c',)}))


if __name__=='__main__':main()
