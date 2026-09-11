#!/usr/bin/env python3
# BQGATE: 3072 cached tail rows, zero transformer body forwards, frozen donor assignment.
"""Finite private-matched port screen; no fitting or mean compensation.
pred_a: cached native CE<=1e-5, product algebra<=1e-9, prior shared derivative<=1e-7.
pred_b: previously registered donor geometry/mean-shift bars and source checks hold.
pred_c: common-only joint mean CE cost >=.001 on each of the two existing panels.
Null: no beneficial independent common-port variation under this adequate matched control.
C is a descriptive screen, not significance or fresh confirmation.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3]
P=ROOT/'basis_aligned/polynomial_causal';STEM='BRANCH_PRIVATE_MATCHED_PORTS_V1'
sys.path.insert(0,str(RUNNER.parent))


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8<<20),b''):h.update(c)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())
    assert all(digest(path)==sha for path,sha in binding['files'].items())
    from run_branch_context_interchange_v2 import tail,loss_gradient
    matched=torch.load(P/'BRANCH_PRIVATE_MATCHED_DONORS_V1.pt',weights_only=True,map_location='cpu')
    quality=json.loads((P/'BRANCH_PRIVATE_MATCHED_DONORS_V1.json').read_text())
    panels={}
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        rows=torch.load(P/f'SHARED_NODE_PARENT1_{suffix}_ROWS.pt',weights_only=True,map_location='cpu')
        cache=torch.load(P/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt',weights_only=True,map_location='cpu')
        donor=matched[label]['donors']
        assert donor.shape==(384,) and torch.equal(donor.sort().values,torch.arange(384))
        assert bool((donor!=torch.arange(384)).all())
        assert matched[label]['documents']==rows['documents']
        labels=matched[label]['domain_labels']
        for i,j in enumerate(donor.tolist()):
            assert i//128==j//128 and labels[i%128]==labels[j%128]
        panels[label]=(rows,cache)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,tail_rows=3072,fitting=False)));return
    output=P/(STEM+'_RESULT.json');assert not output.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    started=time.perf_counter()
    weights=torch.load(binding['checkpoint'],weights_only=True,mmap=True,map_location='cpu')
    unembedding=weights['lm_head.weight'].float().cuda()
    saved=torch.load(P/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',weights_only=True,map_location='cpu')
    writers=torch.linalg.solve_triangular(saved['output_whitener'].double(),saved['nodes'][1]['writers'].double(),upper=True).cuda()
    errors=[];algebra=[];derivative=[];results={}
    for label,(panel,cache) in panels.items():
        m=matched[label];s=m['common_values'].cuda();p=m['private_values'].cuda();donor=m['donors'].cuda()
        ds=s[donor]-s;dp=p[donor]-p
        amplitude=[ds[:,None]*p,s[:,None]*dp,s[donor,None]*p[donor]-s[:,None]*p]
        mixed=ds[:,None]*dp
        algebra.append(float((amplitude[2]-amplitude[0]-amplitude[1]-mixed).norm()/amplitude[2].norm()))
        changes=[a@writers.T for a in amplitude]
        exact=[];linear=[];common_branches=[]
        for off in range(0,384,8):
            h=(cache['ports']['pre'][off:off+8]+cache['ports']['native_output'][off:off+8]).cuda()
            target=panel['rows'][off:off+8,-1].cuda()
            ce=F.cross_entropy(tail(h,unembedding).double(),target,reduction='none')
            errors.append(float((ce.cpu()-cache['scores'].reshape(384,4)[off:off+8,0]).abs().max()))
            grad=loss_gradient(h,target,unembedding).double()
            exact.append(torch.stack([F.cross_entropy(tail(h+d[off:off+8].float(),unembedding).double(),target,reduction='none')-ce for d in changes],-1).cpu())
            linear.append(torch.stack([(grad*d[off:off+8]).sum(1) for d in changes],-1).cpu())
            common_branches.append((amplitude[0][off:off+8]*(grad@writers)).cpu())
        effects=torch.cat(exact).reshape(3,128,3)
        approx=torch.cat(linear).reshape(3,128,3)
        branch=torch.cat(common_branches).mean(0)
        derivative.append(float((branch-torch.tensor(quality['panels'][label]['mean_linear_effect'],dtype=torch.float64)).abs().max()))
        assert bool(torch.isfinite(effects).all() and torch.isfinite(approx).all())
        means=effects.mean((0,1))
        results[label]=dict(mean_ce_change=means.tolist(),mean_linear_change=approx.mean((0,1)).tolist(),
            common_branch_linear=branch.tolist(),nonlinear_remainder=(effects-approx).mean((0,1)).tolist(),
            per_family=effects.mean(1).tolist(),per_domain={name:effects[:,torch.tensor([i for i,v in enumerate(m['domain_labels']) if v==name])].mean((0,1)).tolist() for name in sorted(set(m['domain_labels']))},
            document_mean_effects=effects.mean(0).tolist(),held=bool(means[0]>=.001))
    a=max(errors)<=1e-5 and max(algebra)<=1e-9 and max(derivative)<=1e-7
    b=a and quality['pred_a'] and quality['pred_b'] and quality['pred_c']
    result={'pred_a':a,'pred_b':b,'pred_c':b and all(v['held'] for v in results.values()),
        'max_native_ce_replay':max(errors),'algebra_errors':algebra,'derivative_errors':derivative,
        'intervention_order':['common_reader_only','both_private_partners','coherent_all_ports'],
        'panels':results,'seconds':time.perf_counter()-started,
        'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'price':{'body_forwards':0,'tail_rows':3072,'native_background_retained':True},
        'scope':'Frozen private-matched port swaps on existing panels. No mean compensation or parameter fitting. '
                'Descriptive empirical screen, no significance, fresh OOD, standalone extraction or semantic claim.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='panels'}))
    print(json.dumps({k:{j:v for j,v in r.items() if j!='document_mean_effects'} for k,r in results.items()},indent=2))


if __name__=='__main__':main()
