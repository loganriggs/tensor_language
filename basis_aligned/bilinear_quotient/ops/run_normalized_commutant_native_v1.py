#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences; two full-weight matrix-free eigensolves,900sec alarm.
"""pred_a numerical convergence/stability; pred_b rounded cuts<=.1 balanced10%,20%controlgain;
pred_c projector overlap>=.9. Ritz estimates are not certified global lower bounds.
"""
import os,sys,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from normalized_commutant_matrixfree_v1 import NormalizedCommutant
from quartic_matrixfree_eigen_v2 import eigenmatrices
from fullu_input_blocks_v1 import partition_metrics
STEM='NORMALIZED_COMMUTANT_NATIVE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,seeds=2,eigenpairs=2,lanczos_vectors=32,max_actions_per_seed=600)));return
    assert (P/'FULLU_BLOCK_OPTIMIZER_V1_RESULT.json').exists(), 'Read local optimizer terminal result before submission'
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_WITNESSES.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(900);start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=state['lm_head.weight'].double().cuda();u-=u.mean(0)
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(state['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;l,r=l1@hs,r1@hs;g=d1.T@(u.T@u)@d1
    operator=NormalizedCommutant(l,r,g)
    identity_error=float(operator.action(operator.trivial).norm())
    reports=[];witnesses=[];projectors=[]
    for seed in (120440,120441):
        values,matrices,report=eigenmatrices(operator.action,1152,device='cuda',k=2,seed=seed,tol=1e-8,ncv=32,maxiter=100,max_actions=600)
        order=values.argsort()[::-1].copy();values=values[order];matrices=[matrices[i] for i in order]
        report.update(seed=seed,relaxed_values=(2*(1-values)).tolist())
        if matrices:
            x=operator.original_witness(matrices[0]);_,q=torch.linalg.eigh(x);q=q[:,:576];projector=q@q.T
            report.update(partition_metrics(l,r,g,projector),projector_error=float((projector@projector-projector).norm()/projector.norm()))
            witnesses.append(x.cpu());projectors.append(projector.cpu())
        reports.append(report);print(json.dumps(report),flush=True)
    overlap=None;agreement=None
    if len(projectors)==2:
        overlap=float((projectors[0]*projectors[1]).sum()/576);overlap=max(overlap,1-overlap)
    if all(len(x['relaxed_values'])==2 for x in reports):
        agreement=max(abs(a-b) for a,b in zip(reports[0]['relaxed_values'],reports[1]['relaxed_values']))
    previous=json.loads((P/'FULLU_INPUT_BLOCKS_V1_RESULT.json').read_text())['reports']
    controls=[x['normalized_cut'] for x in previous if x['metric']=='producer' and x['kind']=='random_basis'];control=sum(controls)/len(controls)
    a=identity_error<=1e-10 and agreement is not None and agreement<=1e-5 and all(x['status']=='converged' and len(x['relaxed_values'])==2 and max(x['relative_eigen_residuals'])<=1e-7 and x.get('projector_error',1.)<=1e-8 for x in reports)
    b=all(x.get('normalized_cut',float('inf'))<=.1 and .1<=x.get('incident_fraction',0.)<=.9 and x['normalized_cut']<=.8*control for x in reports)
    result={'pred_a':a,'pred_b':b,'pred_c':overlap is not None and overlap>=.9}
    torch.save(dict(witnesses=witnesses,projectors=projectors),artifact)
    result.update(reports=reports,overlap=overlap,estimate_agreement=agreement,identity_error=identity_error,k_condition=float(operator.k_values.max()/operator.k_values.min()),control_mean=control,execution_seconds=time.perf_counter()-start,source_shas=binding,artifact_sha=digest(artifact),limitation='Ritz estimates are not certified global lower bounds.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
