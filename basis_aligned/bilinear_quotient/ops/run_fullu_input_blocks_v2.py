#!/usr/bin/env python3
# BQGATE: 0 body forwards, 0 text sequences; full-weight block screen, 600sec limit.
"""Registered in FULLU_INPUT_BLOCKS_V1_MATH.md. pred_a numerical; pred_b cuts;
pred_c cross-seed stability. No behavioral circuit or adoption claim.
"""
import os, sys, json, time, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from fullu_input_blocks_v1 import sandwich, partition_metrics
from quadratic_producer_projection_v1 import atom_gram
from sparse_path_stability_atlas_v1 import digest
STEM='FULLU_INPUT_BLOCKS_V1'

def split(forms, basis):
    rotated=basis.T @ forms @ basis
    edges=rotated.square().sum(0);edges.fill_diagonal_(0)
    degree=edges.sum(0)
    norm=degree.rsqrt()
    lap=torch.eye(len(degree),device=edges.device)-norm[:,None]*edges*norm[None,:]
    vals,vecs=torch.linalg.eigh(lap)
    residual=float((lap@vecs[:,1]-vals[1]*vecs[:,1]).norm())
    ids=vecs[:,1].argsort()[:len(degree)//2]
    readers=basis[:,ids]
    return readers@readers.T,residual

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_V2_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,metrics=2,seeds=2,sketches=16,time_limit=600)));return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROJECTORS.pt')
    assert not out.exists() and not artifact.exists()
    signal.alarm(600);start=time.perf_counter()
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
    u=state['lm_head.weight'].double().cuda();u-=u.mean(0)
    root=torch.linalg.cholesky(u.T@u)
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{n}.mlp.{s}.weight'].double().cuda() for n in (16,17) for s in ('Left','Right','Down')]
    scale=float(state['transformer.h.17.lambdas'][0])
    h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T
    writers=root.T@d1;g=writers.T@writers
    reports=[];saved={};errors=[]
    for name,transform in [('identity',torch.eye(1152,device='cuda')),('producer',hs)]:
        l,r=l1@transform,r1@transform
        k=sandwich(l,r,g,torch.eye(1152,device='cuda'))
        for seed in (120423,120424):
            rng=torch.Generator(device='cuda').manual_seed(seed)
            sketch=torch.randn(16,1152,generator=rng,device='cuda')@writers/4
            forms=torch.stack([l.T@(w[:,None]*r) for w in sketch]);forms=(forms+forms.transpose(-1,-2))/2
            _,basis=torch.linalg.eigh(forms[0])
            random_basis=torch.linalg.qr(torch.randn(1152,1152,generator=rng,device='cuda')).Q
            for kind,q in [('candidate',basis),('random_basis',random_basis)]:
                projector,error=split(forms,q);errors.append(error)
                errors.append(float((projector@projector-projector).norm()/projector.norm()))
                errors.append(abs(float(torch.trace(projector))-576)/576)
                metrics=partition_metrics(l,r,g,projector,k)
                reports.append(dict(metric=name,seed=seed,kind=kind,**metrics))
                if kind=='candidate': saved[f'{name}_{seed}']=projector.cpu()
                print(json.dumps(reports[-1]),flush=True)
    agreements={}
    for name in ('identity','producer'):
        overlap=float((saved[f'{name}_120423']*saved[f'{name}_120424']).sum()/576)
        agreements[name]=max(overlap,1-overlap)
    b=True
    for seed in (120423,120424):
        candidate,control=[next(r for r in reports if r['metric']=='producer' and r['seed']==seed and r['kind']==kind) for kind in ('candidate','random_basis')]
        b &= candidate['normalized_cut']<=.1 and .1<=candidate['incident_fraction']<=.9 and candidate['normalized_cut']<=.8*control['normalized_cut']
    finite=all(torch.isfinite(torch.tensor([r[k] for k in ('normalized_cut','incident_fraction','offblock_energy_fraction')])).all().item() for r in reports)
    torch.save(saved,artifact)
    result=dict(**{'pred_a':max(errors)<1e-8 and finite,'pred_b':bool(b),'pred_c':agreements['producer']>=.9},
                reports=reports,agreement=agreements,numerical_errors=errors,execution_seconds=time.perf_counter()-start,
                source_shas=binding,artifact_sha=digest(artifact),scope='weight-only block screen; no behavioral identification')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
