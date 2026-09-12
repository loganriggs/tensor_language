#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;1365OLSupdates,2conditionalfits,128cachedendpoints;900sec.
"""pred_a exact replay; pred_b2xgraph capture; pred_c native fidelity;
pred_d1.25xrandom capture. Matched conditional cost, weight-only selection.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from shared_dictionary_ols_v1 import select
from quartic_frozen_native_score_v2 import score
STEM='COMPOSED_DICTIONARY_OLS_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    assert json.loads((P/'SHARED_DICTIONARY_OLS_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,products=1365,cached_endpoints=128)));return
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not artifact.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();u-=u.mean(0);root=torch.linalg.cholesky(u.T@u).T;del u
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;rawgram=atom_gram(l1@hs,r1@hs)
    norms=rawgram.diag().sqrt();gram=rawgram/norms[:,None]/norms[None,:]
    native_writer=root@d1;rawcross=native_writer@rawgram;total=float((rawcross*native_writer).sum());cross=rawcross/norms[None,:]
    prior=json.loads((P/'FULL_INPUT_SPARSE_CORE_V2_RESULT.json').read_text())['reports'][1]
    errors=[abs(total/prior['total']-1)]
    support,gains=select(gram,cross,1365);assert len(support)==1365
    random=torch.randperm(4608,generator=torch.Generator().manual_seed(120541))[:1365].tolist()
    reports=[];physical=None
    for name,indices in [('selected',support),('random',random)]:
        g=gram[indices][:,indices];c=cross[:,indices];chol=torch.linalg.cholesky((g+g.T)/2)
        fitted=torch.cholesky_solve(c.T,chol).T;capture=float((fitted*c).sum()/total)
        residual=float((fitted@g-c).norm()/c.norm());errors.append(residual)
        if name=='selected':
            errors.append(abs(capture-sum(gains)/total));physical=torch.linalg.solve_triangular(root,fitted,upper=True)/norms[indices][None,:]
        ev=torch.linalg.eigvalsh(g);reports.append(dict(name=name,capture=capture,solve_residual=residual,condition=float(ev[-1]/ev[0])))
    x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
    ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
    den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    p=((x@l0.T)*(x@r0.T))@d0.T*scale
    products=(p@l1.T)*(p@r1.T);reference=products@d1.T/den[:,None];candidate=products[:,support]@physical.T/den[:,None]
    compact=((p@l1[support].T)*(p@r1[support].T))@physical.T/den[:,None]
    errors.append(float((compact-candidate).norm()/candidate.norm()))
    cached=torch.load(P/'SPARSE_FRAME_NATIVE_V1_WRITES.pt',weights_only=True);errors.append(float((reference.cpu()-cached['reference']).norm()/cached['reference'].norm()))
    previous=json.loads((P/'SPARSE_FRAME_NATIVE_V1_RESULT.json').read_text())['effects']['reference_effects']
    rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];torch.set_default_dtype(torch.float32)
    effects=score([reference.cpu(),cached['candidate'],candidate.cpu()],ports['pre']+ports['native_output'],rows,sd['lm_head.weight'].float(),reference_effects=previous)
    families=[]
    for name in sorted({r['family'] for r in rows}):
        ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==name],device='cuda');ep=(2*ids[:,None]+torch.tensor([0,1],device='cuda')).flatten()
        families.append(dict(family=name,write_relative_error=float((candidate[ep]-reference[ep]).norm()/reference[ep].norm())))
    baseline=json.loads((P/'STREAMED_SPARSE_FRAME_NATIVE_V1_RESULT.json').read_text())['best_capture']
    result={'pred_a':max(errors)<=1e-8 and effects['pred_a'],'pred_b':reports[0]['capture']>=2*baseline,
            'pred_c':effects['pred_b'] and effects['pred_c'] and all(v['write_relative_error']<=.05 for v in families),
            'pred_d':reports[0]['capture']>=1.25*reports[1]['capture']}
    torch.save(dict(support=support,physical_writer=physical.cpu(),candidate_write=candidate.cpu(),required_native_maps=['MLP16.Left','MLP16.Right','MLP16.Down','MLP17.Left[support]','MLP17.Right[support]']),artifact)
    result.update(reports=reports,support=support,random_support=random,identity_errors=errors,effects=effects,families=families,
                  conditional_floats=20642688,graph_baseline_floats=20643840,graph_baseline_capture=baseline,
                  execution_seconds=time.perf_counter()-tic,artifact_sha=digest(artifact),source_shas=binding,
                  scope='Composed trained-product dictionary baseline; greedy support, exact conditional writers; no semantic/OOD circuit claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','effects','support','random_support')}),flush=True)

if __name__=='__main__':main()
