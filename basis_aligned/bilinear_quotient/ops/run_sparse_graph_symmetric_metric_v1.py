#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;8192synthetic quartic probes;300sec.
"""pred_a exact contraction identities; pred_b25%relative symmetric capture gain;
pred_c cross-distribution agreement andSE<=.01. Frozen graph, no fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from sparse_frame_function_inner_v1 import coefficients
from composed_quartic_contraction_v1 import contract
STEM='SPARSE_GRAPH_SYMMETRIC_METRIC_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'COMPOSED_QUARTIC_CONTRACTION_V1_CONTROL.json').read_text());assert all(v for k,v in control.items() if k.startswith('pred_'))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,synthetic_probes=8192)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();uc=u-u.mean(0);roots={'full':torch.linalg.cholesky(u.T@u).T,'centered':torch.linalg.cholesky(uc.T@uc).T};del u,uc
    weights=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    l0,r0,d0,l1,r1,d1=weights;scale=float(sd['transformer.h.17.lambdas'][0])
    h=scale**2*d0@atom_gram(l0,r0)@d0.T;he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    fit=json.loads((P/'STREAMED_SPARSE_FRAME_NATIVE_V1_RESULT.json').read_text());best=max(range(2),key=lambda i:fit['reports'][i]['history'][-1]['capture'])
    frames=torch.load(P/'STREAMED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt',weights_only=True);q=frames['frames'][best].cuda();edges=frames['edges'][best].cuda()
    writer=coefficients(q,l1@hs,r1@hs,d1,edges);physical=hi@q;i,j=edges
    graph=[l0,r0,d0,physical[:,i].T,physical[:,j].T,writer*torch.where(i==j,1.,2.**.5)]
    errors=[];gen=torch.Generator(device='cuda').manual_seed(120543);x=torch.randn(8,4,1152,device='cuda',generator=gen)
    for w in (weights,graph):
        y=contract(x,*w,scale);permuted=contract(x[:,[2,0,3,1]],*w,scale)
        z=x[:,0];p=((z@w[0].T)*(z@w[1].T))@w[2].T*scale
        direct=((p@w[3].T)*(p@w[4].T))@w[5].T
        diagonal=contract(z[:,None,:].expand(-1,4,-1),*w,scale)
        errors.extend([float((y-permuted).norm()/y.norm()),float((direct-diagonal).norm()/direct.norm())])
    totals=json.loads((P/'FULLU_PAIRED_PRODUCER_V1_RESULT.json').read_text())['reports']
    paired={name:float((root@writer).square().sum())/next(v['total_paired_coefficient_energy'] for v in totals if v['name']==name+'_producer') for name,root in roots.items()}
    errors.append(abs(paired['centered']-fit['best_capture']))
    reports=[]
    for distribution,seed in [('gaussian',120547),('rademacher',120551)]:
        gen=torch.Generator(device='cuda').manual_seed(seed);samples={name:[] for name in roots}
        for _ in range(64):
            x=torch.randn(64,4,1152,device='cuda',generator=gen) if distribution=='gaussian' else torch.randint(0,2,(64,4,1152),device='cuda',generator=gen).double()*2-1
            target=contract(x,*weights,scale);approx=contract(x,*graph,scale)
            for name,root in roots.items():
                a=(target@root.T).square().sum(-1);b=((target-approx)@root.T).square().sum(-1);samples[name].append(torch.stack([a,b],1))
        metrics={}
        for name,batches in samples.items():
            v=torch.cat(batches);ratio=v[:,1].mean()/v[:,0].mean();se=(v[:,1]-ratio*v[:,0]).std(unbiased=True)/(len(v)**.5*v[:,0].mean())
            metrics[name]=dict(capture=float(1-ratio),estimated_capture_se=float(se),target_norm2=float(v[:,0].mean()),residual_norm2=float(v[:,1].mean()))
        reports.append(dict(distribution=distribution,seed=seed,probes=4096,metrics=metrics));print(json.dumps(reports[-1]),flush=True)
    a,b=[v['metrics']['centered'] for v in reports];combined=(a['estimated_capture_se']**2+b['estimated_capture_se']**2)**.5
    result={'pred_a':max(errors)<=1e-9 and all(torch.isfinite(torch.tensor([m['capture'],m['estimated_capture_se'],m['target_norm2'],m['residual_norm2']])).all().item() for v in reports for m in v['metrics'].values()),
            'pred_b':all(v['metrics']['centered']['capture']>=1.25*paired['centered'] for v in reports),
            'pred_c':abs(a['capture']-b['capture'])<=3*combined and max(a['estimated_capture_se'],b['estimated_capture_se'])<=.01}
    result.update(reports=reports,paired_capture=paired,identity_errors=errors,execution_seconds=time.perf_counter()-tic,source_shas=binding,
                  scope='Frozen graph rescore under symmetric quartic coefficients; synthetic estimator, no text/refit/native behavior claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','reports')}),flush=True)

if __name__=='__main__':main()
