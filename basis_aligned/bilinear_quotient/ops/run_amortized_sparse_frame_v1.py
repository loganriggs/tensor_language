#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;2weight-only full-frame fits;1500sec alarm.
"""pred_a identities<=1e-8; pred_b both locally converge; pred_c10%gain over oldbest;
pred_d cross-start functioncos>=.95. 600sec/2000steps each; no data fit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from fullu_input_blocks_v1 import sandwich
from amortized_sparse_frame_v1 import fit,control
from sparse_frame_function_inner_v1 import coefficients,inner,control as inner_control
STEM='AMORTIZED_SPARSE_FRAME_NATIVE_V1'

def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    controls=[control(),inner_control()];assert all(v['passed'] for v in controls)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,controls=controls)));return
    signal.alarm(1500);tic=time.perf_counter();torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_FRAMES.pt');assert not out.exists() and not artifact.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();u-=u.mean(0)
    metric=u.T@u;root=torch.linalg.cholesky(metric).T;del u
    l0,r0,d0,l1,r1,d1=[sd[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(sd['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;l,r=l1@hs,r1@hs;w=root@d1
    eye=torch.eye(1152,device='cuda');k=sandwich(l,r,d1.T@metric@d1,eye);k=(k+k.T)/2
    ev,frame=torch.linalg.eigh(k);frame=frame.flip(1);total=float(torch.trace(k))
    prior=json.loads((P/'FULL_INPUT_SPARSE_CORE_V2_RESULT.json').read_text())['reports'][1]
    errors=[abs(total/prior['total']-1)];reports=[];frames=[];edges=[];cs=[]
    oldfit=json.loads((P/'STREAMED_SPARSE_FRAME_NATIVE_V1_RESULT.json').read_text())
    oldframes=torch.load(P/'STREAMED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt',weights_only=True)
    for index,arm in enumerate(('spectral','haar120463')):
        q=oldframes['frames'][index].cuda()
        def callback(q,e,row):
            print(json.dumps(dict(arm=arm,**row)),flush=True)
            if row['iteration']%20==0:
                torch.save(dict(arm=arm,q=q.cpu(),edges=e.cpu(),row=row),P/(STEM+'_PROGRESS.pt'))
        q,e,report=fit(q,l,r,w,total,4096,max_steps=2000,seconds=600,tolerance=1e-6,inner_steps=20,callback=callback)
        c=coefficients(q,l,r,w,e)
        errors.extend([float((q.T@q-eye).abs().max()),abs(float(c.square().sum()/total)-report['history'][-1]['capture'])])
        errors.append(abs(report['history'][0]['capture']/oldfit['reports'][index]['history'][-1]['capture']-1))
        reports.append(dict(arm=arm,**report));frames.append(q);edges.append(e);cs.append(c)
        torch.save(dict(frames=[v.cpu() for v in frames],edges=[v.cpu() for v in edges],reports=reports,
                        scope='Regenerate writers and physical producer readers from bound weights; not a standalone program.'),artifact)
    cosine=float(inner(frames[0],edges[0],cs[0],frames[1],edges[1],cs[1])/(cs[0].norm()*cs[1].norm()))
    best=max(r['history'][-1]['capture'] for r in reports)
    result={'pred_a':max(errors)<=1e-8,'pred_b':all(r['converged'] for r in reports),
            'pred_c':best>=1.10*oldfit['best_capture'],'pred_d':cosine>=.95}
    result.update(reports=reports,cosine=cosine,best_capture=best,old_best_capture=oldfit['best_capture'],identity_errors=errors,
                  controls=controls,execution_seconds=time.perf_counter()-tic,source_shas=binding,artifact_sha=digest(artifact),
                  scope='Continuation from old terminal frames with amortized support and PR+; paired coefficients; no causal circuit claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('reports','source_shas')}),flush=True)

if __name__=='__main__':main()
