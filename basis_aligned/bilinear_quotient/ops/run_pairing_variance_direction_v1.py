#!/usr/bin/env python3
# BQGATE:0bodyforwards;2x512gradientprobes/grade;2048validation;batch64;300sec.
"""pred_a FD<=1e-4 and orthogonality<=1e-8; pred_b noise<=half baseline and gradient cosine>=.5;
pred_c heldout improvement>.001 and >3SE, runtime<=240sec.
Null: no reliable new coefficient descent at this probe budget.
"""
import sys,os,json,time,signal,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from mixed_repeated_contraction_v1 import contract,matchings
from mixed_pairing_control_variate_v1 import pairing_values,sampled_remainder
from graded_source_projection_v1 import graded_norms,retained_grades
STEM='PAIRING_VARIANCE_DIRECTION_V1'
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'MIXED_PAIRING_CONTROL_VARIATE_V1_CONTROL.json').read_text())
    assert max(v for z in control for k,v in z.items() if k!='grade')<=1e-9
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0 body forwards; two independent weight gradients; independent paired validation');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def weight(k):return state[k].double().cuda()
    l=weight('transformer.h.15.mlp.Left.weight');r=weight('transformer.h.15.mlp.Right.weight')
    d=weight('transformer.h.15.mlp.Down.weight')*weight('transformer.h.16.lambdas')[0]
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True)
    a=data['forms'].cuda();wg=data['writer_gram'].cuda();root=data['root'].cuda();whitened=torch.linalg.solve(root,d)
    p=torch.load(P/'GRADED_SOURCE_LBFGS_V1_PROGRAM.pt',weights_only=True)['frames'][0].cuda()
    energies=torch.load(P/'MIXED_REPEATED_NATIVE_V1_ENERGIES.pt',weights_only=True)['energies']
    norms={k:float(energies[str(k)][:,0].mean()) for k in range(1,5)}
    def down(frame):return root@frame@(frame.T@whitened)
    def probes(gen,k):
        b=torch.randint(0,2,(64,4-k,1152),device='cuda',generator=gen).double()*2-1
        x=torch.randint(0,2,(64,2*k,1152),device='cuda',generator=gen).double()*2-1
        return b,x
    def errors(frame,b,x,k,ref):
        e=contract(b,x,l,r,down(frame),a,k)-ref
        return torch.einsum('ni,ij,nj->n',e,wg,e)/norms[k]
    full=graded_norms(a,root@root.T,wg)
    def analytic(frame):
        retained=retained_grades(a,root,frame,wg)
        return sum(math.comb(4,k)*(full[k]-retained[k])/(len(matchings(tuple(range(2*k))))*norms[k]) for k in range(1,5))/4
    frame=p.detach().requires_grad_(True);av=analytic(frame);ag=torch.autograd.grad(av,frame)[0]
    ad=ag-p@(p.T@ag);ad=ad/ad.norm();eps=1e-4
    with torch.no_grad():
        plus=torch.linalg.qr(p+eps*ad,mode='reduced')[0];minus=torch.linalg.qr(p-eps*ad,mode='reduced')[0]
        numeric=(analytic(plus)-analytic(minus))/(2*eps);expected=(ag*ad).sum()
        fd=float(abs(numeric-expected)/expected.abs().clamp_min(1e-12))
    gradients=[];fit_losses=[]
    for seed in (73220,73221):
        gen=torch.Generator(device='cuda').manual_seed(seed);g=ag.detach().clone();losses=[]
        for k in range(1,5):
            for batch in range(8):
                b,x=probes(gen,k)
                with torch.no_grad():ref=pairing_values(b,x,l,r,d,a,k)
                frame=p.detach().requires_grad_(True)
                e=pairing_values(b,x,l,r,down(frame),a,k)-ref
                loss=sampled_remainder(e,wg).mean()/norms[k]
                grad=torch.autograd.grad(loss,frame)[0];g+=grad/32;losses.append(float(loss))
        g=g-p@(p.T@g);gradients.append(g);fit_losses.append(float(av.detach())+sum(losses)/len(losses))
        print(json.dumps(dict(stage='gradient',seed=seed,norm=float(g.norm()),loss=fit_losses[-1])),flush=True)
    cosine=float((gradients[0]*gradients[1]).sum()/(gradients[0].norm()*gradients[1].norm()))
    direction=-gradients[0]/gradients[0].norm();steps=(.01,.05,.15)
    frames=[p]+[torch.linalg.qr(p+step*direction,mode='reduced')[0] for step in steps]
    orth=max(float((f.T@f-torch.eye(128,device='cuda')).norm()) for f in frames)
    gen=torch.Generator(device='cuda').manual_seed(73222);validation={}
    with torch.no_grad():
        for k in range(1,5):
            rows=[]
            for batch in range(32):
                b,x=probes(gen,k);ref=contract(b,x,l,r,d,a,k)
                rows.append(torch.stack([errors(f,b,x,k,ref) for f in frames],-1).cpu())
            validation[str(k)]=torch.cat(rows)
    means=torch.stack([e.mean(0) for e in validation.values()]).mean(0);reports=[]
    for j,step in enumerate(steps,1):
        delta=[e[:,0]-e[:,j] for e in validation.values()]
        gain=sum(float(v.mean()) for v in delta)/4;se=(sum(float(v.var(unbiased=True))/len(v) for v in delta)/16)**.5
        reports.append(dict(step=step,improvement=gain,standard_error=se,loss=float(means[j])))
    seconds=time.perf_counter()-tic;finite=all(bool(torch.isfinite(e).all()) for e in validation.values()) and all(bool(torch.isfinite(g).all()) for g in gradients)
    baseline=torch.load(P/'REPEATED_METRIC_DIRECTION_V1_ARTIFACT.pt',weights_only=True)['gradients']
    old_noise=float((baseline[0]-baseline[1]).square().sum()/2)
    new_noise=float((gradients[0]-gradients[1]).square().sum()/2);noise_ratio=new_noise/old_noise
    result={'pred_a':fd<=1e-4 and orth<=1e-8 and finite,'pred_b':cosine>=.5 and noise_ratio<=.5,'pred_c':any(z['improvement']>.001 and z['improvement']>3*z['standard_error'] for z in reports) and seconds<=240}
    torch.save(dict(frames=[f.cpu() for f in frames],gradients=[g.cpu() for g in gradients],validation=validation),artifact)
    result.update(noise_ratio=noise_ratio,noise_squared=new_noise,analytic_gradient_norm=float((ag-p@(p.T@ag)).norm()),fd_error=fd,orthogonality_error=orth,gradient_cosine=cosine,gradient_norms=[float(g.norm()) for g in gradients],fit_losses=fit_losses,initial_validation_loss=float(means[0]),reports=reports,execution_seconds=seconds,artifact_sha=digest(artifact),source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
