#!/usr/bin/env python3
# BQGATE:0bodyforwards;2x512gradientprobes/grade;2048validation;batch64;300sec.
"""pred_a FD<=1e-4 and orthogonality<=1e-8; pred_b gradient cosine>=.5;
pred_c heldout improvement>.001 and >3SE, runtime<=240sec.
Null: no reliable new coefficient descent at this probe budget.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from mixed_repeated_contraction_v1 import contract
STEM='REPEATED_METRIC_DIRECTION_V1'
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
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
    gradients=[];fit_losses=[];fd=None
    for seed in (73220,73221):
        gen=torch.Generator(device='cuda').manual_seed(seed);g=torch.zeros_like(p);losses=[]
        for k in range(1,5):
            for batch in range(8):
                b,x=probes(gen,k)
                with torch.no_grad():ref=contract(b,x,l,r,d,a,k)
                frame=p.detach().requires_grad_(True);loss=errors(frame,b,x,k,ref).mean()
                grad=torch.autograd.grad(loss,frame)[0];g+=grad/32;losses.append(float(loss))
                if fd is None:
                    direction=grad-p@(p.T@grad);direction/=direction.norm();eps=1e-4
                    with torch.no_grad():numeric=(errors(p+eps*direction,b,x,k,ref).mean()-errors(p-eps*direction,b,x,k,ref).mean())/(2*eps)
                    analytic=(grad*direction).sum();fd=float(abs(numeric-analytic)/analytic.abs().clamp_min(1e-12))
        g=g-p@(p.T@g);gradients.append(g);fit_losses.append(sum(losses)/len(losses))
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
    result={'pred_a':fd<=1e-4 and orth<=1e-8 and finite,'pred_b':cosine>=.5,'pred_c':any(z['improvement']>.001 and z['improvement']>3*z['standard_error'] for z in reports) and seconds<=240}
    torch.save(dict(frames=[f.cpu() for f in frames],gradients=[g.cpu() for g in gradients],validation=validation),artifact)
    result.update(fd_error=fd,orthogonality_error=orth,gradient_cosine=cosine,gradient_norms=[float(g.norm()) for g in gradients],fit_losses=fit_losses,initial_validation_loss=float(means[0]),reports=reports,execution_seconds=seconds,artifact_sha=digest(artifact),source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
