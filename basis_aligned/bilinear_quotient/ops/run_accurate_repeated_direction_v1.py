#!/usr/bin/env python3
# BQGATE:0bodyforwards;2x65536gradientprobes/grade;16384validation;batch64;900sec.
"""pred_a FD<=1e-4 and orthogonality<=1e-8; pred_b gradient cosine>=.5;
pred_c heldout improvement>.001 and >3SE, runtime<=840sec.
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
STEM='ACCURATE_REPEATED_DIRECTION_V1'
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'MIXED_PAIRING_CONTROL_VARIATE_V1_CONTROL.json').read_text())
    assert max(v for z in control for k,v in z.items() if k!='grade')<=1e-9
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('0 body forwards; two independent weight gradients; independent paired validation');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
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
    gradients=[];fit_losses=[];blocks=[];checkpoints=[]
    for seed in (73240,73241):
        gen=torch.Generator(device='cuda').manual_seed(seed);arm_blocks=[];arm_checks={};losses=[]
        for block in range(16):
            g=ag.detach().clone();block_losses=[]
            # Grade1 remainder is identically zero and is integrated analytically.
            for k in range(2,5):
                for batch in range(64):
                    b,x=probes(gen,k)
                    with torch.no_grad():ref=pairing_values(b,x,l,r,d,a,k)
                    frame=p.detach().requires_grad_(True)
                    e=pairing_values(b,x,l,r,down(frame),a,k)-ref
                    loss=sampled_remainder(e,wg).mean()/norms[k]
                    grad=torch.autograd.grad(loss,frame)[0];g+=grad/256;block_losses.append(float(loss.detach()))
            g=g-p@(p.T@g);arm_blocks.append(g.cpu());losses.append(float(av.detach())+sum(block_losses)/256)
            running=torch.stack(arm_blocks).mean(0)
            if block+1 in (1,4,16):arm_checks[str((block+1)*4096)]=running.clone()
            print(json.dumps(dict(stage='gradient_block',seed=seed,block=block+1,probes_per_grade=(block+1)*4096,norm=float(running.norm()),elapsed=time.perf_counter()-tic)),flush=True)
        gradients.append(torch.stack(arm_blocks).mean(0).cuda());blocks.append(torch.stack(arm_blocks));checkpoints.append(arm_checks);fit_losses.append(sum(losses)/len(losses))
    cosine=float((gradients[0]*gradients[1]).sum()/(gradients[0].norm()*gradients[1].norm()))
    direction=-gradients[0]/gradients[0].norm();steps=(.05,.15,.5)
    frames=[p]+[torch.linalg.qr(p+step*direction,mode='reduced')[0] for step in steps]
    orth=max(float((f.T@f-torch.eye(128,device='cuda')).norm()) for f in frames)
    gen=torch.Generator(device='cuda').manual_seed(73242);validation={}
    with torch.no_grad():
        for k in range(1,5):
            rows=[]
            for batch in range(256):
                b,x=probes(gen,k);ref=contract(b,x,l,r,d,a,k)
                rows.append(torch.stack([errors(f,b,x,k,ref) for f in frames],-1).cpu())
            validation[str(k)]=torch.cat(rows)
    means=torch.stack([e.mean(0) for e in validation.values()]).mean(0);reports=[]
    for j,step in enumerate(steps,1):
        delta=[e[:,0]-e[:,j] for e in validation.values()]
        gain=sum(float(v.mean()) for v in delta)/4;se=(sum(float(v.var(unbiased=True))/len(v) for v in delta)/16)**.5
        reports.append(dict(step=step,improvement=gain,standard_error=se,loss=float(means[j])))
    seconds=time.perf_counter()-tic;finite=all(bool(torch.isfinite(e).all()) for e in validation.values()) and all(bool(torch.isfinite(g).all()) for g in gradients)
    replica_noise=float((gradients[0]-gradients[1]).square().sum()/2)
    agreement={key:float(torch.nn.functional.cosine_similarity(checkpoints[0][key].flatten(),checkpoints[1][key].flatten(),dim=0)) for key in checkpoints[0]}
    block_noise=[float((z-z.mean(0)).square().sum()/(len(z)-1)/len(z)) for z in blocks]
    result={'pred_a':fd<=1e-4 and orth<=1e-8 and finite,'pred_b':cosine>=.5,'pred_c':any(z['improvement']>.001 and z['improvement']>3*z['standard_error'] for z in reports) and seconds<=840}
    torch.save(dict(frames=[f.cpu() for f in frames],gradients=[g.cpu() for g in gradients],validation=validation,blocks=blocks,checkpoints=checkpoints),artifact)
    result.update(noise_squared=replica_noise,block_estimated_mean_noise=block_noise,checkpoint_agreement=agreement,analytic_gradient_norm=float((ag-p@(p.T@ag)).norm()),fd_error=fd,orthogonality_error=orth,gradient_cosine=cosine,gradient_norms=[float(g.norm()) for g in gradients],fit_losses=fit_losses,initial_validation_loss=float(means[0]),reports=reports,execution_seconds=seconds,artifact_sha=digest(artifact),source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
