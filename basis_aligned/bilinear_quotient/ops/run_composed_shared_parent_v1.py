#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;4weight-only shared-parent fits,900sec alarm.
"""pred_a identities<=1e-8; pred_b all normalized tangent gradients<=1e-8;
pred_c readercos>=.99 and25%capturegain; pred_d rank16partners>=90%energy.
Existing exact solver, new producer-folded object. No circuit claim.
"""
import os,sys,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from fullu_input_blocks_v1 import sandwich
from shared_input_factor_v1 import optimize,native_partner
STEM='COMPOSED_SHARED_PARENT_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,seeds=4,max_steps=4000)));return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not artifact.exists()
    signal.alarm(900);start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    u=state['lm_head.weight'].double().cuda();u-=u.mean(0);root=torch.linalg.cholesky(u.T@u)
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(state['transformer.h.17.lambdas'][0]);h=scale**2*d0@atom_gram(l0,r0)@d0.T
    he,hv=torch.linalg.eigh(h);assert float(he.min())>0
    hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T;l,r=l1@hs,r1@hs
    g=d1.T@(u.T@u)@d1;eye=torch.eye(1152,device='cuda');k=sandwich(l,r,g,eye);k=(k+k.T)/2
    ke,kv=torch.linalg.eigh(k);total=float(torch.trace(k));bound=2*float(ke[-1])/total
    old=json.loads((P/'SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json').read_text())['objectives']['centered']
    full=json.loads((P/'FULLU_PAIRED_PRODUCER_V1_RESULT.json').read_text())['reports'][-1]
    errors=[abs(total/full['total_paired_coefficient_energy']-1)]
    fits=[];readers=[]
    for seed in (None,1103,1109,1117):
        if seed is None:initial=kv[:,-1]
        else:initial=torch.randn(1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
        a,fit=optimize(initial,l,r,g,k,max_steps=4000,tolerance=1e-8)
        readers.append(a);fits.append(dict(seed=seed,**fit));print(json.dumps({k:v for k,v in fits[-1].items() if k!='history'}),flush=True)
    best=max(range(4),key=lambda i:fits[i]['capture']);a=readers[best];partner=native_partner(a,l,r,d1)
    j=eye+(2**.5-1)*a[:,None]*a[None,:];weighted=root.T@partner@j
    errors.append(abs(float(weighted.square().sum()/2/total)-fits[best]['capture']))
    singular=torch.linalg.svdvals(weighted);energy=singular.square();rank16=float(energy[:16].sum()/energy.sum())
    physical_reader=hi@a;physical_partner=partner@hi
    x=torch.randn(64,1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(120447))
    producer=((x@l0.T)*(x@r0.T))@d0.T*scale;z=producer@hi
    native=lambda xx:((xx@l.T)*(xx@r.T))@d1.T
    star=(z@a)[:,None]*(z@partner.T)
    difference=native(z)-native(z-(z@a)[:,None]*a)
    physical=(producer@physical_reader)[:,None]*(producer@physical_partner.T)
    errors.extend([float((difference-star).norm()/star.norm()),float((physical-star).norm()/star.norm())])
    cosines=[float(abs(a@v)) for v in readers];valid=max(errors)<=1e-8 and all(torch.isfinite(v).all().item() for v in (physical,partner,a))
    torch.save(dict(readers=torch.stack(readers).cpu(),best=best,physical_reader=physical_reader.cpu(),physical_partner=physical_partner.cpu(),producer_scale=scale),artifact)
    result={'pred_a':valid,'pred_b':all(f['converged'] for f in fits),'pred_c':min(cosines)>=.99 and fits[best]['capture']>=1.25*old['best_capture'],'pred_d':rank16>=.9}
    result.update(fits=fits,best=best,best_capture=fits[best]['capture'],prior_capture=old['best_capture'],one_parent_upper_bound=bound,cosines=cosines,partner_rank16_capture=rank16,partner_rank90=int(torch.searchsorted(energy.cumsum(0),.9*energy.sum()))+1,identity_errors=errors,execution_seconds=time.perf_counter()-start,source_shas=binding,artifact_sha=digest(artifact),scope='Full-U composed shared-parent coefficient structure; native behavioral validation not yet performed.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('fits','source_shas')}),flush=True)

if __name__=='__main__':main()
