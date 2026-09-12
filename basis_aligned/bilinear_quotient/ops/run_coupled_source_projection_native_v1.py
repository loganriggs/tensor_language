#!/usr/bin/env python3
# BQGATE:0bodyforwards;nativeweightobjective128/512;300sec;50MBartifact.
"""pred_a tangentFD<=1e-5/fullrank<=1e-8; pred_b one step improves>=1e-6;
pred_c evaluation<=2sec/peak<=10GiB. Null: unusable coupled objective.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(P),str(ROOT)]
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from attention_source_quartic_v1 import unpack
from shared_producer_interface_v1 import geometry
from coupled_source_projection_v1 import norm,retained,tangent,retract
STEM='COUPLED_SOURCE_PROJECTION_NATIVE_V1'

def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,ranks=[128,512],limit_seconds=300)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_INPUTS.pt');assert not out.exists() and not ap.exists();signal.alarm(300)
    tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def weight(key):return state[key].double().cuda()
    l=weight('transformer.h.15.mlp.Left.weight');r=weight('transformer.h.15.mlp.Right.weight');d=weight('transformer.h.15.mlp.Down.weight');scale=weight('transformer.h.16.lambdas')[0]
    g=atom_gram(l,r);h=scale.square()*(d@g@d.T);del g,l,r,d
    packed=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True);forms=unpack(packed,h);geo=geometry(forms,h);root=geo['root']
    writers=packed['writers'].double().cuda();wg=torch.zeros(2,2,device='cuda',dtype=torch.float64)
    for start in range(0,len(state['lm_head.weight']),4096):
        uw=state['lm_head.weight'][start:start+4096].double().cuda()@writers;wg+=uw.T@uw
    eye=torch.eye(1152,device='cuda',dtype=torch.float64);full=norm(forms,eye+h,wg);background=norm(forms,eye,wg);energy=full-background;assert energy>0
    full_error=float((retained(forms,root,eye,wg)-full).abs()/energy)
    records=[];times=[];starts={};first_step=None
    for rank in [128,512]:
        p=geo['vectors'][:,:rank].clone().detach().requires_grad_();starts[str(rank)]=p.detach().cpu()
        torch.cuda.synchronize();clock=time.perf_counter();loss=(full-retained(forms,root,p,wg))/energy;grad=torch.autograd.grad(loss,p)[0];tg=tangent(p,grad);torch.cuda.synchronize();times.append(time.perf_counter()-clock)
        direction=-tg/tg.norm();eps=1e-4
        with torch.no_grad():
            def value(q):return (full-retained(forms,root,q,wg))/energy
            fd=(value(retract(p,eps*direction))-value(retract(p,-eps*direction)))/(2*eps);analytic=(tg*direction).sum();error=float((fd-analytic).abs()/analytic.abs().clamp_min(1e-30))
            step=.1;new=float(loss)
            for backtrack in range(12):
                proposal=retract(p,step*direction);new=float(value(proposal))
                if new<float(loss):break
                step/=2
            records.append(dict(rank=rank,loss=float(loss),tangent_norm=float(tg.norm()),finite_difference_error=error,one_step_loss=new,step=step,improvement=float(loss)-new))
            if rank==128:first_step=proposal.cpu()
    torch.save(dict(forms=forms.cpu(),root=root.cpu(),writer_gram=wg.cpu(),starts=starts,first_step128=first_step,full=float(full),background=float(background),producer_energy=float(energy)),ap)
    peak=torch.cuda.max_memory_allocated()/2**30
    result={'pred_a':full_error<=1e-8 and max(x['finite_difference_error'] for x in records)<=1e-5,'pred_b':records[0]['improvement']>=1e-6,'pred_c':sum(times)/len(times)<=2 and peak<=10}
    result.update(records=records,fullrank_error=full_error,evaluation_seconds=times,peak_gib=peak,full=float(full),background=float(background),producer_energy=float(energy),artifact_sha=digest(ap),source_shas=binding,execution_seconds=time.perf_counter()-tic)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
