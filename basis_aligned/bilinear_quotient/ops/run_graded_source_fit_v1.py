#!/usr/bin/env python3
# BQGATE:0bodyforwards;2starts540sec each;rank128;1320secwatchdog.
"""pred_a both locallyconverged grad<=1e-5/orth<=1e-10; pred_b loss improves10%;
pred_c botharms bothbranches swap<=.1/sign>=.9/CE<=.02. Null: no faithful shared interface.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from graded_source_projection_v1 import graded_norms,balanced_loss,retained_grades
from coupled_source_projection_v1 import tangent,retract
from shared_producer_interface_v1 import apply
from packed_quadratic_branch_v1 import execute
from quartic_frozen_native_score_v2 import score
STEM='GRADED_SOURCE_FIT_V1'

def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    receipt=json.loads((P/'GRADED_SOURCE_PROJECTION_NATIVE_V1_RESULT.json').read_text());assert receipt['pred_a'] and receipt['pred_b'] and receipt['pred_c']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('2 starts, 540 seconds each, 0 body forwards');return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(1320)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True,map_location='cpu');a=data['forms'].cuda();rawroot=data['root'].cuda();wg=data['writer_gram'].cuda();root=rawroot/rawroot.square().sum().div(len(rawroot)).sqrt();full=graded_norms(a,root@root.T,wg)
    gen=torch.Generator(device='cuda').manual_seed(73142)
    starts=[data['starts']['128'].cuda(),torch.linalg.qr(torch.randn(1152,128,device='cuda',dtype=torch.float64,generator=gen)).Q]
    fits=[];frames=[];progress=P/(STEM+'_PROGRESS.json')
    for arm,initial in enumerate(starts):
        p=initial.detach();step=4.;start=time.perf_counter();history=[];status='iteration_limit';monotone=True;initial_loss=None
        for iteration in range(10000):
            p=p.detach().requires_grad_();loss=balanced_loss(a,root,p,wg,full);tg=tangent(p,torch.autograd.grad(loss,p)[0]);lv=float(loss.detach());gn=float(tg.norm().detach())
            if initial_loss is None:initial_loss=lv
            if iteration%50==0:
                history.append(dict(iteration=iteration,loss=lv,gradient=gn,seconds=time.perf_counter()-start));progress.write_text(json.dumps(dict(arm=arm,history=history,completed=fits),indent=2)+'\n')
                print(json.dumps(dict(arm=arm,iteration=iteration,loss=lv,gradient=gn)),flush=True)
            if gn<=1e-5:status='converged';break
            if time.perf_counter()-start>=540:status='time_limit';break
            with torch.no_grad():
                accepted=False
                for _ in range(20):
                    proposal=retract(p,-step*tg);new=float(balanced_loss(a,root,proposal,wg,full))
                    if new<=lv-1e-4*step*gn*gn:accepted=True;break
                    step/=2
                if not accepted:status='line_search_limit';break
                monotone=monotone and new<=lv+1e-10;p=proposal;step=min(100.,1.5*step)
        # Always recompute terminal gradient, including iteration-limit exit.
        p=p.detach().requires_grad_();loss=balanced_loss(a,root,p,wg,full);tg=tangent(p,torch.autograd.grad(loss,p)[0]);gn=float(tg.norm().detach())
        if gn<=1e-5:status='converged'
        orth=float((p.T@p-torch.eye(128,device='cuda')).norm().detach())
        frames.append(p.detach());fits.append(dict(arm=arm,status=status,iterations=iteration,initial_loss=initial_loss,loss=float(loss.detach()),gradient=gn,orthogonality=orth,monotone=monotone,seconds=time.perf_counter()-start,history=history))
    # Frames are frozen before loading any native validation endpoint.
    state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    bias=(state['transformer.h.16.lambdas'][0].double()*state['transformer.h.15.mlp.Down_bias'].double()).cuda();inverse=torch.linalg.inv(rawroot)
    programs=[dict(read=(p.T@inverse).float(),write=(rawroot@p).float()) for p in frames]
    source=torch.load(P/'MATCHED_PARTNER_MLP15_SOURCE_V1_PORTS.pt',weights_only=True,map_location='cpu')['sources'];b=source['background'].double().cuda();m=source['producer'].double().cuda()
    n16=(b+m).square().mean(-1)+torch.finfo(torch.float32).eps
    cache=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True,map_location='cpu');ports=cache['ports'];n17=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    packed=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True)
    writes=[execute(packed,(b+apply({k:v.double() for k,v in prog.items()},m,bias))/n16.sqrt()[:,None],n17) for prog in programs]
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows'];h=ports['pre']+ports['native_output'];effects={};passed=[]
    for branch in (3,8):
        e=score([cache['writes'][str(branch)],writes[0][branch].cpu(),writes[1][branch].cpu()],h,rows,state['lm_head.weight'].float());effects[str(branch)]=e
        for report in e['reports']:
            passed.extend(f['swap_relative_rms']<=.1 and f['swap_sign_agreement']>=.9 and f['swap_live']>=4 and f['zero_ce_meanabs_disagreement']<=.02 for f in report['families'] if f['family']!='quoted_control')
    torch.save(dict(frames=[p.cpu() for p in frames],programs=[{k:v.cpu() for k,v in prog.items()} for prog in programs],bias=bias.float().cpu()),ap)
    result={'pred_a':all(f['status']=='converged' and f['orthogonality']<=1e-10 and f['monotone'] for f in fits),'pred_b':fits[0]['loss']<=.9*fits[0]['initial_loss'],'pred_c':all(passed)}
    result.update(fits=fits,effects=effects,artifact_sha=digest(ap),source_shas=binding,execution_seconds=time.perf_counter()-tic)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('effects','source_shas','fits')}),flush=True)
if __name__=='__main__':main()
