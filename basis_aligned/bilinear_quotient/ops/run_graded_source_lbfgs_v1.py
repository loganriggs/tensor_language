#!/usr/bin/env python3
# BQGATE:0bodyforwards;2continuations300sec each;rank128;780secwatchdog.
"""pred_a both locallyconverged grad<=1e-5/orth<=1e-10; pred_b loss improves10%;
pred_c botharms bothbranches swap<=.1/sign>=.9/CE<=.02. Null: no faithful shared interface.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from graded_source_projection_v1 import graded_norms,balanced_loss,retained_grades
from quartic_manifold_lbfgs_v1 import fit as manifold_fit
from quartic_manifold_cg_v1 import tangent as manifold_tangent
from shared_producer_interface_v1 import apply
from packed_quadratic_branch_v1 import execute
from quartic_frozen_native_score_v2 import score
STEM='GRADED_SOURCE_LBFGS_V1'

def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    receipt=json.loads((P/'GRADED_SOURCE_PROJECTION_NATIVE_V1_RESULT.json').read_text());assert receipt['pred_a'] and receipt['pred_b'] and receipt['pred_c']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('2 continuations, 300 seconds each, 0 body forwards');return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(780)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True,map_location='cpu');a=data['forms'].cuda();rawroot=data['root'].cuda();wg=data['writer_gram'].cuda();root=rawroot/rawroot.square().sum().div(len(rawroot)).sqrt();full=graded_norms(a,root@root.T,wg)
    prior=json.loads((P/'GRADED_SOURCE_FIT_V1_RESULT.json').read_text())
    original_path=P/'GRADED_SOURCE_FIT_V1_PROGRAM.pt';assert digest(original_path)==prior['artifact_sha']
    fit_source=ROOT/'basis_aligned/bilinear_quotient/ops/run_graded_source_fit_v1.py'
    assert prior['source_shas'][str(fit_source)]=='aa056891e946d5ce09e79fa9ce5e4905a7bea7b5123eb97f9f56953d258ca273'
    starts=torch.load(original_path,weights_only=True,map_location='cpu')['frames']
    fits=[];frames=[];progress=P/(STEM+'_PROGRESS.json')
    def evaluate(b,n,divisor,gradient=False):
        b=b.detach().requires_grad_(gradient)
        with torch.set_grad_enabled(gradient):
            loss=balanced_loss(a,root,b[0],wg,full)
            if gradient:
                gb=torch.autograd.grad(loss,b)[0];gb,gn=manifold_tangent(b,n,gb,torch.zeros_like(n))
                return float(loss.detach()),None,gb.detach(),gn.detach()
            return float(loss),None
    for arm,initial in enumerate(starts):
        begin=time.perf_counter();initial=initial.cuda()[None,:,:];n=torch.ones(1,1,device='cuda',dtype=initial.dtype)
        trace=[]
        def callback(row,b,n,mix):
            if row['iteration']%25==0:
                trace.append(row.copy());progress.write_text(json.dumps(dict(arm=arm,history=trace,completed=fits),indent=2)+'\n');print(json.dumps(dict(arm=arm,**row)),flush=True)
        b,n,_,history,reason=manifold_fit(initial,n,evaluate,1.,max_steps=2000,max_seconds=300,callback=callback)
        last=history[-1];p=b[0];gn=last['projected_gradient_norm'];orth=float((p.T@p-torch.eye(128,device='cuda')).norm())
        frames.append(p.detach());fits.append(dict(arm=arm,status='converged' if gn<=1e-5 else reason,optimizer_stop=reason,iterations=len(history),
            initial_loss=prior['fits'][arm]['initial_loss'],continuation_initial_loss=history[0]['objective'],loss=last['objective'],gradient=gn,orthogonality=orth,
            monotone=all(history[i+1]['objective']<=history[i]['objective']+1e-10 for i in range(len(history)-1)),seconds=time.perf_counter()-begin,history=history))
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
