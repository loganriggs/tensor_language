#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_improves pred_c_small_error
"""Direct full joint tensor random-student optimization, no activation fitting."""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(widths=[128,512,1024],optimizers=['muon'],seeds=[0,1],steps=600,lr=.05,metrics=['gaussian','frobenius'],learning_rates=[.005,.05],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_write,guard_torch_save
 sys.path.insert(0,str(P));from implicit_quadratic import inner
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'NATIVE_FULL_QUADRATIC_V2.json';assert not out.exists();start=time.perf_counter()
 snap=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240');state=torch.load(snap/'pytorch_model.bin',map_location='cpu',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  U=w('lm_head.weight');D=w('transformer.h.17.mlp.Down.weight');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight')
  E=torch.cat([torch.eye(1152,device='cuda'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.17.attn.c_proj.weight')],1)
  qu,ru=torch.linalg.qr(U);qe,re=torch.linalg.qr(E.T);c,a,b=ru@D,L@re.T,R@re.T
  torch.manual_seed(711);z=torch.randn(8,E.shape[1],device='cuda');h=z@E.T;native=((h@L.T)*(h@R.T))@D.T@U.T;reduced=(((z@qe@a.T)*(z@qe@b.T))@c.T)@qu.T
  replay=float((native-reduced).norm()/native.norm());scale=inner(c,a,b,c,a,b).sqrt();c=c/scale
  teacher_g=float(inner(c,a,b,c,a,b));teacher_f=float(inner(c,a,b,c,a,b,False))
 records=[];best={}
 trace_square=float((c@(a*b).sum(-1)).square().sum())
 radial=dict(gaussian_relative_error=math.sqrt(max(0,1-(1+2/1152)*trace_square/teacher_g)),frobenius_relative_error=math.sqrt(max(0,1-trace_square/1152/teacher_f)),scope='Isotropic quadratic in exact reduced input frame; frame cost not free')
 for width in PLAN['widths']:
  for objective,rate in itertools.product(PLAN['metrics'],PLAN['learning_rates']):
   optimizer='muon';use_gaussian=objective=='gaussian';den=teacher_g if use_gaussian else teacher_f
   for seed in PLAN['seeds']:
    torch.manual_seed(seed);d=torch.nn.Parameter(torch.randn(1152,width,device='cuda')*.1/math.sqrt(1152*width));l=torch.nn.Parameter(torch.randn(width,1152,device='cuda')*.3);r=torch.nn.Parameter(torch.randn(width,1152,device='cuda')*.3)
    opt=torch.optim.Adam([d,l,r],lr=rate) if optimizer=='adam' else torch.optim.Muon([d,l,r],lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
    history=[];valuebest=float('inf');saved=None;t0=time.perf_counter()
    for step in range(PLAN['steps']):
     opt.zero_grad();ll=l/l.norm(dim=1,keepdim=True);rr=r/r.norm(dim=1,keepdim=True)
     loss=(den+inner(d,ll,rr,d,ll,rr,use_gaussian)-2*inner(c,a,b,d,ll,rr,use_gaussian))/den
     if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite native objective')
     val=float(loss.detach())
     if val<valuebest:valuebest=val;saved=(d.detach().clone(),ll.detach().clone(),rr.detach().clone())
     if step%100==0 or step==PLAN['steps']-1:history.append(dict(step=step,loss=val))
     loss.backward();opt.step()
     lr=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/PLAN['steps'])))
     for group in opt.param_groups:group['lr']=lr
    with torch.no_grad():
     dd,ll,rr=saved;f=(teacher_f+float(inner(dd,ll,rr,dd,ll,rr,False))-2*float(inner(c,a,b,dd,ll,rr,False)))/teacher_f
    g=(teacher_g+float(inner(dd,ll,rr,dd,ll,rr))-2*float(inner(c,a,b,dd,ll,rr)))/teacher_g
    row=dict(width=width,optimizer=optimizer,objective=objective,lr=rate,seed=seed,gaussian_relative_error=math.sqrt(max(0,g)),frobenius_relative_error=math.sqrt(max(0,f)),reduced_parameter_values=3*1152*width,expanded_parameter_values=width*(50304+2*6912),seconds=time.perf_counter()-t0,history=history);records.append(row)
    key=f'{width}_{objective}_{rate}'
    if key not in best or valuebest<best[key]['loss']:best[key]=dict(loss=valuebest,factors=[v.cpu() for v in saved])
    result=dict(plan=PLAN,radial_baseline=radial,teacher_gaussian_energy=teacher_g,teacher_frobenius_energy=teacher_f,replay=replay,records=records,seconds=time.perf_counter()-start,scope='Full lastMLP unnormalized joint numerator, exactQR frames; no native activation optimization or circuit claim')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
 result['predictions']=dict(pred_a_replay=replay<2e-4,pred_b_improves=all(x['history'][-1]['loss']<x['history'][0]['loss'] for x in records),pred_c_small_error=any(x['gaussian_relative_error']<.1 for x in records))
 guard_torch_save(dict(best=best,output_frame=qu.cpu(),input_frame=qe.cpu(),teacher_scale=scale.cpu()),str(P/'NATIVE_FULL_QUADRATIC_V2.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
