#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_empirical pred_c_tradeoff
"""Native weighted banks: pred_a_integrity pred_b_empirical pred_c_tradeoff."""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(metrics=['isotropic','centered_floor01','centered_floor10','second_floor01'],optimizers=['adam','muon'],seeds=[0,1],lr=.005,steps=400,energy_probes=4096,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from shared_quadratic_bank import normalize_bank,bank_gram,native_bank_cross,bank_entries,roots
 from quartic_cp import directional
 from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_WEIGHTED_BANK_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True);capture=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True);prior={(r['optimizer'],r['seed']):r for r in json.load(open(P/'NATIVE_LEARNED_SHARED_BANK_V1.json'))['records'] if r['lr']==.005}
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];d=1152;i,j=roots(4,'cuda');gen=torch.Generator(device='cuda');gen.manual_seed(1651);idx=torch.randint(d,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in idx.split(512)]);gaussian=torch.randn(256,d,device='cuda',generator=gen);eval_rows=[gaussian]+[p['rows'].cuda().float() for p in capture['panels']]
  def evaluate_teacher(x):
   C,L,R,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;return ((h@L.T)*(h@R.T))@C.T
  eval_targets=[evaluate_teacher(x) for x in eval_rows];metrics={'isotropic':None};metric_summary={}
  for name,key,floor in [('centered_floor01','covariance',.01),('centered_floor10','covariance',.1),('second_floor01','second_moment',.01)]:
   M=capture['panels'][0][key].cuda().double();M=(M+M.T)/2;M=M/(M.trace()/d);ev,V=torch.linalg.eigh(M);clamped=ev.clamp_min(floor);L=(V*clamped.sqrt())@V.T;metrics[name]=L.float();metric_summary[name]=dict(floor=floor,floored_eigenvalues=int((ev<floor).sum()),original_minimum=float(ev.min()),original_maximum=float(ev.max()),post_floor_trace=float(clamped.sum()),condition=float(clamped.max()/clamped.min()))
 records=[];students={};norms={}
 for metric,transform in metrics.items():
  with torch.no_grad():
   t=teacher if transform is None else [*teacher[:-2],teacher[-2]@transform,teacher[-1]@transform];g=torch.Generator(device='cuda');g.manual_seed(1840);samples=[]
   for _ in range(32):
    vectors=[torch.randint(2,(128,d),device='cuda',generator=g).float()*2-1 for _ in range(4)];samples.append(directional(*t,vectors).double().square().sum(1))
   samples=torch.cat(samples);norms[metric]=dict(energy=float(samples.mean()),standard_error=float(samples.std()/len(samples)**.5))
  for optimizer,seed in itertools.product(PLAN['optimizers'],PLAN['seeds']):
   torch.manual_seed(seed);us=[torch.nn.Parameter(torch.randn(4,d,device='cuda')/d**.5) for _ in range(4)];vs=[torch.nn.Parameter(torch.randn(4,d,device='cuda')/d**.5) for _ in range(4)];opt=torch.optim.Adam(us+vs,lr=.005) if optimizer=='adam' else torch.optim.Muon(us+vs,lr=.005,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=-float('inf');base=None
   for step in range(401):
    opt.zero_grad();U,V=normalize_bank(torch.stack(us),torch.stack(vs));u,v=(U,V) if transform is None else (U@transform,V@transform);G=bank_gram(u,v).double();cross=native_bank_cross(t,u,v).double();C=torch.linalg.solve(G+1e-6*G.diag().mean()*torch.eye(len(G),device='cuda',dtype=torch.float64),cross.T).T;loss=((C@G)*C).sum()-2*(cross*C).sum();gain=float(-loss.detach());assert math.isfinite(gain)
    if gain>best:best=gain;beststep=step;bu=U.detach().clone();bv=V.detach().clone();bc=C.detach().float()
    if base is None:base=max(gain,1e-30)
    if step==400:break
    (loss/base).backward();opt.step()
    for group in opt.param_groups:group['lr']=.005*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/400)))
   with torch.no_grad():
    coeff=torch.cat([bank_entries(bu,bv,z)@bc.T for z in idx.split(512)]);errors=[]
    for x,y in zip(eval_rows,eval_targets):
     q=((x@bu.flatten(0,1).T)*(x@bv.flatten(0,1).T)).reshape(len(x),4,4).sum(2);prediction=(q[:,i]*q[:,j])@bc.T;errors.append(float((prediction-y).norm()/y.norm()))
    row=dict(metric=metric,optimizer=optimizer,seed=seed,selected_step=beststep,weighted_gain=best,weighted_explained_fraction_estimate=best/norms[metric]['energy'],coefficient_error=float((coeff-target).norm()/target.norm()),gaussian_error=errors[0],empirical_calibration_error=errors[1],empirical_evaluation_error=errors[2]);records.append(row);students[(metric,optimizer,seed)]=dict(U=bu.cpu(),V=bv.cpu(),C=bc.cpu());print(json.dumps(row),flush=True);out.write_text(json.dumps(dict(plan=PLAN,records=records,norms=norms,metric_summary=metric_summary,complete=False),indent=2)+'\n')
 controls={(r['optimizer'],r['seed']):r for r in records if r['metric']=='isotropic'};improved=[r for r in records if r['metric']!='isotropic' and r['empirical_evaluation_error']<.95*controls[(r['optimizer'],r['seed'])]['empirical_evaluation_error']];pairs={(r['optimizer'],r['seed']) for r in improved};integrity=all(abs(r['weighted_gain']/prior[key]['explained_fraction_using_estimated_teacher_norm']-1)<1e-3 for key,r in controls.items()) and all(math.isfinite(r[k]) for r in records for k in ['weighted_gain','coefficient_error','gaussian_error','empirical_calibration_error','empirical_evaluation_error']);predictions=dict(pred_a_integrity=integrity,pred_b_empirical=len(pairs)>=2,pred_c_tradeoff=any(r['coefficient_error']>controls[(r['optimizer'],r['seed'])]['coefficient_error'] for r in improved));guard_torch_save(dict(students=students,metric_transforms={k:None if v is None else v.cpu() for k,v in metrics.items()},teacher_scale=scale),str(P/'NATIVE_WEIGHTED_BANK_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,norms=norms,metric_summary=metric_summary,complete=True,predictions=predictions,seconds=time.perf_counter()-start,scope='Original-coordinate optimization under isotropic or four-slot covariance coefficient metrics; estimatedteacher norms. Empirical diagnostics evaluate purequartic only, not fullmodel. No circuitidentification.'),indent=2)+'\n')
if __name__=='__main__':main()
