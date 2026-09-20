#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_price pred_c_compact
"""Native centered compact fit; full preregistration CENTERED_COMPACT_PLAN_V1.md.
Arms: metric / linear-rank / quadratic-width / optimizer / seed.
pred_a_replay: FP64 fold vs independent degree expansion <1e-10, finite solves.
pred_b_price: all exports exactly48,384 reduced scalars, shared QR map excluded.
pred_c_compact: a training-selected winner per metric beats .25281581 panel2 error.
Null: none beat the prior quartic; diagnostic panels never select models.
"""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(metrics=['spherical','centered_floor01'],allocations=[[2,12],[8,8],[14,4],[20,0]],optimizers=['adam','muon'],seeds=[0,1],steps=400,lr=.005,native_forwards=0,selection='Within metric maximize linear gain + 2 quadratic gain; free Gaussian mean correction',scalar_budget=48384)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from centered_quadratic_factors import fold,check
 from centered_quartic import degree_terms
 from quadratic_student_fit import fit
 check();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_CENTERED_COMPACT_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True);capture=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];cached=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];d=1152
 def w(key):return state[key].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];teacher=[t.double() for t in teacher];mu=capture[0]['mean'].cuda().double();constant,J,C,A,B=fold(teacher,mu);cov=capture[0]['covariance'].cuda().double();cov=(cov+cov.T)/2;variance=float(cov.trace()/d);ev,V=torch.linalg.eigh(cov);L=(V*ev.clamp_min(.01*variance).sqrt())@V.T;metrics=dict(spherical=torch.eye(d,device='cuda',dtype=torch.float64)*variance**.5,centered_floor01=L);xs=[p['rows'].cuda().double()-mu for p in capture];targets=[t.cuda().double() for t in cached]
  def truncated(x):return constant+x@J.T+((x@A.T)*(x@B.T))@C.T
  taylor=[torch.cat([truncated(z) for z in x.split(128)]) for x in xs];ref=degree_terms(teacher,xs[0][:16]+mu,mu)[:,:3].sum(1);replay=float((truncated(xs[0][:16])-ref).norm()/ref.norm());assert replay<1e-10
  full_ref=degree_terms(teacher,xs[0][:16]+mu,mu).sum(1);cache_replay=float((full_ref-targets[0][:16]).norm()/targets[0][:16].norm());assert cache_replay<1e-4
  del teacher,state;Cf,Af,Bf=C.float(),A.float(),B.float();records=[];students={};metric_info={}
 for metric,transform in metrics.items():
  with torch.no_grad():
   left,singular,right=torch.linalg.svd(J@transform,full_matrices=False);Aw,Bw=A@transform,B@transform;trace_teacher=C@(Aw*Bw).sum(1);metric_info[metric]=dict(trace=float(transform.square().sum()),linear_energy=float(singular.square().sum()),quadratic_mean_norm=float(trace_teacher.norm()))
  for rank,width in PLAN['allocations']:
   with torch.no_grad():
    writer_linear=left[:,:rank]*singular[:rank];reader_linear=torch.linalg.solve(transform.T,right[:rank].T).T;linear_gain=float(singular[:rank].square().sum())
   configs=itertools.product(PLAN['optimizers'],PLAN['seeds']) if width else [('none',0)]
   for optimizer,seed in configs:
    if width:receipt,(a,b,c)=fit(Cf,Af,Bf,width,transform.float(),optimizer,PLAN['lr'],seed,PLAN['steps'])
    else:receipt=dict(gain=0.,selected_step=0,initial_gain=0.);a=b=torch.empty(0,d,device='cuda');c=torch.empty(d,0,device='cuda',dtype=torch.float64)
    with torch.no_grad():
     a,b,c=a.double(),b.double(),c.double();trace_student=c@((a@transform)*(b@transform)).sum(1);corrected=constant+trace_teacher-trace_student;model=dict(mu=mu.float().cpu(),constant=corrected.float().cpu(),linear_writer=writer_linear.float().cpu(),linear_reader=reader_linear.float().cpu(),quadratic_writer=c.float().cpu(),quadratic_left=a.float().cpu(),quadratic_right=b.float().cpu());price=sum(t.numel() for t in model.values());assert price==48384;errors=[];taylor_errors=[];uncorrected_errors=[];export_errors=[]
     for x,y,t in zip(xs,targets,taylor):
      variable=(x@reader_linear.T)@writer_linear.T+((x@a.T)*(x@b.T))@c.T;prediction=variable+corrected;errors.append(float((prediction-y).norm()/y.norm()));taylor_errors.append(float((prediction-t).norm()/t.norm()));uncorrected_errors.append(float((variable+constant-y).norm()/y.norm()))
      # Replay archived FP32 parameter program in FP64, including its stored center.
      delta=x+mu-model['mu'].cuda().double();replayed=model['constant'].cuda().double()+(delta@model['linear_reader'].cuda().double().T)@model['linear_writer'].cuda().double().T+((delta@model['quadratic_left'].cuda().double().T)*(delta@model['quadratic_right'].cuda().double().T))@model['quadratic_writer'].cuda().double().T;export_errors.append(float((replayed-prediction).norm()/prediction.norm()))
     row=dict(metric=metric,linear_rank=rank,quadratic_width=width,optimizer=optimizer,seed=seed,linear_gain=linear_gain,quadratic_gain=receipt['gain'],selection_score=linear_gain+2*receipt['gain'],selected_step=receipt['selected_step'],scalar_count=price,products=width,full_quartic_errors=errors,taylor_errors=taylor_errors,uncorrected_full_quartic_errors=uncorrected_errors,export_replay=max(export_errors));records.append(row);students[(metric,rank,width,optimizer,seed)]=model;print(json.dumps(row),flush=True);out.write_text(json.dumps(dict(plan=PLAN,records=records,complete=False),indent=2)+'\n')
 winners=[max((r for r in records if r['metric']==metric),key=lambda r:r['selection_score']) for metric in metrics];predictions=dict(pred_a_replay=replay<1e-10 and cache_replay<1e-4 and all(math.isfinite(r['selection_score']) and r['export_replay']<1e-4 for r in records),pred_b_price=all(r['scalar_count']==48384 for r in records),pred_c_compact=any(r['full_quartic_errors'][1]<.25281581 for r in winners));guard_torch_save(dict(students=students,teacher_scale=scale,plan=PLAN),str(P/'NATIVE_CENTERED_COMPACT_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,winners=winners,metric_info=metric_info,fold_replay=replay,cached_teacher_replay=cache_replay,predictions=predictions,complete=True,seconds=time.perf_counter()-start,scope='Fixed calibration mean; weight-derived Taylor-degree<=2 approximation compressed at matched48,384 FP32 scalars. Free Gaussian mean correction; no empirical target fitting. Same reused panels; no fullmodel/OOD/causal claim.'),indent=2)+'\n')
if __name__=='__main__':main()
