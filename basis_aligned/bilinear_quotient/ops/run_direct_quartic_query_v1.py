#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_precision pred_c_estimator
"""Native full-input symmetric quartic queries, no student training.
pred_a_replay <2e-4 relative polynomial error; pred_b_precision<2e-4 relative fp32/fp64;
pred_c_estimator uniform and stratified energy within5 combined estimated SE.
Null: uniform sampling is too noisy for useful full-input fitting.
No native forward; full1152input and exact reduced1152output frame.
"""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(uniform_batches=16,stratum_batches=4,batch_size=512,strata=['4','31','22','211','1111'],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P));from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_QUARTIC_QUERY_V1.json';assert not out.exists();start=time.perf_counter()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));params=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];C,L2,R2,D1,L1,R1=params;d=L1.shape[1];torch.manual_seed(1611)
  coords=torch.randperm(d,device='cuda')[:4];local=torch.tensor(list(itertools.product(range(4),repeat=4)),device='cuda');idx=coords[local];H=entries(*params,idx);amp=torch.randn(2,4,device='cuda');x=torch.zeros(2,d,device='cuda');x[:,coords]=amp;h=((x@L1.T)*(x@R1.T))@D1.T;direct=((h@L2.T)*(h@R2.T))@C.T;contract=amp[:,local].prod(-1)@H;replay=float((direct-contract).norm()/direct.norm())
  fp64=entries(*[p.double() for p in params],idx[:8]);precision=float((H[:8].double()-fp64).norm()/fp64.norm())
  counts={'4':d,'31':4*d*(d-1),'22':3*d*(d-1),'211':6*d*(d-1)*(d-2),'1111':d*(d-1)*(d-2)*(d-3)};assert sum(counts.values())==d**4
  summaries={}
  for kind in ['uniform']+PLAN['strata']:
   n=PLAN['uniform_batches'] if kind=='uniform' else PLAN['stratum_batches'];means=[];maximum=0.;squares=0.;total=0.;num=0
   for batch in range(n):
    ids=torch.randint(d,(PLAN['batch_size'],4),device='cuda')
    if kind!='uniform':
     collision=(ids.sort(dim=1).values.diff(dim=1)==0).any(dim=1)
     while bool(collision.any()):ids[collision]=torch.randint(d,(int(collision.sum()),4),device='cuda');collision=(ids.sort(dim=1).values.diff(dim=1)==0).any(dim=1)
     slots={'4':[0,0,0,0],'31':[0,0,0,1],'22':[0,0,1,1],'211':[0,0,1,2],'1111':[0,1,2,3]}[kind];ids=ids[:,slots]
    v=entries(*params,ids).double().square().sum(-1);means.append(float(v.mean()));maximum=max(maximum,float(v.max()));squares+=float(v.square().sum());total+=float(v.sum());num+=len(v)
   mean=total/num;se=math.sqrt(max(0,(squares-num*mean*mean)/(num-1))/num);count=d**4 if kind=='uniform' else counts[kind]
   summaries[kind]=dict(samples=num,mean_entry_output_energy=mean,standard_error=se,maximum_entry_output_energy=maximum,batch_means=means,ordered_tuple_count=count,estimated_energy=mean*count,estimated_energy_se=se*count)
   print(kind,summaries[kind],flush=True)
  uniform=summaries['uniform']['estimated_energy'];use=summaries['uniform']['estimated_energy_se'];strat=sum(summaries[k]['estimated_energy'] for k in counts);sse=math.sqrt(sum(summaries[k]['estimated_energy_se']**2 for k in counts));gap=abs(uniform-strat)/math.sqrt(use*use+sse*sse)
 result=dict(plan=PLAN,input_dimension=d,output_dimension=C.shape[0],replay=replay,precision=precision,summaries=summaries,stratified_energy=strat,stratified_standard_error=sse,estimator_gap_in_estimated_se=gap,seconds=time.perf_counter()-start,predictions=dict(pred_a_replay=replay<2e-4,pred_b_precision=precision<2e-4,pred_c_estimator=gap<5),scope='Full-input MLP16/17 pure bilinear quartic numerator, exact output QR; no intervening terms/norm polynomialization, no student fitting. Estimated SE does not certify unobserved tails.')
 out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
