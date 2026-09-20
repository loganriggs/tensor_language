"""Replicate sparse shared-quartic bases over all declared native contexts."""
import itertools,json,time,math
from pathlib import Path
import torch
from core import coefficients_from_dense,metric,inner
from sweep import fit
from quartic_sparse_basis import gauge,atoms,refits

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;out=p/'QUARTIC_CONTEXT_BASIS_V1.json';assert not out.exists();start=time.perf_counter()
 cases=torch.load(p.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',weights_only=True);old=json.load(open(p/'NATIVE_QUARTIC_CONTEXTS_V1.json'))['records'];records=[];budgets=[2,4,6,8,10];M=metric(5,4)
 for index,case in enumerate(cases):
  target=coefficients_from_dense(case['hessian_not_applicable_quartic'][0].double());target/=inner(target,target,M).sqrt()
  selected=min([r for r in old if r['case_index']==index and r['kind']=='dag' and r['width']==4 and r['metric']=='frobenius'],key=lambda r:r['relative_error'])
  receipt,state=fit(target,5,4,'dag',4,'muon',.05,selected['seed'],1200,metric_kind='frobenius')
  replay=abs(receipt['relative_error']-selected['relative_error']);assert replay<1e-8
  bank,root=gauge(state,torch.zeros(4,4,dtype=torch.float64),5);reference=root@atoms(bank,5)
  baseline=refits(bank,root,target,5,budgets);runs=[]
  for seed in [0,1]:
   torch.manual_seed(seed);K=torch.nn.Parameter(torch.randn(4,4,dtype=torch.float64)*.05);opt=torch.optim.Muon([K],lr=.05,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');saved=None
   for step in range(1500):
    opt.zero_grad();bank,root=gauge(state,K,5);loss=root.norm(dim=0).sum();value=float(loss.detach())
    if value<best:best=value;saved=K.detach().clone()
    loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/1500)))
   with torch.no_grad():
    bank,root=gauge(state,saved,5);delta=root@atoms(bank,5)-reference;preserve=float((inner(delta,delta,M)/inner(reference,reference,M)).sqrt());assert preserve<1e-8
    runs.append(dict(seed=seed,preservation_error=preserve,condition=float(torch.linalg.cond(torch.matrix_exp(saved))),bank=bank.tolist(),root=root.tolist(),refits=refits(bank,root,target,5,budgets)))
  records.append(dict(case_index=index,context={k:case[k] for k in ['panel','role','template']},starting_seed=selected['seed'],starting_replay=replay,baseline=baseline,runs=runs))
  result=dict(records=records,seconds=time.perf_counter()-start,scope='Independent per-context banks,32 basis searches over16 groups; no cross-context shared circuit claim')
  out.write_text(json.dumps(result,indent=2)+'\n');print(index,[(b,min(r['refits'][i]['error'] for r in runs),baseline[i]['error']) for i,b in enumerate(budgets)],flush=True)
 gains=[r['baseline'][1]['error']/min(x['refits'][1]['error'] for x in r['runs']) for r in records]
 result['predictions']=dict(pred_a_reconstruction=all(r['starting_replay']<1e-8 for r in records),pred_b_preservation=all(x['preservation_error']<1e-8 for r in records for x in r['runs']),pred_c_general_gain=sum(g>=2 for g in gains)>=12,pred_d_fixed_span=all(abs(x['refits'][-1]['error']-r['baseline'][-1]['error'])<1e-8 for r in records for x in r['runs']))
 result['four_product_improvement_factors']=gains;out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
