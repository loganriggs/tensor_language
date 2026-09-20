"""Fit f_v(x)=q(x)*r_v(x), with one shared quadratic and exact linear quotient solves."""
import itertools,json,math,time
from pathlib import Path
import torch
from core import terms,metric,multiply,inner,coefficients_from_dense
from sweep import target_cases
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);out=P/'COMMON_FACTOR_SWEEP_V1.json';assert not out.exists();start=time.perf_counter();cases=[dict(name='planted_shared',d=6,target=next(x['target'] for x in target_cases() if x['name']=='shared_quartic_dag'))]
 native=torch.load(P.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',weights_only=True)
 for i,c in enumerate(native):
  t=coefficients_from_dense(c['hessian_not_applicable_quartic'][0].double());t/=inner(t,t,metric(5,4)).sqrt();cases.append(dict(name=f'native_{i}',d=5,target=t))
 records=[]
 for case in cases:
  d=case['d'];target=case['target'];M2=metric(d,2);M4=metric(d,4);F=metric(d,4,'frobenius');chol=torch.linalg.cholesky(M4);n=len(terms(d,2));eye=torch.eye(n,dtype=torch.float64);rhs=(target@chol).T
  for optname,rate,seed in itertools.product(['adam','muon'],[.01,.05],[0,1]):
   torch.manual_seed(seed);q=torch.nn.Parameter(torch.randn(1,n,dtype=torch.float64));opt=torch.optim.Adam([q],lr=rate) if optname=='adam' else torch.optim.Muon([q],lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');saved=None;history=[]
   for step in range(600):
    opt.zero_grad();qn=q/inner(q,q,M2).sqrt();features=multiply(qn,eye,d,2,2);design=(features@chol).T;K=design.T@design;cross=design.T@rhs;quot=torch.linalg.solve(K+1e-10*K.diag().mean()*eye,cross);error=(design@quot-rhs).square().sum()/rhs.square().sum();value=float(error.detach())
    if value<best:best=value;saved=(qn.detach().clone(),quot.T.detach().clone())
    if step%100==0 or step==599:history.append(dict(step=step,relative_squared_error=value))
    if value<1e-12:break
    error.backward();opt.step()
    for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/600)))
   with torch.no_grad():
    qn,quot=saved;coeff=multiply(qn,quot,d,2,2);diff=coeff-target;frob=float((inner(diff,diff,F)/inner(target,target,F)).sqrt())
   row=dict(case=case['name'],optimizer=optname,lr=rate,seed=seed,gaussian_relative_error=math.sqrt(best),frobenius_relative_error=frob,parameter_values=n*(1+target.shape[0]),common_quadratic=qn.tolist(),quotients=quot.tolist(),history=history);records.append(row)
  print(case['name'],min(r['gaussian_relative_error'] for r in records if r['case']==case['name']),flush=True)
  out.write_text(json.dumps(dict(records=records,seconds=time.perf_counter()-start,scope='One shared quadratic factor times output-specific quadratics; coefficients only; native contexts independently fitted.'),indent=2)+'\n')
 result=json.load(open(out));result['predictions']=dict(pred_a_toy=min(r['gaussian_relative_error'] for r in records if r['case']=='planted_shared')<1e-3,pred_b_native=sum(min(r['gaussian_relative_error'] for r in records if r['case']==f'native_{i}')<.1 for i in range(16))>=8,pred_c_finite=all(math.isfinite(r['gaussian_relative_error']) for r in records));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
