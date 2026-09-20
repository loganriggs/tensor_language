"""Frozen common-factor row transfer, oracle-reuse controls, independent quadrature exports."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from core import metric,inner,multiply,coefficients_from_dense,evaluate,terms
from sweep import target_cases
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
sweep=json.load(open(P/'COMMON_FACTOR_SWEEP_V1.json'))['records'];cases=torch.load(P.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',weights_only=True);M=metric(5,4);chol=torch.linalg.cholesky(M);eye=torch.eye(15);records=[];exports=[];nodes,weights=np.polynomial.hermite.hermgauss(5)
for index,case in enumerate(cases):
 best=min((r for r in sweep if r['case']==f'native_{index}'),key=lambda r:r['gaussian_relative_error']);q=torch.tensor(best['common_quadratic']);quot=torch.tensor(best['quotients']);coeff=multiply(q,quot,5,2,2);t0=coefficients_from_dense(case['hessian_not_applicable_quartic'][0].double());scale=inner(t0,t0,M).sqrt();pred=coeff*scale;features=multiply(q,eye,5,2,2);design=(features@chol).T
 torch.manual_seed(999);random=torch.randn(1,15);random/=inner(random,random,metric(5,2)).sqrt();random_design=(multiply(random,eye,5,2,2)@chol).T
 x=torch.tensor(list(itertools.product(nodes,repeat=5)))*2**.5;w=torch.tensor([np.prod(a) for a in itertools.product(weights,repeat=5)])/np.pi**2.5
 actual=evaluate(q,x,2)*evaluate(quot,x,2);truth=evaluate(t0/scale,x,4);quaderr=float((((actual-truth).square().sum(-1)*w).sum()/(truth.square().sum(-1)*w).sum()).sqrt());assert abs(quaderr-best['gaussian_relative_error'])<1e-8
 exports.append(dict(case_index=index,common_quadratic=q.tolist(),quotients=quot.tolist(),original_scale=float(scale),quadrature_error=quaderr,parameter_values=75))
 for row,raw in enumerate(case['hessian_not_applicable_quartic']):
  t=coefficients_from_dense(raw.double());den=inner(t,t,M);diff=pred-t;fixed=float((inner(diff,diff,M)/den).sqrt());alpha=inner(pred,t,M)/inner(pred,pred,M);diff=alpha*pred-t;scalar=float((inner(diff,diff,M)/den).sqrt());rhs=(t@chol).T
  ls=torch.linalg.lstsq(design,rhs).solution;reuse=float((design@ls-rhs).norm()/rhs.norm());lsrand=torch.linalg.lstsq(random_design,rhs).solution;randerr=float((random_design@lsrand-rhs).norm()/rhs.norm())
  records.append(dict(case_index=index,row=row,frozen_program_error=fixed,oracle_scalar_error=scalar,oracle_fixed_factor_error=reuse,oracle_random_factor_error=randerr))
  if row==0:assert abs(fixed-best['gaussian_relative_error'])<1e-8
held=[r for r in records if r['row']>0];median=float(np.median([r['frozen_program_error'] for r in held]));fraction=sum(r['oracle_fixed_factor_error']<r['oracle_random_factor_error'] for r in held)/len(held)
# Independent planted check, same exact-degree quadrature.
best=min((r for r in sweep if r['case']=='planted_shared'),key=lambda r:r['gaussian_relative_error']);q=torch.tensor(best['common_quadratic']);quot=torch.tensor(best['quotients']);x=torch.tensor(list(itertools.product(nodes,repeat=6)))*2**.5;w=torch.tensor([np.prod(a) for a in itertools.product(weights,repeat=6)])/np.pi**3;t=next(r['target'] for r in target_cases() if r['name']=='shared_quartic_dag');truth=evaluate(t,x,4);actual=evaluate(q,x,2)*evaluate(quot,x,2);qe=float((((actual-truth).square().sum(-1)*w).sum()/(truth.square().sum(-1)*w).sum()).sqrt());assert abs(qe-best['gaussian_relative_error'])<1e-8
out=dict(records=records,exports=exports,planted_quadrature_error=qe,predictions=dict(pred_a_replay=True,pred_b_frozen_prediction=median<.5,pred_c_reuse=fraction>=.75),scope='176 heldout row coefficient targets; frozen program is prediction, scalar and quotient refits are oracle reuse-capacity tests using heldout weights, not OOD predictive validation.')
(P/'COMMON_FACTOR_TRANSFER_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions']);print('heldout rows',len(held),'fixed median',median,'reuse fraction',fraction)
for name in ['frozen_program_error','oracle_scalar_error','oracle_fixed_factor_error','oracle_random_factor_error']:print(name,'median',np.median([r[name] for r in held]),'worst',max(r[name] for r in held))
