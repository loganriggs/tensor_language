"""CPU audit: execution precision, parameter sensitivity, and feature matching across gauges."""
import json,itertools
from pathlib import Path
import torch
from core import metric,inner,evaluate
from quartic_sparse_basis import atoms
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
source=json.load(open(P/'QUARTIC_CONTEXT_BASIS_V1.json'));M2=metric(5,2);M4=metric(5,4);records=[];matches=[]
torch.manual_seed(381);x=torch.randn(2048,5)
for context in source['records']:
 for run in context['runs']:
  bank=torch.tensor(run['bank']);refit=next(r for r in run['refits'] if r['products']==4);root=torch.tensor(refit['root']);ids=refit['support'];coeff=root@atoms(bank,5)[ids];den=inner(coeff,coeff,M4)
  pairs=list(itertools.combinations_with_replacement(range(4),2));q=evaluate(bank.float(),x.float(),2);prod=torch.stack([q[:,pairs[i][0]]*q[:,pairs[i][1]] for i in ids],-1);pred=(prod@root.float().T).double();truth=evaluate(coeff,x,4);fp32=float((pred-truth).norm()/truth.norm())
  coef32=(root.float()@atoms(bank.float(),5)[ids]).double();delta=coef32-coeff;coefficient32=float((inner(delta,delta,M4)/den).sqrt())
  feature=atoms(bank,5)[ids];term_energy=root.square().sum(0)*((feature@M4)*feature).sum(1);cancellation=float(term_energy.sqrt().sum()/den.sqrt());perturb=[]
  for eps,seed in itertools.product([1e-6,1e-4,1e-2],range(3)):
   torch.manual_seed(seed);bn=bank+eps*bank.square().mean().sqrt()*torch.randn_like(bank);rn=root+eps*root.square().mean().sqrt()*torch.randn_like(root);diff=rn@atoms(bn,5)[ids]-coeff
   perturb.append(dict(relative_rms_noise=eps,seed=seed,relative_function_change=float((inner(diff,diff,M4)/den).sqrt())))
  records.append(dict(case_index=context['case_index'],seed=run['seed'],gauge_condition=run['condition'],fp32_execution_error=fp32,fp32_coefficient_error=coefficient32,term_norm_sum_ratio=cancellation,perturbations=perturb))
 b0,b1=[torch.tensor(r['bank']) for r in context['runs']];corr=(b0@M2@b1.T).abs();perms=list(itertools.permutations(range(4)));perm=max(perms,key=lambda p:sum(float(corr[i,p[i]]) for i in range(4)));scores=[float(corr[i,perm[i]]) for i in range(4)]
 matches.append(dict(case_index=context['case_index'],optimal_permutation=perm,absolute_gaussian_correlations=scores,mean_correlation=sum(scores)/4,minimum_correlation=min(scores)))
out=dict(records=records,feature_matches=matches,predictions=dict(pred_a_float32=all(r['fp32_execution_error']<1e-5 for r in records),pred_b_perturbation=all(z['relative_function_change']<1e-3 for r in records for z in r['perturbations'] if z['relative_rms_noise']==1e-4),pred_c_feature_agreement=all(m['minimum_correlation']>.99 for m in matches)),scope='Four-product exports; float32 execution versus exact polynomial, relative-RMS factor noise, two gauge restarts from same fitted feature span. Feature matching allows sign/permutation only; disagreement does not refute functional equivalence.')
(P/'QUARTIC_STABILITY_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions']);print('maxfp32',max(r['fp32_execution_error'] for r in records));print('maxnoise1e-4',max(z['relative_function_change'] for r in records for z in r['perturbations'] if z['relative_rms_noise']==1e-4));print('worstmatch',min(m['minimum_correlation'] for m in matches))
