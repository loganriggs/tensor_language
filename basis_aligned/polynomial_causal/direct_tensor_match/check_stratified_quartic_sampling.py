"""Exact finite-population controls for quartic coefficient sampling allocations."""
import itertools,json,math
from pathlib import Path
import torch
from implicit_quartic import entries
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1613);d=12
idx=torch.tensor(list(itertools.product(range(d),repeat=4)));patterns=[]
for row in idx.tolist():
 counts=sorted([row.count(x) for x in set(row)],reverse=True);patterns.append(''.join(map(str,counts)))
keys=['4','31','22','211','1111'];groups={k:torch.tensor([i for i,p in enumerate(patterns) if p==k]) for k in keys};counts=torch.tensor([len(groups[k]) for k in keys],dtype=torch.float64)
params=[torch.randn(*s) for s in [(3,4),(4,5),(4,5),(5,7),(7,d),(7,d)]];v=entries(*params,idx).square().sum(-1)
coord=(idx==0).all(1).double();distinct=(idx.sort(dim=1).values==torch.tensor([0,1,2,3])).all(1).double()/24**2;paired=(idx.sort(dim=1).values==torch.tensor([0,0,1,1])).all(1).double()/6**2
rows=[]
for name,values in [('dense',v),('single_diagonal',coord),('single_distinct',distinct),('single_paired',paired)]:
 true=float(values.sum());strata=[values[groups[k]] for k in keys];std=torch.tensor([float(x.std(unbiased=False)) for x in strata]);optimal=counts*std
 for budget in [500,5000]:
  allocations={'equal':[budget//5]*5}
  # Oracle Neyman allocation knows every stratum variance; not available from a few pilot samples.
  raw=optimal/optimal.sum()*(budget-5);n=raw.floor().long()+1;remaining=budget-int(n.sum());order=(raw-raw.floor()).argsort(descending=True);n[order[:remaining]]+=1;allocations['oracle_neyman']=n.tolist()
  torch.manual_seed(budget);estimate=values[torch.randint(len(values),(200,budget))].mean(1)*len(values)
  rows.append(dict(target=name,method='uniform',budget=budget,relative_rmse=float(((estimate/true-1)**2).mean().sqrt()),zero_estimates=int((estimate==0).sum()),repetitions=200))
  for method,allocation in allocations.items():
   torch.manual_seed(budget);estimate=torch.zeros(200)
   for s,n in enumerate(allocation):estimate+=strata[s][torch.randint(len(strata[s]),(200,n))].mean(1)*counts[s]
   se=math.sqrt(sum(float(counts[s]*std[s])**2/allocation[s] for s in range(5)))/true
   rows.append(dict(target=name,method=method,budget=budget,allocation=allocation,theoretical_relative_se=se,relative_rmse=float(((estimate/true-1)**2).mean().sqrt()),zero_estimates=int((estimate==0).sum()),repetitions=200))
out=dict(input_dimension=d,stratum_counts={k:len(groups[k]) for k in keys},records=rows,scope='Known finite toy tensors; uniform/equal-stratum allocations at equal query budgets. Oracle Neyman is an unattainable-information comparator unless variances are known; empirical pilot zeros do not prove zero mass.')
(P/'STRATIFIED_QUARTIC_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 if r['budget']==500:print(r['target'],r['method'],r['relative_rmse'],r['zero_estimates'])
