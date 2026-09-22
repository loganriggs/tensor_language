"""Exhaustive small allocation control for independent quadratic outputs."""
import itertools,json
from pathlib import Path
import torch
from audit_output_local_bilinear import select
rows=[]
for seed in range(5):
 g=torch.Generator().manual_seed(29000+seed);e=torch.randn(3,4,generator=g,dtype=torch.float64);energy=e.square().sum(1);gains=torch.zeros_like(e)
 for v in range(3):
  p,n=select(e[v],4);gains[v,:len(p)]+=e[v,p].square();gains[v,:len(n)]+=e[v,n].square()
 for metric in ['natural','equal_output']:
  score=gains if metric=='natural' else gains/energy[:,None]
  budget=4;ids=score.flatten().argsort(descending=True)[:budget];ks=torch.bincount(ids//4,minlength=3)
  chosen=sum(float(score[v,:k].sum()) for v,k in enumerate(ks.tolist()))
  oracle=max(sum(float(score[v,:k].sum()) for v,k in enumerate(allocation)) for allocation in itertools.product(range(5),repeat=3) if sum(allocation)==budget)
  err=abs(chosen-oracle)/max(oracle,1e-30);assert err<1e-12
  rows.append(dict(seed=seed,metric=metric,relative_gap=err))
Path(__file__).with_name('OUTPUT_LOCAL_ALLOCATION_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n')
print('10 exhaustive allocation checks passed')
