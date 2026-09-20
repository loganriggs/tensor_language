"""Export sparse shared quartic programs; independently validate Gaussian errors by quadrature."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from core import evaluate,terms
from sweep import target_cases
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
sweep=json.load(open(P/'QUARTIC_BASIS_SWEEP_V1.json'));native=torch.load(P/'NATIVE_QUARTIC_PILOT_V1.pt',weights_only=False)
targets={'planted_shared':next(x['target'] for x in target_cases() if x['name']=='shared_quartic_dag'),'native_shared':native['target']}
exports=[]
for name,d,budget in [('planted_shared',6,3),('native_shared',5,4)]:
 candidates=[r for r in sweep['records'] if r['target']==name]
 row=min(candidates,key=lambda r:next(q['error'] for q in r['refits'] if q['products']==budget))
 refit=next(q for q in row['refits'] if q['products']==budget)
 bank=torch.tensor(row['bank']);root=torch.tensor(refit['root']);pairs=list(itertools.combinations_with_replacement(range(len(bank)),2));support=[pairs[i] for i in refit['support']]
 nodes,weights=np.polynomial.hermite.hermgauss(5)
 x=torch.tensor(list(itertools.product(nodes,repeat=d)))*2**.5
 w=torch.tensor([np.prod(a) for a in itertools.product(weights,repeat=d)])/np.pi**(d/2)
 # Direct execution of quadratic features followed by selected shared products.
 q=evaluate(bank,x,2);products=torch.stack([q[:,i]*q[:,j] for i,j in support],-1);prediction=products@root.T
 truth=evaluate(targets[name],x,4);error=float((((prediction-truth).square().sum(-1)*w).sum()/(truth.square().sum(-1)*w).sum()).sqrt())
 assert abs(error-refit['error'])<1e-10
 exports.append(dict(target=name,input_dimension=d,quadratic_monomial_order=terms(d,2),bank=bank.tolist(),shared_products=support,output_weights=root.tolist(),parameter_values=bank.numel()+root.numel(),quadrature_error=error,coefficient_metric_error=refit['error'],quadrature_points=len(x),selected_optimizer=row['optimizer'],selected_lr=row['lr'],selected_seed=row['seed']))
 print(name,budget,error)
(P/'SPARSE_QUARTIC_PROGRAMS_V1.json').write_text(json.dumps(dict(programs=exports,scope='Two independently executable sparse quartic programs. Native scope is one five-amplitude four-output local numerator; no causal identification.'),indent=2)+'\n')
