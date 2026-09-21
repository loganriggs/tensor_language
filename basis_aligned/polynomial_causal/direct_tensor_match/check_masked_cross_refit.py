from pathlib import Path
import json,torch
from masked_cross_refit import solve
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[]
for case in range(5):
 rng=torch.Generator().manual_seed(38000+case);r=5+case;p=7+case
 L=torch.randn(r,r,dtype=torch.float64,generator=rng);R=torch.randn(p,p,dtype=torch.float64,generator=rng);L=L/L.norm(dim=0);R=R/R.norm(dim=0);T=torch.randn(2,r,p,dtype=torch.float64,generator=rng);mask=torch.rand(r,p,generator=rng)>.6;ridge=1e-8
 x,diag=solve(L,R,T,mask,ridge=ridge,steps=500,tolerance=1e-10)
 ids=mask.flatten().nonzero().flatten();G=torch.kron((L.T@L).contiguous(),(R.T@R).contiguous())[ids][:,ids]+ridge*torch.eye(len(ids),dtype=torch.float64);rhs=(L.T@T@R)[:,mask];dense=torch.linalg.solve(G,rhs.T).T
 error=float((x[:,mask]-dense).norm()/dense.norm());assert error<1e-8 and diag['relative_normal_residual']<1e-8
 rows.append(dict(case=case,dense_solution_replay=error,**diag))
(P/'MASKED_CROSS_REFIT_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');print('PASS five independent dense Kronecker solves')
