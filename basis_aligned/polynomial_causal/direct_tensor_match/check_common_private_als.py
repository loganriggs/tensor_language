from pathlib import Path
import json,torch
from common_private_als import private_forms,fit
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[]
for case in range(5):
 rng=torch.Generator().manual_seed(39000+case);r=2+case;p=3+case;E=torch.randn(r,p,dtype=torch.float64,generator=rng);B=torch.randn(2,r,p,dtype=torch.float64,generator=rng);C0=torch.randn(2,p,p,dtype=torch.float64,generator=rng);C0=(C0+C0.transpose(-1,-2))/2
 C=private_forms(E,B,C0);M=.5*torch.eye(p,dtype=E.dtype)+E.T@E;rhs=C0+E.T@B+B.transpose(-1,-2)@E
 K=torch.kron(M.contiguous(),torch.eye(p,dtype=E.dtype))+torch.kron(torch.eye(p,dtype=E.dtype),M.contiguous());dense=torch.linalg.solve(K,rhs.reshape(2,-1).T).T.reshape_as(C)
 replay=float((C-dense).norm()/dense.norm());assert replay<1e-10
 fitted,F,diag=fit(B,C0,steps=50);assert diag['final_squared_error']<diag['initial_squared_error']
 rows.append(dict(case=case,independent_lyapunov_replay=replay,initial_error=diag['initial_squared_error'],final_error=diag['final_squared_error'],monotone_steps=diag['iterations']))
(P/'COMMON_PRIVATE_ALS_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Five dense Kronecker controls for symmetric private-form minimizer and monotonealternation. No native optimization or generalglobaloptimality claim.'),indent=2)+'\n');print('PASS five independent symmetric solves and monotone alternating fits')
