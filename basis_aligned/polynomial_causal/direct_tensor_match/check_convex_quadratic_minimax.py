from pathlib import Path
import json,numpy as np
from convex_quadratic_minimax import solve
P=Path(__file__).parent;rng=np.random.default_rng(83109);records=[]
for n in (2,3,5,8,14):
 truth=rng.normal(size=n);A=rng.normal(size=(n+3,n));G=A.T@A+np.eye(n)
 out=solve(G[None],(-G@truth)[None],np.array([truth@G@truth]));error=float(np.linalg.norm(np.array(out['x'])-truth));assert error<1e-6 and out['primal_violation']<1e-8 and out['kkt_residual']<1e-5
 records.append(dict(kind='planted_unique',dimension=n,error=error,**out))
 # Symmetric opposing targets have known optimum x=0, maximum squared error=4.
 G=np.repeat(np.eye(n)[None],2*n,axis=0);b=np.concatenate([2*np.eye(n),-2*np.eye(n)]);c=np.full(2*n,4.)
 out=solve(G,b,c,initial=rng.normal(size=n));assert np.linalg.norm(out['x'])<1e-6 and abs(out['maximum']-4)<1e-8 and out['kkt_residual']<1e-5
 records.append(dict(kind='incompatible_unit_bars',dimension=n,**out))
(P/'CONVEX_QUADRATIC_CONTROLS_V1.json').write_text(json.dumps(dict(records=records,pass_all=True,scope='Solver controls only, not native feasibility or circuit recovery. Both attainable planted solutions and incompatible norm bars; KKT residual is a numerical diagnostic, not an interval certificate.'),indent=2)+'\n')
print('10 solver controls pass')
