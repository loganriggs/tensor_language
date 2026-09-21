"""Independent small-case check: probe MSE is a lifted moment quadratic form.
With numerator quartic and known RMS s, M=mean(v4(x)v4(x)^T/s^4).
This check materializes M only in five dimensions, never on model inputs.
"""
import json
from pathlib import Path
import torch
torch.set_num_threads(2)
torch.manual_seed(525)
dtype=torch.float64
x=torch.randn(80,5,dtype=dtype)
x[:,-1]=1.
x[:,-2]=x[:,-2].abs()+.5
scale=x[:,-2]
A=torch.randn(5,5,dtype=dtype);A=(A+A.T)/2
B=torch.randn(5,5,dtype=dtype);B=(B+B.T)/2
C=torch.randn(5,5,dtype=dtype);C=(C+C.T)/2
D=torch.randn(5,5,dtype=dtype);D=(D+D.T)/2
delta=torch.einsum('ij,kl->ijkl',A,B)-torch.einsum('ij,kl->ijkl',C,D)
lift=torch.einsum('bi,bj,bk,bl->bijkl',x,x,x,x).flatten(1)
numerator=((x@A)*x).sum(1)*((x@B)*x).sum(1)-((x@C)*x).sum(1)*((x@D)*x).sum(1)
checks={}
for normalized in [False,True]:
 features=lift/scale.square()[:,None] if normalized else lift
 values=numerator/scale.square() if normalized else numerator
 M=features.T@features/len(x)
 left=delta.flatten()@M@delta.flatten()
 right=values.square().mean()
 checks['normalized' if normalized else 'numerator']=float((left-right).abs()/right)
assert max(checks.values())<1e-12
# Equal mean and covariance, unequal eighth moment: no covariance-only rule
# can give the exact quartic functional norm for both empirical distributions.
rademacher=torch.tensor([-1.,1.,-1.,1.],dtype=dtype)
mixture=torch.tensor([0.,0.,-2**.5,2**.5],dtype=dtype)
counterexample={'variances':[float(t.square().mean()) for t in [rademacher,mixture]],
 'eighth_moments':[float(t.pow(8).mean()) for t in [rademacher,mixture]]}
assert abs(counterexample['variances'][0]-counterexample['variances'][1])<1e-12
assert abs(counterexample['eighth_moments'][1]-8)<1e-12
out=dict(passed=True,relative_errors=checks,equal_covariance_counterexample=counterexample,
 scope='Verifies empirical full moment loss on constrained quartic inputs, including inverse RMS fourth-power weighting. It does not prove empirical moments generalize to new documents.')
path=Path(__file__).with_name('EMPIRICAL_QUARTIC_MOMENT_CHECK_V1.json')
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
