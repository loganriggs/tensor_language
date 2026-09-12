"""Packed eigensolver versus a dense signed quartic flattening.

A pack/unpack metric<=1e-12; B eigenvalues/residual<=1e-8;
C action budget stops explicitly with no invented eigenpairs.
"""
from pathlib import Path
import json
import numpy as np
import torch
from quartic_matrixfree_eigen_v2 import SymmetricCoordinates,eigenmatrices
from quartic_weighted_trace_v1 import fitted_action

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91952)
d=8
torch.set_default_dtype(torch.float32)
coordinates=SymmetricCoordinates(d)
torch.set_default_dtype(torch.float64)
x=torch.randn(d,d);x=(x+x.T)/2
roundtrip=float((coordinates.unpack(coordinates.pack(x))-x).norm()/x.norm())
metric=abs(float(coordinates.pack(x).square().sum()/x.square().sum())-1)
b=torch.linalg.qr(torch.randn(4,d,2),mode='reduced')[0]
n=torch.randn(4,2);n/=n.norm(dim=-1,keepdim=True)
mix=torch.tensor([1.,-.8,.3,-.2])
def action(q):return fitted_action(b,n,mix,q)
dense=torch.stack([coordinates.pack(action(coordinates.unpack(v))) for v in torch.eye(coordinates.size)],1)
expected=torch.linalg.eigvalsh(dense).numpy();expected=expected[np.argsort(abs(expected))[-4:]]
values,matrices,diagnostic=eigenmatrices(action,d,k=4,tol=1e-11,ncv=16)
error=float(np.linalg.norm(np.sort(values)-np.sort(expected))/np.linalg.norm(expected))
_,missing,capped=eigenmatrices(action,d,k=4,tol=1e-11,ncv=16,max_actions=1)
result=dict(pred_a=max(roundtrip,metric)<=1e-12,
            pred_b=diagnostic['status']=='converged' and error<=1e-8 and max(diagnostic['relative_eigen_residuals'])<=1e-8,
            pred_c=capped['status']=='action_limit' and capped['operator_actions']==1 and not missing,
            metric_error=metric,roundtrip_error=roundtrip,eigenvalue_error=error,
            diagnostics=diagnostic,capped_diagnostics=capped,
            scope='CPU packed-coordinate and signed quartic eigensolver validity only; no native spectrum or atom-recovery claim.')
out=Path(__file__).with_name('QUARTIC_MATRIXFREE_EIGEN_V2_CONTROL.json');assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b'] and result['pred_c']
