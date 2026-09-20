"""Verify exact isotropic quartic norm-equivalence eigenvalues in small dimension."""
import json,math
from pathlib import Path
import torch
from core import metric
P=Path(__file__).resolve().parent;torch.set_num_threads(1);d=5;F=metric(d,4,'frobenius');M=metric(d,4,'gaussian');scale=F.diag().rsqrt();K=scale[:,None]*M*scale[None,:];eig=torch.linalg.eigvalsh(K);expected=torch.tensor([24.]*(math.comb(d+3,4)-math.comb(d+1,2))+[12.*(d+6)]*(math.comb(d+1,2)-1)+[3.*(d+4)*(d+6)],dtype=torch.float64);err=float((eig-expected).abs().max());assert err<1e-9
n=1152;low=24;mid=12*(n+6);high=3*(n+4)*(n+6)
out=dict(verified_dimension=d,maximum_eigenvalue_absolute_error=err,native_dimension=n,gaussian_relative_to_frobenius_eigenvalues=dict(harmonic_quartic=low,radius_squared_times_harmonic_quadratic=mid,radial_quartic=high),squared_metric_condition_number=high/low,norm_equivalence_condition=math.sqrt(high/low),scope='Exact homogeneous quartic coefficient metric under standard Gaussian inputs in fixed Euclidean coordinates. Bounds absolute norms; relative errors also depend on target directions. Covariance/noncentral/native activation distributions are different metrics.')
(P/'QUARTIC_METRIC_SPECTRUM_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
