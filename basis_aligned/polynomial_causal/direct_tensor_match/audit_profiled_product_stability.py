"""Align quadratic product atoms across the four profiled graph fits.
Gauge-invariant joint atom inner products include output weights. These are
nearby/perturbed starts, not independent model/data discovery replications.
"""
from pathlib import Path
import torch,json,numpy as np
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);fit=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());programs=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)
def factors(p):
 p=p['shared_mixed'];return root@p['left_reader'],root@p['right_reader'],p['product_weights'].T/d['scales'][:4,None]
def gram(x,y):
 L,R,W=x;A,B,V=y
 return (W.T@V)*.5*((L.T@A)*(R.T@B)+(L.T@B)*(R.T@A))
def matrix(x):
 L,R,W=x;raw=torch.einsum('or,ir,jr->oij',W,L,R);return .5*(raw+raw.transpose(-1,-2))
def match(x,y):
 normx=gram(x,x).diag().clamp_min(0);normy=gram(y,y).diag().clamp_min(0);cos=gram(x,y)/(normx[:,None]*normy[None,:]).sqrt();a,b=linear_sum_assignment(-cos.abs().numpy());matched=cos[a,b];importance=normx[a]/normx.sum()
 return dict(median_abs_cosine=float(matched.abs().median()),median_signed_cosine=float(matched.median()),negative_matches=int((matched<0).sum()),fraction_atoms_abs_above_090=float((matched.abs()>=.9).double().mean()),atom_energy_fraction_abs_above_090=float((importance*(matched.abs()>=.9)).sum()),energy_weighted_abs_cosine=float((importance*matched.abs()).sum()),min_abs_cosine=float(matched.abs().min()))
reference=factors(programs[fit['winner']]);L,R,W=reference;n=L.shape[1];perm=torch.randperm(n,generator=torch.Generator().manual_seed(714));a=torch.linspace(.5,2,n);b=torch.linspace(.7,3,n)*torch.where(torch.arange(n)%2==0,1.,-1.);gauge=(L[:,perm]*a,R[:,perm]*b,W[:,perm]/(a*b));gauge_check=match(reference,gauge);assert gauge_check['min_abs_cosine']>1-1e-12
records=[];refmatrix=matrix(reference)
for key,p in programs.items():
 if key==fit['winner']:continue
 x=factors(p);record=dict(candidate=key,**match(reference,x),function_coefficient_disagreement=float((matrix(x)-refmatrix).norm()/refmatrix.norm()));records.append(record)
out=dict(reference=fit['winner'],gauge_control=gauge_check,records=records,predictions=dict(pred_a_gauge=gauge_check['min_abs_cosine']>1-1e-12,pred_b_stable_atoms=all(r['atom_energy_fraction_abs_above_090']>=.8 for r in records)),scope='Hungarian alignment on absolute joint quadratic-atom cosine, with signed matches separately reported. Atom energies are before cancellation. Coefficient-level stability across two rates and a nearby start does not establish monosemanticity, cross-data identification or global uniqueness.')
(P/'PROFILED_PRODUCT_STABILITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
