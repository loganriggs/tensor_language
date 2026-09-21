"""Reconstruct finite cuts and certify a numerical convex-combination lower bound."""
import json
from pathlib import Path
import numpy as np
import torch
from fixed_read_robust_problem import build
from quadratic_ball_extrema import extrema
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);p=build();d=p['data'];H=p['H'];V=p['V'];atoms=p['atoms'];G,b,c=p['G'],p['b'],p['c'];result=json.loads((P/'ROBUST_FIXED_READ_V1.json').read_text());worst=json.loads((P/'NATIVE_WORST_READ_V1.json').read_text());target=1.1*next(r['worst_absolute_error'] for r in worst['rows'] if r['candidate']=='isotropic_baseline' and r['component']==3 and r['read']=='b');old=next(r['worst_absolute_error'] for r in worst['rows'] if r['candidate']=='graph' and r['component']==3 and r['read']=='b');x=torch.zeros(32,dtype=H.dtype);max_radius_excess=0.;cut_vectors=[]
 for row in result['history']:
  error=H[5]+torch.einsum('k,kij->ij',x,atoms)-p['metric'].true[5];linear=-2*error@d['mu'];bias=-torch.trace(d['old_covariance']@error)+d['mu']@error@d['mu'];out=extrema(error,linear,bias,1152**.5)
  for key in ('minimum','maximum'):
   z=out[key]['x'];max_radius_excess=max(max_radius_excess,float(z.norm())-1152**.5);psi=((z-d['mu'])@V).square()-torch.einsum('ik,ij,jk->k',V,d['old_covariance'],V);v=psi/target;v0=(z@error@z+linear@z+bias-psi@x)/target;G=torch.cat((G,torch.outer(v,v)[None]));b=torch.cat((b,(v*v0)[None]));c=torch.cat((c,v0.square()[None]));cut_vectors.append(z.numpy())
  x=torch.tensor(row['solver']['x'],dtype=H.dtype)
 solver=result['history'][-1]['solver'];active=solver['active'];weights=np.zeros(len(c));weights[active]=solver['multipliers'];weights/=weights.sum();Ga=np.einsum('k,kij->ij',weights,G.numpy());ba=weights@b.numpy();ca=float(weights@c.numpy());minimum=-np.linalg.solve(Ga,ba);lower=ca+ba@minimum;stationarity=float(np.linalg.norm(Ga@minimum+ba));finite=np.einsum('i,kij,j->k',x.numpy(),G.numpy(),x.numpy())+2*b.numpy()@x.numpy()+c.numpy();replay=abs(float(finite.max())-solver['maximum']);assert replay<1e-8 and stationarity<1e-8 and weights.min()>=0 and np.linalg.eigvalsh(Ga).min()>0
 original_upper=max(float(p['c'].max()),(old/target)**2)
 np.savez_compressed(P/'ROBUST_FIXED_READ_FINITE_CERTIFICATE_V1.npz',G=G.numpy(),b=b.numpy(),c=c.numpy(),weights=weights,cut_vectors=np.array(cut_vectors),minimizer=minimum)
 out=dict(lower_bound=float(lower),original_graph_upper_bound=original_upper,finite_model_replay=replay,dual_stationarity=stationarity,dual_min_eigenvalue=float(np.linalg.eigvalsh(Ga).min()),weight_sum=float(weights.sum()),max_cut_radius_excess=max_radius_excess,numerical_infeasible_at_one=bool(lower>1),scope='Numerical finite-cut lower bound excludes this fixed32direction/one-read edit under registered limits. All cuts lie on RMSball to floating-point tolerance. Not interval arithmetic, full minimax convergence or a general circuit impossibility. Original graph retained; finaltrial not accepted.')
 (P/'ROBUST_FIXED_READ_BOUND_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
