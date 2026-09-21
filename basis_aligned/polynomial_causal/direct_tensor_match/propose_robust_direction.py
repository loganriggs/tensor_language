"""A missing rank-one direction from the finite minimax dual gradient."""
import json,numpy as np,torch
from pathlib import Path
from fixed_read_robust_problem import build
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);p=build();d=p['data'];H=p['H'];cert=np.load(P/'ROBUST_FIXED_READ_FINITE_CERTIFICATE_V1.npz');x=torch.tensor(cert['minimizer'],dtype=H.dtype);weights=torch.tensor(cert['weights'],dtype=H.dtype);z=torch.tensor(cert['cut_vectors'],dtype=H.dtype);Q=(H[5]+torch.einsum('k,kij->ij',x,p['atoms'])).detach().requires_grad_(True);forms=torch.cat((H[:5],Q[None]),0);ratios=p['metric'].ratios_full(forms)
 results=json.loads((P/'NATIVE_WORST_READ_V1.json').read_text());limit=1.1*next(r['worst_absolute_error'] for r in results['rows'] if r['component']==3 and r['read']=='b' and r['candidate']=='isotropic_baseline');error=Q-p['metric'].true[5];zc=z-d['mu'];values=(torch.einsum('ni,ij,nj->n',zc,error,zc)-torch.trace(d['old_covariance']@error))/limit;loss=weights[:8]@ratios+weights[8:]@values.square();gradient=torch.autograd.grad(loss,Q)[0];gradient=(gradient+gradient.T)/2
 with torch.no_grad():
  e,U=torch.linalg.eigh(gradient);k=int(e.abs().argmax());direction=U[:,k];generator=torch.Generator().manual_seed(920);random=torch.randn(1152,generator=generator,dtype=H.dtype);random/=random.norm();restricted=torch.einsum('ik,ij,jk->k',p['V'],gradient,p['V']);audit=json.loads((P/'ROBUST_FIXED_READ_BOUND_AUDIT_V1.json').read_text());assert abs(float(loss)-audit['lower_bound'])<1e-8
  assert float(restricted.abs().max())<1e-8
  out=dict(dual_objective=float(loss),gradient_eigenvalue=float(e[k]),restricted_gradient_max=float(restricted.abs().max()),directed_slope=float(direction@gradient@direction),random_slope=float(random@gradient@random),directions=dict(directed=direction.tolist(),random=random.tolist()),seed=920,scope='Direction proposal from numerical dual objective, not proof of minimax improvement. Both arms add one dense square at equal cost; initial amplitude zero preserves original graph.')
 (P/'ROBUST_DIRECTION_PROPOSAL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='directions'}))
if __name__=='__main__':main()
