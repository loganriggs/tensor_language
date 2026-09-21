"""Diagnose the least-energy output contrast without dropping it from the target."""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);T=d['teacher'][:4];F=T.flatten(1);eig,U=torch.linalg.eigh(F@F.T);u=U[:,0];weak=torch.einsum('o,oij->ij',u,T);removed=torch.einsum('o,ij->oij',u,weak);inv=d['inverse_root'];Qs=torch.stack([q for p in d['pairs'][:2] for q in p['Qs']]);D=torch.stack([inv@M@inv for M in removed])*d['scales'][:4,None,None];ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();rows=[]
for name in ['exact','whole_quadratic_read','centered_quadratic_remainder']:
 Q=Qs if name=='exact' else Qs-D
 reads=torch.einsum('ni,oij,nj->no',z,Q,z)
 if name=='centered_quadratic_remainder':
  linear=2*torch.einsum('oij,j->oi',D,d['mu']);bias=torch.einsum('ij,oji->o',d['old_covariance'],D)-torch.einsum('i,oij,j->o',d['mu'],D,d['mu']);reads+=z@linear.T+bias
 errors=[]
 for j,p in enumerate(d['pairs'][:2]):
  phi=((h@p['a']-.5*reads[:,2*j])/s-p['alpha'])*(reads[:,2*j+1]/s-p['beta']);truth=p['truth'][ids];errors.append(float((phi-truth).norm()/(truth-truth.mean()).norm()))
 rows.append(dict(arm=name,component_errors=errors))
# Right-mode rank lower bounds after output-energy reweighting; no fit.
bounds=[]
for power in [0,.5,1]:
 weights=eig.pow(-power/2);TT=torch.einsum('ao,oij->aij',U.T,T)*weights[:,None,None]
 K=sum(Q@Q for Q in TT);values=torch.linalg.eigvalsh(K).clamp_min(0).flip(0);bounds.append(dict(output_whitening_power=power,rank512_coefficient_lower_bound=float((values[512:].sum()/values.sum()).sqrt())))
out=dict(original_output_eigenvalues=eig.flip(0).tolist(),weakest_output_energy_fraction=float(eig[0]/eig.sum()),arms=rows,output_reweighting_bounds=bounds,scope='Opened-state counterfactual removal of the weakest original output contrast; both whole-read and centered-remainder variants. No target scope reduction or semantic claim. Rank512 bound is a necessary common-input-span bound for256mixed products after specified output reweighting.')
(P/'WEAK_SOURCE_CONTRAST_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
