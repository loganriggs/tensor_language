"""Gauge-invariant graph sharing and disjoint-consumer least-squares control.
Assign each product to its strongest pair before refitting; no data outcomes
choose the graph. Compare with optimal unrestricted output coefficients.
"""
from pathlib import Path
import argparse,json,torch,time
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['square','mixed'],required=True);args=parser.parse_args();start=time.perf_counter()
prefix='SHARED' if args.family=='square' else 'MIXED'
fit=json.loads((P/f'{prefix}_PRODUCT_NATIVE_FIT_V1.json').read_text());old=torch.load(P/f'{prefix}_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True)[fit['winner']]
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);T=d['teacher']
left=old.get('left_reader',old.get('shared_reader'));right=old.get('right_reader',left)
A=root@left;B=root@right
K=.5*((A.T@A)*(B.T@B)+(A.T@B)*(B.T@A));norm=K.diag().clamp_min(0).sqrt();assert norm.min()>0
# Normalize atom Frobenius norms. This makes assignment and Gram cutoff
# invariant to individual input/product rescalings.
A=A/norm.sqrt();B=B/norm.sqrt();left=left/norm.sqrt();right=right/norm.sqrt();coeff=old['product_weights']/d['scales'][None,:]*norm[:,None]
K=.5*((A.T@A)*(B.T@B)+(A.T@B)*(B.T@A));rhs=torch.einsum('ir,oij,jr->ro',A,T,B)
energy=coeff.reshape(-1,3,2).square().sum(2);fraction=energy/energy.sum(1,keepdim=True).clamp_min(1e-300);effective=1/fraction.square().sum(1)
assignment=energy.argmax(1)
def solve(mask):
 w=torch.zeros_like(rhs);checks=[]
 for output in range(6):
  ids=mask[:,output].nonzero().flatten()
  gram=K[ids[:,None],ids[None,:]];r=rhs[ids,output]
  ev,V=torch.linalg.eigh(gram);keep=ev>ev[-1]*1e-12
  w[ids,output]=V[:,keep]@((V[:,keep].T@r)/ev[keep])
  checks.append(dict(output=output,products=len(ids),discarded=int((~keep).sum()),normal_relative_error=float((gram@w[ids,output]-r).norm()/r.norm())))
 return w,checks
allmask=torch.ones_like(rhs,dtype=torch.bool);privatemask=assignment[:,None]==torch.arange(6)[None,:]//2
full,full_checks=solve(allmask);private,private_checks=solve(privatemask)
def evaluate(w):
 raw=torch.einsum('ro,ir,jr->oij',w,A,B);hat=.5*(raw+raw.transpose(-1,-2));coefficient=float((hat-T).norm()/T.norm())
 weights=w*d['scales'][None,:];raw_native=torch.einsum('ro,ir,jr->oij',weights,left,right);Qhat=.5*(raw_native+raw_native.transpose(-1,-2))
 delta=d['z']-d['mu'];reads=((delta@left)*(delta@right))@weights
 for j,pair in enumerate(d['pairs']):
  for k,Q in enumerate(pair['Qs']):
   pos=2*j+k;reads[:,pos]+=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q)+delta@(2*Q@d['mu'])-torch.trace(d['old_covariance']@Qhat[pos])
 phi=((d['h']@old['h_readers']-.5*reads[:,::2])/d['scale'][:,None]-old['alpha'])*(reads[:,1::2]/d['scale'][:,None]-old['beta'])
 errors=[]
 for j,pair in enumerate(d['pairs']):
  truth=pair['truth'][d['indices']];errors.append(float((phi[d['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
 return dict(coefficient_error=coefficient,per_mode_errors=errors,readout_nonzeros=int((w!=0).sum()))
a=evaluate(full);b=evaluate(private);ratio=b['coefficient_error']/a['coefficient_error']
output=dict(family=args.family,parent_winner=fit['winner'],source_products=len(assignment),pair_assignment_counts=torch.bincount(assignment,minlength=3).tolist(),median_effective_pair_consumers=float(effective.median()),energy_weighted_effective_pair_consumers=float((effective*energy.sum(1)).sum()/energy.sum()),fraction_products_above_1p5_effective_consumers=float((effective>1.5).double().mean()),unrestricted=a,private=b,coefficient_error_ratio=ratio,least_squares_checks=dict(unrestricted=full_checks,private=private_checks),predictions=dict(pred_a_normal_equations=max(c['normal_relative_error'] for c in full_checks+private_checks)<1e-8,pred_b_sharing_necessity=ratio>=1.25),seconds=time.perf_counter()-start,scope='Fixed learned atoms, no optimization of input directions or outcome-based edge assignment. Pair-private control tests this deterministic edit, not the optimal partition or impossibility of alternative private circuits. Energy is coefficient contribution before cancellation, not semantic/causal importance; parent fidelity failures remain.')
(P/f'{prefix}_PRODUCT_REUSE_AUDIT_V1.json').write_text(json.dumps(output,indent=2)+'\n');torch.save(dict(normalizing_atom_norm=norm,unrestricted_weights=full,private_weights=private,assignment=assignment),P/f'{prefix}_PRODUCT_REUSE_WEIGHTS_V1.pt');print(json.dumps(output,indent=2))
