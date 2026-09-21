"""Coefficient-Frobenius principal angles between quadratic feature dictionaries.
A subspace diagnostic, not semantic identity or activation-weighted stability.
"""
from pathlib import Path
import json,itertools
import torch
P=Path(__file__).resolve().parent

def gram(u,v,a,b):
 dot=lambda x,y:torch.einsum('ikd,jld->ijkl',x,y)
 return .5*(dot(u,a)*dot(v,b)+dot(u,b)*dot(v,a)).sum((-1,-2))

def whitening(g):
 e,v=torch.linalg.eigh((g+g.T)/2);mask=e>e.max()*1e-10
 return v[:,mask]/e[mask].sqrt(),int(mask.sum())

def compare(p,q):
 a,ra=whitening(gram(p['U'],p['V'],p['U'],p['V']));b,rb=whitening(gram(q['U'],q['V'],q['U'],q['V']));s=torch.linalg.svdvals(a.T@gram(p['U'],p['V'],q['U'],q['V'])@b)
 assert float(s.max())<1+1e-8
 return dict(ranks=[ra,rb],cosines=s.tolist(),mean_squared_cosine=float(s.square().mean()),minimum_cosine=float(s.min()),directions_above_099=int((s>.99).sum()),directions_above_09=int((s>.9).sum()))

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(61)
 u,v=torch.randn(3,2,7,dtype=torch.float64),torch.randn(3,2,7,dtype=torch.float64);a,b=torch.randn(4,2,7,dtype=torch.float64),torch.randn(4,2,7,dtype=torch.float64)
 dense=lambda u,v:.5*(torch.einsum('ikd,ike->ide',u,v)+torch.einsum('ikd,ike->ide',v,u))
 expected=dense(u,v).flatten(1)@dense(a,b).flatten(1).T
 dense_error=float((gram(u,v,a,b)-expected).norm()/expected.norm())
 assert dense_error<1e-12
 p=dict(U=u,V=v);identity=compare(p,p);assert identity['minimum_cosine']>1-1e-10
 # Permute and rescale individual quadratics; their span must stay fixed.
 perm=torch.tensor([2,0,1]);scale=torch.tensor([2.,-.7,.2],dtype=u.dtype);changed=dict(U=u[perm]*scale[:,None,None],V=v[perm]);invariant=compare(p,changed);assert invariant['minimum_cosine']>1-1e-10
 files={'inherited':'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt','value':'QUARTIC_RESPONSE_FEATURES_32_WEIGHT_0_V1.pt','half':'QUARTIC_RESPONSE_FEATURES_32_WEIGHT_1_V1.pt','path':'QUARTIC_PATH_FIT_V1.pt'}
 programs={k:{n:t.double() for n,t in torch.load(P/f,weights_only=True).items() if n in ['U','V']} for k,f in files.items()}
 rows=[]
 for a,b in itertools.combinations(programs,2):
  row=dict(left=a,right=b,**compare(programs[a],programs[b]));rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='cosines'}))
 result=dict(controls=dict(dense_replay=dense_error,identity_minimum=identity['minimum_cosine'],rescaled_permutation_minimum=invariant['minimum_cosine']),records=rows,scope='Post-hoc coefficient-space principal angles among32quadratic features. Gauge-invariant span comparison; not semantic identity, independently restarted recovery, or natural-input behavior.')
 (P/'QUARTIC_FEATURE_SUBSPACE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
