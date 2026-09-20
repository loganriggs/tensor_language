"""Native derivative fixture: unrelated batch peers cannot change singleton fit."""
import json
from pathlib import Path
import numpy as np
import torch
from audit_singleton_derivative_minimax import singleton_fit
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    prior=json.loads((P/'SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json').read_text());F=np.array(prior['plane'])
    atoms=[np.outer(F[:,0],F[:,0]),(np.outer(F[:,0],F[:,1])+np.outer(F[:,1],F[:,0]))/2**.5,np.outer(F[:,1],F[:,1])]
    for i,j in prior['supports']['shared2_plus3']:
        z=np.zeros((5,5));z[i,j]=z[j,i]=1/(1 if i==j else 2**.5);atoms.append(z)
    atoms=np.array(atoms);c=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)['cases'][0]
    G=c['gradient'].numpy()[:2];H=c['hessian_reference'].numpy()[:2];sets=[[i] for i in range(5)]+[[i,j] for i in range(5) for j in range(i+1,5)]
    a=np.stack([np.broadcast_to(np.isin(np.arange(5),s).astype(float),(2,5)) for s in sets]+[x.numpy()[:2] for x in c['amplitudes'].values()])
    baseline,_=singleton_fit(G,H,a,atoms);permuted,_=singleton_fit(G[::-1],H[::-1],a[:,::-1],atoms);alone,_=singleton_fit(G[:1],H[:1],a[:,:1],atoms)
    errors=dict(permutation=float(abs(baseline-permuted[::-1]).max()),peer_removal=float(abs(baseline[:1]-alone).max()))
    assert max(errors.values())<1e-12
    (P/'SINGLETON_MINIMAX_INDEPENDENCE_RESULT.json').write_text(json.dumps(errors,indent=2)+'\n');print(errors)
if __name__=='__main__':main()
