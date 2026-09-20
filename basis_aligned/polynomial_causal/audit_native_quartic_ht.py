"""Canonical-polynomial baseline and rank2 pair-tree control for native quartics."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from quartic_pair_tree import factor,reconstruct,symmetrize
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    cases=torch.load(A/'native_two_mlp_quartic_ht_v1r1.pt',map_location='cpu',weights_only=True)
    keys=list(itertools.combinations_with_replacement(range(5),4));index={k:i for i,k in enumerate(keys)}
    slots=list(itertools.product(range(5),repeat=4));mapping=np.array([index[tuple(sorted(k))] for k in slots])
    x=np.random.default_rng(832).normal(size=(32,5));monomials=np.array([np.prod(x[:,k],axis=1) for k in keys]).T
    rows=[];replay=[]
    for c in cases:
        for row,H in enumerate(c['hessian_not_applicable_quartic'].numpy()):
            coeff=np.zeros((4,70))
            for i,j in enumerate(mapping):coeff[:,j]+=H.reshape(4,-1)[:,i]
            native=np.einsum('oijkl,bi,bj,bk,bl->bo',H,x,x,x,x);canonical=monomials@coeff.T
            replay.append(float(np.linalg.norm(native-canonical)/max(np.linalg.norm(native),1e-30)))
            symmetric=symmetrize(H);den=max(np.linalg.norm(symmetric),1e-30)
            for name,target in [('native_pair',H),('symmetric',symmetric)]:
                approx=reconstruct(factor(target,2));values=np.einsum('oijkl,bi,bj,bk,bl->bo',approx,x,x,x,x)
                rows.append(dict(panel=c['panel'],role=c['role'],template=c['template'],row=row,representative=name,rank2_polynomial_coefficient_error=float(np.linalg.norm(symmetrize(approx)-symmetric)/den),rank2_random_amplitude_error=float(np.linalg.norm(values-native)/max(np.linalg.norm(native),1e-30))))
    assert max(replay)<1e-10
    result=dict(canonical_monomial_count=70,canonical_values=280,raw_tensor_values=2500,rank8_tree_values=656,rank2_tree_values=116,canonical_replay=max(replay),summary={name:dict(max_coefficient_error=max(r['rank2_polynomial_coefficient_error'] for r in rows if r['representative']==name),max_random_amplitude_error=max(r['rank2_random_amplitude_error'] for r in rows if r['representative']==name)) for name in ['native_pair','symmetric']},rows=rows,scope='Same homogeneous two-MLP numerator only, conditional native derivative frames/readers. Standard dense pair-tree cores; no sparse-core learning, no normalized finite-response fidelity. Random amplitude test is algebraic validation, not text OOD.')
    (P/'NATIVE_QUARTIC_HT_CPU_AUDIT.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
