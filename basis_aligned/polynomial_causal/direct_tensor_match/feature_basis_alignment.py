"""Sign/permutation alignment distinguished from subspace agreement."""
import itertools,torch

def compare(reference,candidate):
 C=reference.T@candidate;k=C.shape[0];perm=max(itertools.permutations(range(k)),key=lambda p:sum(float(C[i,p[i]].abs()) for i in range(k)));matched=[float(C[i,perm[i]].abs()) for i in range(k)];singular=torch.linalg.svdvals(C)
 return dict(permutation=list(perm),matched_absolute_cosines=matched,minimum_matched_cosine=min(matched),principal_cosines=singular.tolist(),minimum_principal_cosine=float(singular.min()))
def toy_check():
 reference=torch.eye(4,dtype=torch.float64);permuted=reference[:,[2,0,3,1]]*torch.tensor([-1,1,-1,1]);rotated=torch.tensor([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=torch.float64)/2
 a=compare(reference,permuted);b=compare(reference,rotated);assert a['minimum_matched_cosine']==1 and b['minimum_matched_cosine']==.5 and abs(b['minimum_principal_cosine']-1)<1e-12
 return dict(sign_permutation_recovered=True,rotated_individual_cosine=b['minimum_matched_cosine'],rotated_subspace_cosine=b['minimum_principal_cosine'])
