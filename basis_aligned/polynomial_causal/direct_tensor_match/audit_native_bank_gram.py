"""Exact symmetric quartic coefficient Gram for products of quadratic forms."""
import itertools,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def gram(Q):
 pairs=list(itertools.combinations_with_replacement(range(len(Q)),2));out=torch.zeros(len(pairs),len(pairs),dtype=Q.dtype)
 for u,(a,b) in enumerate(pairs):
  for v,(c,d) in enumerate(pairs):
   out[u,v]=((Q[a]*Q[c]).sum()*(Q[b]*Q[d]).sum()+(Q[a]*Q[d]).sum()*(Q[b]*Q[c]).sum()+4*torch.trace(Q[a]@Q[c]@Q[b]@Q[d]))/6*(1 if a==b else 2**.5)*(1 if c==d else 2**.5)
 return out

def main():
 torch.set_num_threads(2);torch.manual_seed(935);q=torch.randn(3,4,4,dtype=torch.float64);q=(q+q.transpose(-1,-2))/2;columns=[]
 for a,b in itertools.combinations_with_replacement(range(3),2):
  h=torch.einsum('ij,kl->ijkl',q[a],q[b]);h=sum(h.permute(p) for p in itertools.permutations(range(4)))/24;columns.append(h.flatten()*(1 if a==b else 2**.5))
 C=torch.stack(columns);direct=C@C.T;replay=float((gram(q)-direct).norm()/direct.norm());assert replay<1e-12
 rows=[]
 for file,key in [('QUARTIC_BANK_CORE_V1.pt','core'),('ROOT_ARCHIVE_METRIC_V1.pt','Q')]:
  Q=torch.load(P/file,weights_only=True)[key].double();G=gram(Q);norm=G.diag().sqrt();normalized=G/norm[:,None]/norm[None,:];e=torch.linalg.eigvalsh(normalized);rows.append(dict(source=file,gram_eigenvalues=e.tolist(),min_over_max=float(e[0]/e[-1]),numerical_rank=int((e>1e-10*e[-1]).sum()),root_features=len(e)))
 out=dict(dense_24permutation_control=replay,banks=rows,scope='Exact coefficient-inner-product contractions in saved full-feature input spans. Input geometry is that of each saved covariance-scaled coordinate system; rank invariant under invertible input transforms, conditioning is not. No teacher full-model fidelity claim.')
 (P/'NATIVE_BANK_EXACT_GRAM_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
