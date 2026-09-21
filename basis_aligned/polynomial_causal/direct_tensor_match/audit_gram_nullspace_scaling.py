"""Check whether Gram freedom survives lifting a fixed-size quadratic dictionary."""
import json,itertools
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
 rng=np.random.default_rng(932);rows=[]
 for d in (5,6,8,12):
  pairs=list(itertools.combinations_with_replacement(range(d),2));basis=np.zeros((len(pairs),d,d))
  for k,(i,j) in enumerate(pairs):basis[k,i,j]=basis[k,j,i]=1 if i==j else 1/np.sqrt(2)
  U=np.linalg.qr(rng.normal(size=(len(pairs),15)))[0];Q=np.einsum('ai,ajk->ijk',U,basis);columns=[]
  for a,b in itertools.combinations_with_replacement(range(15),2):
   tensor=np.einsum('ij,kl->ijkl',Q[a],Q[b]);sym=sum(tensor.transpose(perm) for perm in itertools.permutations(range(4)))/24
   assert np.max(abs(sym-sym.transpose(1,0,2,3)))<1e-12 and np.max(abs(sym-sym.transpose(0,2,1,3)))<1e-12
   columns.append((sym*(1 if a==b else np.sqrt(2))).ravel())
  C=np.array(columns);w=np.linalg.eigvalsh(C@C.T);rank=int(sum(w>1e-10*w[-1]));rows.append(dict(input_dimension=d,quadratic_dictionary=15,root_coefficients=120,symmetric_quartic_dimension=d*(d+1)*(d+2)*(d+3)//24,rank=rank,nullity=120-rank,min_eigenvalue=float(w[0]),max_eigenvalue=float(w[-1])))
 out=dict(rows=rows,scope='Random orthonormal quadratic dictionaries, one fixed seed, numerical rank threshold1e-10. Not a theorem of generic rank or a measurement of trained native dictionaries. The d5dictionary spans all quadratic forms; larger dictionaries remain fixed at15features.')
 (P/'GRAM_NULLSPACE_SCALING_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
