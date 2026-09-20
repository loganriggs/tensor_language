"""Exact degree0..4 expansion about a fixed input center; no model linearization error."""
import torch

def degree_terms(teacher,x,center):
 C,L,R,D,A,B=teacher;delta=x-center;am=center@A.T;bm=center@B.T;ad=delta@A.T;bd=delta@B.T
 h0=(am*bm)@D.T;h1=(am*bd+ad*bm)@D.T;h2=(ad*bd)@D.T
 left=[h0@L.T,h1@L.T,h2@L.T];right=[h0@R.T,h1@R.T,h2@R.T]
 terms=[]
 for degree in range(5):
  value=sum(left[i]*right[degree-i] for i in range(3) if 0<=degree-i<3)@C.T
  if degree==0:value=value.expand(len(x),-1)
  terms.append(value)
 return torch.stack(terms,1)


def check():
 import math
 from quartic_cp import directional
 torch.manual_seed(1846);teacher=[torch.randn(*shape,dtype=torch.float64) for shape in [(2,4),(4,3),(4,3),(3,5),(5,6),(5,6)]];x=torch.randn(17,6,dtype=torch.float64);mu=torch.randn(6,dtype=torch.float64);terms=degree_terms(teacher,x,mu);delta=x-mu;center=mu.expand_as(x);ref=torch.stack([math.comb(4,k)*directional(*teacher,[delta]*k+[center]*(4-k)) for k in range(5)],1);relative=float((terms-ref).norm()/ref.norm());full=directional(*teacher,[x]*4);replay=float((terms.sum(1)-full).norm()/full.norm());assert max(relative,replay)<1e-12;return dict(degree_polarization_error=relative,full_replay_error=replay)
if __name__=='__main__':print(check())
