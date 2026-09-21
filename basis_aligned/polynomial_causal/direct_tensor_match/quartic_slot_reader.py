"""Exact first-slot linear map of a symmetric quartic's multilinear extension."""
import torch
from quartic_cp import directional

def first_slot(teacher,b,c,d):
 C,L2,R2,D1,L1,R1=teacher
 def pair(v,w):return .5*((v@L1.T)*(w@R1.T)+(w@L1.T)*(v@R1.T))@D1.T
 def term(u,v,w):
  t=pair(v,w);outer=.5*((C[None,:,:]*(t@R2.T)[:,None,:])@L2+(C[None,:,:]*(t@L2.T)[:,None,:])@R2);through=outer@D1
  return .5*((through*(u@R1.T)[:,None,:])@L1+(through*(u@L1.T)[:,None,:])@R1)
 return (term(b,c,d)+term(c,b,d)+term(d,b,c))/3

def controls():
 torch.set_num_threads(2);torch.manual_seed(173);dtype=torch.float64;dim=4;teacher=[torch.randn(3,5,dtype=dtype),torch.randn(5,4,dtype=dtype),torch.randn(5,4,dtype=dtype),torch.randn(4,6,dtype=dtype),torch.randn(6,dim,dtype=dtype),torch.randn(6,dim,dtype=dtype)];a,b,c,d=[torch.randn(11,dim,dtype=dtype) for _ in range(4)];actual=first_slot(teacher,b,c,d)
 indices=torch.cartesian_prod(*[torch.arange(dim) for _ in range(4)]);eye=torch.eye(dim,dtype=dtype);h=directional(*teacher,[eye[indices[:,i]] for i in range(4)]).reshape(dim,dim,dim,dim,3).permute(4,0,1,2,3);expected=torch.einsum('oijkl,nj,nk,nl->noi',h,b,c,d);replay=float((actual-expected).norm()/expected.norm());pred=torch.einsum('noi,ni->no',actual,a);truth=directional(*teacher,[a,b,c,d]);linear=float((pred-truth).norm()/truth.norm());assert max(replay,linear)<1e-12
 # Complete basis enumeration gives the EXACT mode Gram, unlike random sketches.
 triples=torch.cartesian_prod(*[torch.arange(dim) for _ in range(3)]);maps=first_slot(teacher,*[eye[triples[:,i]] for i in range(3)]).reshape(-1,dim);mode=h.permute(0,2,3,4,1).reshape(-1,dim);gram=float((maps.T@maps-mode.T@mode).norm()/(mode.T@mode).norm());assert gram<1e-12
 return dict(dense_slot_replay=replay,polarization_replay=linear,complete_basis_gram_replay=gram)
if __name__=='__main__':
 import json
 from pathlib import Path
 result=controls();Path(__file__).with_name('QUARTIC_SLOT_READER_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
