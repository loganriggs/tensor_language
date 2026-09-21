"""Conditional derivative w.r.t. supplied later h, holding earlier z fixed.
Not the end-to-end derivative: actual h and z are not independent model states.
"""
from pathlib import Path
import json
import torch
from compact_source_graph import source_reads as compact_reads
from shared_mixed_source_graph import source_reads as partial_reads
P=Path(__file__).parent
torch.set_num_threads(2)
eps=torch.finfo(torch.float32).eps

def value_jac(h,q,a,alpha,beta):
 s=(h.square().mean(-1)+eps).sqrt()[:,None]
 t=h@a-.5*q[:,::2]; A=t/s-alpha; B=q[:,1::2]/s-beta
 jac=B[:,:,None]*a.T[None]/s[:,:,None]-(t*B+A*q[:,1::2])[:,:,None]*h[:,None,:]/(h.shape[-1]*s[:,:,None]**3)
 return A*B,jac

def main():
 torch.manual_seed(9144);checks=[]
 for k in range(5):
  h=torch.randn(4,8,dtype=torch.float64,requires_grad=True);q=torch.randn(4,6,dtype=h.dtype);a=torch.randn(8,3,dtype=h.dtype);alpha=torch.randn(3,dtype=h.dtype);beta=torch.randn(3,dtype=h.dtype)
  v,j=value_jac(h,q,a,alpha,beta)
  actual=torch.stack([torch.autograd.grad(v[:,c].sum(),h,retain_graph=True)[0] for c in range(3)],1)
  error=float((j-actual).norm().detach()/actual.norm().detach());assert error<1e-12;checks.append(error)
 with torch.no_grad():
  d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=d['indices'];z=d['z'][ids];h=d['h'][ids]
  Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]);truthq=torch.einsum('ni,oij,nj->no',z,Q,z)
  a=torch.stack([p['a'] for p in d['pairs']],1);alpha=torch.stack([p['alpha'] for p in d['pairs']]);beta=torch.stack([p['beta'] for p in d['pairs']])
  truth,J=value_jac(h,truthq,a,alpha,beta)
  meta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());p=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[meta['winner']]
  models={'partial512':partial_reads(z,p),'compact399':compact_reads(z,torch.load(P/'COMPACT_GROUP_FROZEN_V1.pt',weights_only=True))}
  if (P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').exists():
   m=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text());programs=torch.load(P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)
   models.update({key:compact_reads(z,programs[key]) for key in m['winners'].values()})
  rows=[]
  for name,q in models.items():
   v,j=value_jac(h,q,a,alpha,beta)
   rows.append(dict(name=name,component_errors=((v-truth).norm(dim=0)/(truth-truth.mean(0)).norm(dim=0)).tolist(),later_input_jacobian_errors=((j-J).square().sum((0,2))/J.square().sum((0,2))).sqrt().tolist()))
 out=dict(autograd_checks=checks,rows=rows,scope='Descriptive conditional later-input derivatives on opened448 states. Complements earlier-input fixed-h derivatives; neither is the composed upstream Jacobian, fresh validation, or an adoption gate.')
 (P/'LATER_INPUT_RESPONSE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
