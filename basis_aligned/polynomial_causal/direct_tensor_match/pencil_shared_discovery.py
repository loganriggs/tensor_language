"""Enumerate pair-pencil blocks, then enforce cross-consumer input sharing.

Exact low-rank diagnostic. Architecture widths are given; planted directions are
not used. Requires a regular diagonalizable pair pencil; enumeration is
exponential in block count. Do not apply to native 352-wide pairs unboundedly.
"""
import itertools,torch
from pairwise_exact_supports import supports
from pairwise_reader_graph import GROUPS
from quadratic_pair_blocks import compile_pair

def intersection(U,V,tol=1e-8):
 left,s,_=torch.linalg.svd(U.T@V,full_matrices=False)
 return U@left[:,s>1-tol]

def discover(target,edge_width,max_blocks=18):
 spaces=supports(target);edges,pairs=spaces[:3],spaces[3:];candidates=[];stats=[]
 for j,(a,b) in enumerate(GROUPS):
  U=pairs[j];Q=U.T@target[2*j:2*j+2]@U;compiled=compile_pair(Q[0],Q[1]);F=U@compiled['input_transform'];blocks=compiled['blocks']
  if len(blocks)>max_blocks:raise ValueError('Enumeration budget exceeded')
  options=[];tested=0
  for mask in itertools.product((False,True),repeat=len(blocks)):
   if sum(len(v) for v,on in zip(blocks,mask) if on)!=2*edge_width:continue
   tested+=1;cols=[i for v,on in zip(blocks,mask) if on for i in v];other=[i for v,on in zip(blocks,mask) if not on for i in v]
   common=torch.linalg.qr(F[:,cols],mode='reduced').Q
   if intersection(common,edges[a]).shape[1]>=edge_width and intersection(common,edges[b]).shape[1]>=edge_width:
    options.append(dict(common=common,private=F[:,other],selected_blocks=[i for i,on in enumerate(mask) if on]))
  stats.append(dict(pair=j,block_count=len(blocks),enumerated_subsets=tested,candidates=len(options),compiler=compiled['diagnostics']))
  candidates.append(options)
 if any(len(c)!=1 for c in candidates):return dict(success=False,stats=stats,reason='Shared-subspace candidate missing or ambiguous')
 common=[c[0]['common'] for c in candidates];private=[c[0]['private'] for c in candidates];recovered=[intersection(common[a],common[b]) for a,b in ((0,1),(0,2),(1,2))]
 if any(p.shape[1]!=edge_width for p in recovered):return dict(success=False,stats=stats,reason='Recovered edge width differs from architecture')
 return dict(success=True,stats=stats,bases=recovered,private=private)


def propose_approximate(target,edge_width,private_width,max_blocks=18):
 """Budgeted support truncation and scored compatibility, not exact discovery."""
 pair_width=2*edge_width+private_width;pairs=[]
 for j in range(3):
  T=target[2*j:2*j+2];_,U=torch.linalg.eigh((T@T).sum(0));pairs.append(U[:,-pair_width:])
 common=[];private=[];stats=[]
 for j in range(3):
  U=pairs[j];Q=U.T@target[2*j:2*j+2]@U;compiled=compile_pair(Q[0],Q[1]);F=U@compiled['input_transform'];blocks=compiled['blocks']
  if len(blocks)>max_blocks:raise ValueError('Enumeration budget exceeded')
  options=[]
  for mask in itertools.product((False,True),repeat=len(blocks)):
   if sum(len(v) for v,on in zip(blocks,mask) if on)!=2*edge_width:continue
   cols=[i for v,on in zip(blocks,mask) if on for i in v];other=[i for v,on in zip(blocks,mask) if not on for i in v]
   C=torch.linalg.qr(F[:,cols],mode='reduced').Q
   score=sum(float(edge_width-torch.linalg.svdvals(C.T@pairs[k])[:edge_width].square().sum()) for k in range(3) if k!=j)
   options.append((score,C,F[:,other]))
  if not options:raise ValueError('No block grouping at requested width')
  options.sort(key=lambda x:x[0]);best=options[0];common.append(best[1]);private.append(best[2]);stats.append(dict(pair=j,score=best[0],next_score=options[1][0] if len(options)>1 else None,candidate_count=len(options)))
 bases=[]
 for a,b in ((0,1),(0,2),(1,2)):
  L,_,Rh=torch.linalg.svd(common[a].T@common[b],full_matrices=False);balanced=common[a]@L[:,:edge_width]+common[b]@Rh.T[:,:edge_width];bases.append(torch.linalg.qr(balanced,mode='reduced').Q)
 return dict(bases=bases,private=private,stats=stats)
