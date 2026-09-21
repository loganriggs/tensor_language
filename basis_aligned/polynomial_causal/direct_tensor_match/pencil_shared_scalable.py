"""Bounded beam proposals for shared pair-pencil groups, with exact final scoring.

Dynamic programming uses additive block-support overlap as a heuristic. It
retains at most beam candidates per width, then reranks complete candidates by
principal-angle compatibility. No exhaustive native-width enumeration.
"""
import torch
from quadratic_pair_blocks import compile_pair

def propose(target,edge_width,private_width,beam=4):
 width=2*edge_width+private_width;pairs=[]
 for j in range(3):
  T=target[2*j:2*j+2];_,U=torch.linalg.eigh((T@T).sum(0));pairs.append(U[:,-width:])
 common=[];private=[];stats=[]
 for j in range(3):
  U=pairs[j];T=U.T@target[2*j:2*j+2]@U;compiled=compile_pair(T[0],T[1]);F=U@compiled['input_transform'];blocks=compiled['blocks'];dp={0:[(0.,())]};visited=0
  for index,block in enumerate(blocks):
   Q=torch.linalg.qr(F[:,block],mode='reduced').Q;value=sum(float((Q.T@pairs[k]).square().sum()) for k in range(3) if k!=j);updated={w:list(v) for w,v in dp.items()}
   for current,items in dp.items():
    new_width=current+len(block)
    if new_width>2*edge_width:continue
    for score,chosen in items:updated.setdefault(new_width,[]).append((score+value,chosen+(index,)));visited+=1
   dp={w:sorted(v,key=lambda x:x[0],reverse=True)[:beam] for w,v in updated.items()}
  options=[]
  for additive,chosen in dp.get(2*edge_width,[]):
   cols=[i for k in chosen for i in blocks[k]];other=[i for k,block in enumerate(blocks) if k not in chosen for i in block];C=torch.linalg.qr(F[:,cols],mode='reduced').Q
   score=sum(float(edge_width-torch.linalg.svdvals(C.T@pairs[k])[:edge_width].square().sum()) for k in range(3) if k!=j)
   options.append((score,C,F[:,other],additive,chosen))
  if not options:raise ValueError('No beam proposal at shared width')
  options.sort(key=lambda x:x[0]);best=options[0];common.append(best[1]);private.append(best[2]);stats.append(dict(pair=j,score=best[0],additive_score=best[3],selected_blocks=list(best[4]),candidate_count=len(options),dp_transitions=visited,block_count=len(blocks),beam=beam))
 bases=[]
 for a,b in ((0,1),(0,2),(1,2)):
  L,_,Rh=torch.linalg.svd(common[a].T@common[b],full_matrices=False);B=common[a]@L[:,:edge_width]+common[b]@Rh.T[:,:edge_width];bases.append(torch.linalg.qr(B,mode='reduced').Q)
 return dict(bases=bases,private=private,stats=stats)
