"""Greedy exact fixed-dictionary group-edge deletions with LS refitting.
Each group is the two output edges to one native component. Keep at least one
component per product. Not a global optimum over graph topologies.
"""
from pathlib import Path
import torch,json,time
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
fit=json.loads((P/'SHARED_PRODUCT_NATIVE_FIT_V1.json').read_text());p=torch.load(P/'SHARED_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True)[fit['winner']];d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
V=torch.linalg.solve(d['inverse_root'],p['shared_reader']);V=V/V.norm(dim=0);K=(V.T@V).square();rhs=torch.einsum('ir,oij,jr->ro',V,d['teacher'],V);total=d['teacher'].square().sum();r=V.shape[1]
inverse=torch.linalg.inv(K);full=inverse@rhs;initial=float(total-(full*rhs).sum());states=[]
for group in range(3):states.append(dict(ids=torch.arange(r),inverse=inverse.clone(),weights=full[:,2*group:2*group+2].clone()))
counts=torch.full((r,),3,dtype=torch.long);increments=0.;history=[]
for step in range(2*r):
 candidates=[]
 for group,state in enumerate(states):
  costs=state['weights'].square().sum(1)/state['inverse'].diag();costs[counts[state['ids']]<=1]=torch.inf;idx=int(costs.argmin());candidates.append((float(costs[idx]),group,idx))
 cost,group,index=min(candidates);assert cost>=0 and cost<float('inf');state=states[group];ids=state['ids'];inv=state['inverse'];w=state['weights'];keep=torch.arange(len(ids))!=index
 col=inv[keep,index];diag=inv[index,index]
 states[group]=dict(ids=ids[keep],inverse=inv[keep][:,keep]-torch.outer(col,col)/diag,weights=w[keep]-col[:,None]/diag*w[index])
 counts[ids[index]]-=1;increments+=cost
 if (step+1)%128==0 or step+1==2*r:history.append(dict(deleted_pair_edges=step+1,remaining_pair_edges=3*r-step-1,predicted_coefficient_error=((initial+increments)/float(total))**.5))
private=torch.zeros_like(full);normal=[];coefficient_replay=[]
for group,state in enumerate(states):
 ids=state['ids'];direct=torch.linalg.solve(K[ids[:,None],ids[None,:]],rhs[ids,2*group:2*group+2]);private[ids,2*group:2*group+2]=direct
 coefficient_replay.append(float((direct-state['weights']).norm()/direct.norm()));normal.append(float((K[ids[:,None],ids[None,:]]@direct-rhs[ids,2*group:2*group+2]).norm()/rhs[ids,2*group:2*group+2].norm()))
hat=torch.einsum('ro,ir,jr->oij',private,V,V);error=float((hat-d['teacher']).norm()/d['teacher'].norm());predicted=((initial+increments)/float(total))**.5
assert abs(error-predicted)<1e-8 and max(coefficient_replay)<1e-8 and bool((counts==1).all())
naive=json.loads((P/'SHARED_PRODUCT_REUSE_AUDIT_V1.json').read_text())
out=dict(parent_winner=fit['winner'],source_products=r,initial_shared_coefficient_error=(initial/float(total))**.5,naive_private_coefficient_error=naive['private']['coefficient_error'],greedy_private_coefficient_error=error,ratio_to_shared=error/((initial/float(total))**.5),private_pair_counts=[len(s['ids']) for s in states],edge_curve=history,predicted_dense_loss_replay=abs(error-predicted),schur_weight_replay=max(coefficient_replay),normal_equation_replay=max(normal),seconds=time.perf_counter()-start,scope='Greedy output-pair edge deletion with analytic exact refits for fixed square atoms. Stronger than largest-energy assignment, not global optimality or a semantic result. Parent fidelity still fails.')
(P/'GREEDY_PRIVATE_PRODUCT_GRAPH_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(dict(unit_readers=V,private_weights=private),P/'GREEDY_PRIVATE_PRODUCT_GRAPH_V1.pt');print(json.dumps(out,indent=2))
