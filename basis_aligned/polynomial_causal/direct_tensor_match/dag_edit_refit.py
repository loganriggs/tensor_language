"""First approximate topology edit: delete a product globally, then refit.

Uses an explicit weighted function loss. Quadrature/probe weights are caller supplied.
Costs describe the fixed candidate topology throughout its continuous optimization.
"""
import torch
from trainable_arithmetic_dag import TrainableDAG

def complexity(price,penalties=(.005,1e-5,1e-5)):
 return sum(w*price[k] for w,k in zip(penalties,['products','additions','stored_coefficients']))

def fit_fixed(dag,outputs,x,y,weights,steps=500,rate=.01):
 model=TrainableDAG(dag,outputs);den=(weights[:,None]*y.square()).sum().clamp_min(1e-30);params=list(model.parameters());opt=torch.optim.Adam(params,lr=rate) if params else None;best=float('inf');state=None
 for step in range(steps+1):
  if opt:opt.zero_grad()
  loss=(weights[:,None]*(model(x)-y).square()).sum()/den
  if float(loss.detach())<best:best=float(loss.detach());state={k:v.detach().clone() for k,v in model.state_dict().items()}
  if opt and step<steps:loss.backward();opt.step()
  elif not opt:break
 model.load_state_dict(state);return model,best

def delete_product_candidates(dag,outputs):
 candidates=[];zero=dag.linear([])
 for n in dag.reachable(outputs):
  if dag.nodes[n][0]=='product':candidates.append((n,dag.replace(outputs,n,zero)))
 return candidates

def one_round(dag,outputs,x,y,weights,steps=500,penalties=(.005,1e-5,1e-5)):
 baseline,error=fit_fixed(dag,outputs,x,y,weights,steps=steps);initial_score=error+complexity(baseline.price(),penalties);current,out=baseline.export();rows=[];winner=None;best=initial_score
 for removed,candidate in delete_product_candidates(current,out):
  model,loss=fit_fixed(current,candidate,x,y,weights,steps=steps);price=model.price();score=loss+complexity(price,penalties);rows.append(dict(removed_node=removed,squared_relative_error=loss,price=price,objective=score))
  if score<best:best=score;winner=model
 return winner,dict(baseline_error=error,baseline_price=baseline.price(),baseline_objective=initial_score,candidates=rows,accepted=winner is not None,accepted_objective=best)
