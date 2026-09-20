"""Differentiable fixed DAG. Shared nodes have one parameter owner.

Topology is frozen during fitting. Unit signs can remain fixed structural constants;
when made trainable they are counted as stored coefficients, even at value +/-1.
No automatic re-interning of learned nodes (which would change parameter tying).
"""
import torch
from torch import nn
from arithmetic_dag import DAG

class TrainableDAG(nn.Module):
 def __init__(self,dag,outputs,learn_units=False):
  super().__init__();self.dag=dag;self.outputs=list(outputs);self.live=dag.reachable(outputs);self.coefficients=nn.ParameterDict();self.learn_units=learn_units;self.masks={}
  for n in self.live:
   key=dag.nodes[n]
   if key[0]=='linear':
    value=torch.tensor([float(c) for _,c in key[1]],dtype=torch.float64)
    mask=torch.tensor([learn_units or abs(c)!=1 for _,c in key[1]],dtype=torch.bool)
    self.register_buffer(f'fixed_{n}',value);self.register_buffer(f'mask_{n}',mask)
    if mask.any():self.coefficients[str(n)]=nn.Parameter(value[mask].clone())
 def linear_weights(self,n):
  value=getattr(self,f'fixed_{n}');mask=getattr(self,f'mask_{n}')
  return value.masked_scatter(mask,self.coefficients[str(n)]) if str(n) in self.coefficients else value
 def forward(self,x):
  values={}
  for n in self.live:
   key=self.dag.nodes[n]
   if key[0]=='input':values[n]=x[...,key[1]]
   elif key[0]=='constant':values[n]=x.new_ones(x.shape[:-1])
   elif key[0]=='product':values[n]=values[key[1]]*values[key[2]]
   elif key[1]:values[n]=torch.stack([values[k] for k,c in key[1]],-1)@self.linear_weights(n)
   else:values[n]=x.new_zeros(x.shape[:-1])
  return torch.stack([values[n] for n in self.outputs],-1)
 def price(self):
  price=self.dag.cost(self.outputs).copy()
  # All parameters occupy storage even when optimization puts a value at zero/unit.
  price['trainable_coefficients']=sum(p.numel() for p in self.parameters())
  price['stored_coefficients']=sum(getattr(self,f'fixed_{n}').numel() if self.learn_units else sum(abs(c)!=1 for _,c in self.dag.nodes[n][1]) for n in self.live if self.dag.nodes[n][0]=='linear')
  return price
 def export(self):
  target=DAG(degree_limit=self.dag.degree_limit);mapping={}
  for n in self.live:
   key=self.dag.nodes[n]
   if key[0]=='input':new=target.input(key[1])
   elif key[0]=='constant':new=target.constant()
   elif key[0]=='product':new=target.product(mapping[key[1]],mapping[key[2]])
   else:new=target.linear([(mapping[k],float(c)) for (k,_),c in zip(key[1],self.linear_weights(n).detach().cpu())])
   mapping[n]=new
  return target,[mapping[n] for n in self.outputs]
