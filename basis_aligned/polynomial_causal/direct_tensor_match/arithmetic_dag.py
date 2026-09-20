"""Scalar arithmetic DAG with exact structural sharing and globally priced rewrites.
Rational coefficients preserve exact input float values; polynomial expansion is toy-only.
"""
from collections import defaultdict
from fractions import Fraction

class DAG:
 def __init__(self):self.nodes=[];self.index={}
 def intern(self,key):
  if key not in self.index:self.index[key]=len(self.nodes);self.nodes.append(key)
  return self.index[key]
 def input(self,index):return self.intern(('input',index))
 def linear(self,terms):
  values=defaultdict(Fraction)
  for node,coefficient in terms:values[node]+=Fraction(coefficient)
  terms=tuple(sorted((n,c) for n,c in values.items() if c))
  if len(terms)==1 and terms[0][1]==1:return terms[0][0]
  return self.intern(('linear',terms))
 def product(self,a,b):return self.intern(('product',*sorted((a,b))))
 def children(self,node):
  key=self.nodes[node]
  return [] if key[0]=='input' else ([n for n,c in key[1]] if key[0]=='linear' else list(key[1:]))
 def reachable(self,outputs):
  seen=set();stack=list(outputs)
  while stack:
   n=stack.pop()
   if n not in seen:seen.add(n);stack.extend(self.children(n))
  return sorted(seen)
 def cost(self,outputs):
  result=dict(products=0,additions=0,stored_coefficients=0,edges=0,nodes=0)
  for n in self.reachable(outputs):
   key=self.nodes[n];op=key[0];result['nodes']+=1;result['edges']+=len(self.children(n))
   if op=='product':result['products']+=1
   if op=='linear':
    result['additions']+=max(0,len(key[1])-1);result['stored_coefficients']+=sum(abs(c)!=1 for _,c in key[1])
  return result
 def evaluate(self,outputs,x):
  import torch
  values={}
  for n in self.reachable(outputs):
   key=self.nodes[n]
   if key[0]=='input':values[n]=x[...,key[1]]
   elif key[0]=='product':values[n]=values[key[1]]*values[key[2]]
   elif key[1]:
    refs,coeff=zip(*key[1]);values[n]=torch.stack([values[r] for r in refs],-1)@x.new_tensor([float(c) for c in coeff])
   else:values[n]=x.new_zeros(x.shape[:-1])
  return torch.stack([values[n] for n in outputs],-1)
 def replace(self,outputs,target,replacement):
  memo={target:replacement}
  def visit(n):
   if n in memo:return memo[n]
   key=self.nodes[n]
   if key[0]=='input':out=n
   elif key[0]=='product':
    a,b=visit(key[1]),visit(key[2]);out=n if (a,b)==key[1:] else self.product(a,b)
   else:
    terms=[(visit(r),c) for r,c in key[1]];out=n if tuple(terms)==key[1] else self.linear(terms)
   memo[n]=out;return out
  return [visit(n) for n in outputs]
 def factorizations(self,node):
  key=self.nodes[node]
  if key[0]!='linear':return []
  groups=defaultdict(list)
  for term,c in key[1]:
   product=self.nodes[term]
   if product[0]=='product':
    a,b=product[1:]
    groups[a].append((term,b,c))
    if a!=b:groups[b].append((term,a,c))
  candidates=[]
  for shared,terms in groups.items():
   if len(terms)<2:continue
   used={n for n,_,_ in terms};inner=self.linear([(other,c) for _,other,c in terms]);factored=self.product(shared,inner);replacement=self.linear([(n,c) for n,c in key[1] if n not in used]+[(factored,1)]);candidates.append((shared,replacement))
  return candidates
 def polynomial(self,outputs,dimensions):
  """Exact small-test oracle; NEVER expand a native model tensor here."""
  values={}
  for n in self.reachable(outputs):
   key=self.nodes[n];p=defaultdict(Fraction)
   if key[0]=='input':e=[0]*dimensions;e[key[1]]=1;p[tuple(e)]=Fraction(1)
   elif key[0]=='linear':
    for ref,c in key[1]:
     for power,v in values[ref].items():p[power]+=c*v
   else:
    for a,ca in values[key[1]].items():
     for b,cb in values[key[2]].items():p[tuple(x+y for x,y in zip(a,b))]+=ca*cb
   values[n]={k:v for k,v in p.items() if v}
  return [values[n] for n in outputs]

def score(cost,weights=(1.,.01,.001)):
 return sum(w*cost[k] for w,k in zip(weights,['products','additions','stored_coefficients']))

def best_factor_edit(dag,outputs,targets=None,weights=(1.,.01,.001)):
 baseline=dag.cost(outputs);best=score(baseline,weights);chosen=None;proposals=[]
 for n in list(dag.reachable(outputs) if targets is None else targets):
  for shared,replacement in dag.factorizations(n):
   new=dag.replace(outputs,n,replacement);cost=dag.cost(new);value=score(cost,weights);proposals.append(dict(target=n,shared=shared,cost=cost,score=value))
   if value<best:best=value;chosen=new
 return chosen,proposals
