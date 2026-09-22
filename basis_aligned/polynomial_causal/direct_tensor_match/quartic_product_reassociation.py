"""Bounded exact reassociation of quartic products using the existing scalar DAG."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
from arithmetic_dag import DAG

@lru_cache(None)
def plans(key):
 if len(key)==1:return ({},)
 split=set()
 for mask in range(1,(1<<len(key))-1):
  a=tuple(v for j,v in enumerate(key) if mask&(1<<j));b=tuple(v for j,v in enumerate(key) if not mask&(1<<j));split.add(tuple(sorted((a,b))))
 result=[]
 for a,b in sorted(split):
  for pa,pb in product(plans(a),plans(b)):
   out=dict(pa);out.update(pb);out[key]=(a,b);result.append(out)
 return tuple(result)

def needed(plan,key,available):
 out=set()
 def visit(k):
  if len(k)==1 or k in available or k in out:return
  a,b=plan[k];visit(a);visit(b);out.add(k)
 visit(key);return out

def compile_products(indices,C):
 coefficients={};nout=len(C)
 for j,ids in enumerate(indices.T.tolist()):
  key=tuple(sorted(ids));values=coefficients.setdefault(key,[Fraction(0) for _ in range(nout)])
  for v in range(nout):values[v]+=Fraction(float(C[v,j]))
 coefficients={k:v for k,v in coefficients.items() if any(v)};keys=sorted(coefficients);frequency=Counter()
 for k in keys:
  for node in set().union(*(set(p) for p in plans(k))):frequency[node]+=1
 orders=[keys,list(reversed(keys)),sorted(keys,key=lambda k:(-sum(frequency[n] for n in set().union(*(set(p) for p in plans(k)))),k)),sorted(keys,key=lambda k:(len(set(k)),k))];best=None
 for order in orders:
  chosen={}
  for key in order:
   options=[]
   for i,p in enumerate(plans(key)):
    add=needed(p,key,chosen);options.append((len(add),-sum(frequency[n] for n in add),sum(len(n)==3 for n in add),i,add))
   _,_,_,i,add=min(options,key=lambda x:x[:4]);p=plans(key)[i]
   for node in add:chosen[node]=p[node]
  dag=DAG(degree_limit=4);nodes={}
  def build(k):
   if k not in nodes:
    if len(k)==1:nodes[k]=dag.input(k[0])
    else:
     a,b=chosen[k];nodes[k]=dag.product(build(a),build(b))
   return nodes[k]
  outputs=[dag.linear([(build(k),coefficients[k][v]) for k in keys if coefficients[k][v]]) for v in range(nout)];cost=dag.cost(outputs)
  depths={}
  for n in dag.reachable(outputs):depths[n]=max((depths[c] for c in dag.children(n)),default=0)+(dag.nodes[n][0]=='product')
  cost['product_depth']=max((depths[n] for n in outputs),default=0)
  score=(cost['products'],cost['product_depth'],cost['edges'])
  if best is None or score<best[0]:best=(score,dag,outputs,dict(cost=cost,unique_roots=len(keys),quadratic_nodes=sum(dag.degrees[n]==2 and dag.nodes[n][0]=='product' for n in dag.reachable(outputs)),cubic_nodes=sum(dag.degrees[n]==3 and dag.nodes[n][0]=='product' for n in dag.reachable(outputs))))
 return best[1:]
