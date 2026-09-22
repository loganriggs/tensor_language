"""Exact-integer tests: each product adds at most one new quadratic generator."""
import json,random
from pathlib import Path
D=3;ZERO=(0,)*D;MON=[tuple(int(k==i)+int(k==j) for k in range(D)) for i in range(D) for j in range(i,D)]
def add(a,b):
 c=a.copy()
 for k,v in b.items():c[k]=c.get(k,0)+v
 return {k:v for k,v in c.items() if v}
def scale(a,s):return {k:v*s for k,v in a.items() if v*s}
def mul(a,b):
 c={}
 for i,u in a.items():
  for j,v in b.items():
   k=tuple(x+y for x,y in zip(i,j));c[k]=c.get(k,0)+u*v
 return {k:v for k,v in c.items() if v}
def check(seed):
 rng=random.Random(seed);gens=[];nodes=[]
 def append(poly,c,lin,coeff):
  q=[sum(v*gens[g][i] for g,v in coeff.items()) for i in range(len(MON))]
  assert q==[poly.get(m,0) for m in MON]
  assert c==poly.get(ZERO,0)
  assert lin==[poly.get(tuple(int(i==j) for i in range(D)),0) for j in range(D)]
  nodes.append((poly,c,lin,coeff))
 append({ZERO:1},1,[0]*D,{})
 for i in range(D):
  m=tuple(int(k==i) for k in range(D));append({m:1},0,list(m),{})
 # Constants, sharing, repeated products, and higher degree nodes occur naturally.
 for step in range(30):
  allowed=[i for i,n in enumerate(nodes) if max(map(sum,n[0]),default=0)<=4]
  i=rng.choice(allowed);j=rng.choice(allowed);a,ac,al,aq=nodes[i];b,bc,bl,bq=nodes[j]
  if step%3:
   w=rng.choice([-2,-1,1,2]);append(add(a,scale(b,w)),ac+w*bc,[u+w*v for u,v in zip(al,bl)],add(aq,scale(bq,w)))
  else:
   gen=[al[i]*bl[j]+(al[j]*bl[i] if i!=j else 0) for i in range(D) for j in range(i,D)]
   coeff=add(scale(bq,ac),scale(aq,bc));coeff[len(gens)]=1;gens.append(gen)
   append(mul(a,b),ac*bc,[ac*v+bc*u for u,v in zip(al,bl)],coeff)
 # Explicit high-degree cancellation: f-f leaves exactly zero in both representations.
 p,c,l,q=nodes[-1];append(add(p,scale(p,-1)),0,[0]*D,add(q,scale(q,-1)))
 return dict(seed=seed,products=len(gens),checked_nodes=len(nodes),maximum_degree=max(max(map(sum,n[0]),default=0) for n in nodes))
if __name__=='__main__':
 rows=[check(s) for s in range(50)]
 result=dict(pass_all=True,arithmetic='Exact Python integers; full polynomial expansion independent of degree-two generator recurrence',rows=rows)
 Path(__file__).with_name('QUADRATIC_DAG_BOUND_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),'circuits',sum(r['checked_nodes'] for r in rows),'node identities')
