"""Five exact sharing structures, near-equality and opaque-boundary controls."""
import json,time
from pathlib import Path
from fractions import Fraction
import torch
from arithmetic_dag import DAG
from dag_functional_sharing import best_equivalence_edit

def case(name):
 d=DAG();a,b,c,e=[d.input(i) for i in range(4)]
 sq=lambda x:d.product(x,x)
 lin=lambda *terms:d.linear(terms)
 if name=='polarization':
  q=lin((sq(lin((a,1),(b,1))),1),(sq(a),-1),(sq(b),-1));r=lin((d.product(a,b),2))
 elif name=='quartic_cancellation':
  aa,bb=sq(a),sq(b);q=lin((sq(lin((aa,1),(bb,1))),1),(sq(lin((aa,1),(bb,-1))),-1));r=lin((d.product(aa,bb),4))
 elif name=='regrouped_quartic':
  q=d.product(d.product(a,b),d.product(c,e));r=d.product(d.product(a,c),d.product(b,e))
 elif name=='affine_square':
  one=d.constant();q=lin((sq(lin((a,1),(one,1))),1),(sq(a),-1),(one,-1));r=lin((a,2))
 elif name=='signed_quadratic':
  q=lin((d.product(lin((a,1),(b,1)),lin((a,1),(b,-1))),1));r=lin((sq(a),1),(sq(b),-1))
 return d,[d.product(q,c),d.product(r,e)]

def main():
 torch.set_num_threads(2);start=time.monotonic();rows=[]
 for name in ['polarization','quartic_cancellation','regrouped_quartic','affine_square','signed_quadratic']:
  d,out=case(name);truth=d.polynomial(out,4);before=d.cost(out);edits=0
  for _ in range(20):
   new,report=best_equivalence_edit(d,out)
   if new is None:break
   assert d.polynomial(new,4)==truth;out=new;edits+=1
  after=d.cost(out);assert edits and after['products']<before['products']
  rows.append(dict(family=name,before=before,after=after,edits=edits,exact_identity=True))
 # A tiny nonzero polynomial must not disappear, regardless of probe precision.
 d=DAG();a,b,c=[d.input(i) for i in range(3)];q=d.product(a,b);r=d.linear([(q,1),(d.product(c,c),Fraction(1,10**20))]);out=[q,r];new,_=best_equivalence_edit(d,out);assert new is None
 # Dense leaf becomes opaque; identities above that SAME boundary still work.
 d=DAG();inputs=[d.input(i) for i in range(20)];a=d.linear([(x,i+1) for i,x in enumerate(inputs)]);b=inputs[0];q=d.linear([(d.product(d.linear([(a,1),(b,1)]),d.linear([(a,1),(b,-1)])),1)]);r=d.linear([(d.product(a,a),1),(d.product(b,b),-1)]);out=[q,r];truth=d.polynomial(out,20);new,report=best_equivalence_edit(d,out,max_terms=8);assert new is not None and report['opaque_nodes']>0 and d.polynomial(new,20)==truth
 # A descendant may equal its ancestor, but replacing the ancestor with it
 # must never create a cycle. A hard degree limit remains respected.
 d=DAG(degree_limit=2);a=d.input(0);q=d.product(a,a);twice=d.linear([(q,2)]);r=d.linear([(twice,Fraction(1,2))]);out=[q,r];truth=d.polynomial(out,1);new,cycle_report=best_equivalence_edit(d,out);assert new is not None and d.polynomial(new,1)==truth
 assert all(item['target'] not in d.reachable([item['replacement']]) for item in cycle_report['proposals'])
 result=dict(records=rows,near_equality_rejected=True,opaque_boundary_identity=True,acyclic_and_degree_controls=True,seconds=time.monotonic()-start,scope='Exact rational frozen-function rewrites with global reachable cost. Five planted structures, including quartic regrouping and cancellation. Not globally optimal arithmetic synthesis; parameter tying after a rewrite changes subsequent fitting freedom.')
 Path(__file__).with_name('DAG_FUNCTIONAL_SHARING_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
