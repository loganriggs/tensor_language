import json
from pathlib import Path
from arithmetic_dag import DAG,best_factor_edit
P=Path(__file__).resolve().parent

def main():
 results=[]
 # User example, output directions kept separate: four products -> two.
 d=DAG();a,b,c,e=[d.input(i) for i in range(4)];out=[d.linear([(d.product(a,b),1),(d.product(a,c),1)]),d.linear([(d.product(e,b),1),(d.product(e,c),1)])];truth=d.polynomial(out,4);before=d.cost(out);edits=0
 while True:
  new,_=best_factor_edit(d,out)
  if new is None:break
  assert d.polynomial(new,4)==truth;out=new;edits+=1
 after=d.cost(out);assert before['products']==4 and after['products']==2;assert after['additions']==1;results.append(dict(case='shared_linear_combination',before=before,after=after,edits=edits,exact_polynomial_replay=True))
 # Cross-depth products are interned once, including reversed operand order.
 d=DAG();a,b,c,e=[d.input(i) for i in range(4)];ab=d.product(a,b);assert ab==d.product(b,a);out=[d.product(ab,c),d.product(ab,e)];assert d.cost(out)['products']==3;results.append(dict(case='cross_depth_product_reuse',cost=d.cost(out),unfolded_tree_products=4))
 # Other consumers preserve the allegedly eliminated intermediates.
 d=DAG();a,b,c=[d.input(i) for i in range(3)];ab=d.product(a,b);ac=d.product(a,c);summed=d.linear([(ab,1),(ac,1)]);out=[summed,ab,ac];new,proposals=best_factor_edit(d,out);assert new is None and proposals and all(p['cost']['products']==3 for p in proposals);results.append(dict(case='other_consumers_prevent_saving',before=d.cost(out),rejected_proposals=proposals))
 # Signed cancellation is exact and zero coefficients do not remain stored.
 d=DAG();a,b,c=[d.input(i) for i in range(3)];ab=d.product(a,b);ac=d.product(a,c);out=[d.linear([(ab,1),(ac,1),(ab,-1)])];assert out==[ac];assert d.polynomial(out,3)==d.polynomial([ac],3);results.append(dict(case='signed_cancellation',cost=d.cost(out)))
 # Repeated squaring shares both children, instead of expanding a tree.
 d=DAG();x=d.input(0);out=x
 for _ in range(5):out=d.product(out,out)
 assert d.cost([out])['products']==5 and d.polynomial([out],1)==[{(32,):1}];results.append(dict(case='repeated_squaring',cost=d.cost([out]),unfolded_tree_products=31))
 data=dict(checks=results,scope='Five exact small-circuit controls for structural DAG sharing, distributive rewrites and global reachable-cost acceptance. Rational coefficient oracle, no approximate merging/general optimality/native discovery claim. Unit signed coefficients implicit; edge references counted separately.')
 (P/'ARITHMETIC_DAG_CHECK_V1.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
