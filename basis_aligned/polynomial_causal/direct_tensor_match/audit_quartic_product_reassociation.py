import json,time
import torch
from arithmetic_dag import DAG
from quartic_product_reassociation import compile_products
from cp_linear_reuse import compile_program,evaluate
from audit_conditional_residual_accounting import P,load

def baseline(indices,C):
 d=DAG(degree_limit=4);leaves=[d.input(i) for i in range(int(indices.max())+1)];roots=[]
 for a,b,c,e in indices.T.tolist():roots.append(d.product(d.product(leaves[a],leaves[b]),d.product(leaves[c],leaves[e])))
 return d,[d.linear(zip(roots,row.tolist())) for row in C]

def controls():
 rows=[];n=4
 cases=[('independent',[[4*i+j for j in range(4)] for i in range(n)],12),('fourth_powers',[[i]*4 for i in range(n)],8),('shared_quadratic',[[0,2+2*i,1,3+2*i] for i in range(n)],9),('shared_cubic',[[0,1,2,3+i] for i in range(n)],6),('duplicate_cancellation',[[0,1,2,3],[3,2,1,0]],0)]
 for name,terms,expected in cases:
  ix=torch.tensor(terms).T;C=torch.ones(1,len(terms),dtype=torch.float64)
  if name=='duplicate_cancellation':C[0,1]=-1
  d,out,meta=compile_products(ix,C);b,bo=baseline(ix,C);dim=int(ix.max())+1;assert d.polynomial(out,dim)==b.polynomial(bo,dim);assert meta['cost']['products']==expected,(name,meta)
  rows.append(dict(name=name,baseline=b.cost(bo),compiled=meta,exact_polynomial_equality=True))
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();x=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)['rows'][:256].double();rows=[]
 for seed in [1001,1002]:
  p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');original=compile_program(p['factors'],p['coefficients']);reuse=torch.load(P/f'CP_LINEAR_REUSE_SEED{seed}_V1.pt',weights_only=True);reuse={k:v.double() if k in ['bank','coefficients'] else v for k,v in reuse.items()}
  for name,q in [('original',original),('linear_reuse',reuse)]:
   d,out,meta=compile_products(q['indices'],q['coefficients']);z=x@q['bank'].T;ref=evaluate(q,x);pred=d.evaluate(out,z);err=float((pred-ref).norm()/ref.norm());assert err<1e-10
   b,bo=baseline(q['indices'],q['coefficients']);rows.append(dict(seed=seed,source=name,reader_count=len(q['bank']),reader_coefficients=q['bank'].numel(),replay_error=err,original_fixed_tree=b.cost(bo),compiled=meta));print(seed,name,meta['cost']['products'],err,flush=True)
 (P/'QUARTIC_PRODUCT_REASSOCIATION_V1.json').write_text(json.dumps(dict(controls=checks,rows=rows,seconds=time.monotonic()-start,scope='Exact reassociation over fixed reader IDs, with rational duplicate-root cancellation. Dense reader bank excluded from DAG prices and separately reported. Greedy root-order search, no global optimum/general algebraic identity claim. Native model fidelity inherited unchanged.'),indent=2)+'\n')
if __name__=='__main__':main()
