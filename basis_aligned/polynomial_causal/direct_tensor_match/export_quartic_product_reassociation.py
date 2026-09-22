"""Export reachable product DAGs using the existing rational graph format."""
import hashlib,json
from fractions import Fraction
import torch
from arithmetic_dag import DAG
from quartic_product_reassociation import compile_products
from cp_linear_reuse import evaluate
from audit_conditional_residual_accounting import P

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);x=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)['rows'][:256].double();rows=[]
 for seed in [1001,1002]:
  source=P/f'CP_LINEAR_REUSE_SEED{seed}_V1.pt';p=torch.load(source,weights_only=True);p['bank']=p['bank'].double();p['coefficients']=p['coefficients'].double();d,out,meta=compile_products(p['indices'],p['coefficients']);live=d.reachable(out);mapping={old:new for new,old in enumerate(live)};nodes=[]
  for n in live:
   v=d.nodes[n]
   if v[0]=='input':nodes.append(dict(op='input',coordinate=v[1]))
   elif v[0]=='product':nodes.append(dict(op='product',inputs=[mapping[v[1]],mapping[v[2]]]))
   else:nodes.append(dict(op='linear',terms=[[mapping[r],c.numerator,c.denominator] for r,c in v[1]]))
  graph=dict(nodes=nodes,outputs=[mapping[n] for n in out],reader_bank_source=source.name,reader_bank_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_interface='z=x@source.bank.T; graph outputs physical16coordinates; source.writer retained for residual-space mapping',scope='Exact parent graph rewrite, not new native fidelity. Rational JSON review representation, not deployment byte compression.')
  path=P/f'QUARTIC_PRODUCT_DAG_SEED{seed}_V1.graph.json';path.write_text(json.dumps(graph,separators=(',',':'))+'\n');restored=json.loads(path.read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==restored['reader_bank_sha256'];dd=DAG(degree_limit=4);refs=[]
  for node in restored['nodes']:
   if node['op']=='input':v=dd.input(node['coordinate'])
   elif node['op']=='product':v=dd.product(*(refs[i] for i in node['inputs']))
   else:v=dd.linear([(refs[k],Fraction(a,b)) for k,a,b in node['terms']])
   refs.append(v)
  outputs=[refs[k] for k in restored['outputs']];z=x@p['bank'].T;actual=dd.evaluate(outputs,z);expected=evaluate(p,x);replay=float((actual-expected).norm()/expected.norm());assert replay<1e-10 and dd.cost(outputs)==d.cost(out)
  rows.append(dict(seed=seed,relative_replay=replay,cost=meta['cost'],reader_floats=p['bank'].numel(),json_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=graph['reader_bank_sha256']));print(seed,meta['cost']['products'],replay,flush=True)
 (P/'QUARTIC_PRODUCT_DAG_EXPORT_V1.json').write_text(json.dumps(dict(rows=rows,scope='Two exact product DAG exports, fixed externally referenced dense reader bank and writer. JSON integer ratios are review format, no byte/runtime improvement asserted.'),indent=2)+'\n')
if __name__=='__main__':main()
