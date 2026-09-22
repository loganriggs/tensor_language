"""Numerical finite-response preservation for exact product graph exports."""
import hashlib,json,time
from fractions import Fraction
import torch
from arithmetic_dag import DAG
from cp_linear_reuse import evaluate
from audit_conditional_residual_accounting import P,SCALE

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();exports=json.loads((P/'QUARTIC_PRODUCT_DAG_EXPORT_V1.json').read_text());data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].float();pairfile=P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json';pairs=torch.tensor(json.loads(pairfile.read_text())['pairs']).T;rows=[]
 for record in exports['rows']:
  seed=record['seed'];path=P/f'QUARTIC_PRODUCT_DAG_SEED{seed}_V1.graph.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'];graph=json.loads(path.read_text());source=P/graph['reader_bank_source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==graph['reader_bank_sha256'];p=torch.load(source,weights_only=True);d=DAG(degree_limit=4);refs=[]
  for node in graph['nodes']:
   if node['op']=='input':v=d.input(node['coordinate'])
   elif node['op']=='product':v=d.product(*(refs[i] for i in node['inputs']))
   else:v=d.linear([(refs[k],Fraction(a,b)) for k,a,b in node['terms']])
   refs.append(v)
  outputs=[refs[k] for k in graph['outputs']];before=[];after=[]
  for xx in x.split(1024):before.append(evaluate(p,xx).double()/SCALE);after.append(d.evaluate(outputs,xx@p['bank'].T).double()/SCALE)
  before=torch.cat(before);after=torch.cat(after);assert torch.isfinite(before).all() and torch.isfinite(after).all();delta=after-before;response=before[pairs[1]]-before[pairs[0]];error=delta[pairs[1]]-delta[pairs[0]];fv=(delta.square().sum(0)/before.square().sum(0)).sqrt();fr=(error.square().sum(0)/response.square().sum(0)).sqrt();value=float(delta.norm()/before.norm());change=float(error.norm()/response.norm());passed=value<1e-5 and change<1e-5 and float(fv.max())<1e-4 and float(fr.max())<1e-4
  rows.append(dict(seed=seed,states=len(x),pairs=pairs.shape[1],relative_value_difference=value,relative_response_difference=change,feature_value_differences=fv.tolist(),feature_response_differences=fr.tolist(),pred_preservation=passed,graph_sha256=record['sha256'],source_sha256=graph['reader_bank_sha256']));print(seed,value,change,float(fv.max()),float(fr.max()),passed,flush=True)
 (P/'PRODUCT_DAG_RESPONSE_REPLAY_V1.json').write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start,pair_receipt_sha256=hashlib.sha256(pairfile.read_bytes()).hexdigest(),scope='Float32graphvsfloat32sourcecandidate; CPU1024statebatches. Samefixedbank/coefficientvalues, changed multiplication association. Nativecandidateaccuracy inherited, not measured here; no native/OOD/causal promotion.'),indent=2)+'\n')
if __name__=='__main__':main()
