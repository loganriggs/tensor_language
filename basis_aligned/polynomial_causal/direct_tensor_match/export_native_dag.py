import json,time
from pathlib import Path
import torch
from arithmetic_dag import DAG,best_factor_edit
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();source=torch.load(P/'NATIVE_MIXED_ROOT_V2.pt',weights_only=True);key=('adam',.005,1);s=source['students'][key];U,V,S,C=[s[k].double() for k in ['U','V','mixing','C']];support=s['support'];d=DAG();inputs=[d.input(i) for i in range(1152)];quadratics=[]
 for g in range(4):
  products=[]
  for k in range(4):
   a=d.linear(zip(inputs,U[g,k].tolist()));b=d.linear(zip(inputs,V[g,k].tolist()));products.append(d.product(a,b))
  quadratics.append(d.linear([(n,1) for n in products]))
 mixed=[d.linear(zip(quadratics,S[g].tolist())) for g in range(4)];pairs=torch.triu_indices(4,4).T;roots=[d.product(mixed[int(pairs[k,0])],mixed[int(pairs[k,1])]) for k in support];outputs=[d.linear(zip(roots,row.tolist())) for row in C];x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][:32].double();q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);q=q@S.T;expected=(q[:,pairs[support,0]]*q[:,pairs[support,1]])@C.T;actual=d.evaluate(outputs,x);replay=float((actual-expected).norm()/expected.norm());cost=d.cost(outputs);accepted,proposals=best_factor_edit(d,outputs,targets=outputs[:8]);assert replay<1e-10
 # Export only reachable nodes, including explicit references; exact coefficient ratios.
 live=d.reachable(outputs);mapping={old:new for new,old in enumerate(live)};nodes=[]
 for n in live:
  v=d.nodes[n]
  if v[0]=='input':nodes.append(dict(op='input',coordinate=v[1]))
  elif v[0]=='product':nodes.append(dict(op='product',inputs=[mapping[v[1]],mapping[v[2]]]))
  else:nodes.append(dict(op='linear',terms=[[mapping[r],c.numerator,c.denominator] for r,c in v[1]]))
 graph=dict(nodes=nodes,outputs=[mapping[n] for n in outputs],source_key=list(key),teacher_scale=source['teacher_scale'],scope='Scalar original-input graph in exactreducedoutputframe, normalizedteacherunits. RationalIRreviewformat, notdeploymentbytecompression. No monosemantic labels inferred.')
 (P/'NATIVE_DAG_EXPORT_V1.graph.json').write_text(json.dumps(graph,separators=(',',':'))+'\n');reloaded=json.loads((P/'NATIVE_DAG_EXPORT_V1.graph.json').read_text());restored=DAG();references=[]
 from fractions import Fraction
 for node in reloaded['nodes']:
  if node['op']=='input':new=restored.input(node['coordinate'])
  elif node['op']=='product':new=restored.product(*(references[k] for k in node['inputs']))
  else:new=restored.linear([(references[k],Fraction(a,b)) for k,a,b in node['terms']])
  references.append(new)
 restored_outputs=[references[k] for k in reloaded['outputs']];serialized_replay=float((restored.evaluate(restored_outputs,x)-actual).norm()/actual.norm());assert serialized_replay<1e-12 and restored.cost(restored_outputs)==cost
 predictions=dict(pred_a_replay=replay<1e-10,pred_b_price=cost['products']==24 and cost['stored_coefficients']==46096,pred_c_global=accepted is None);result=dict(source_key=list(key),cost=cost,replay_relative_error=replay,serialized_roundtrip_error=serialized_replay,local_proposal_count=len(proposals),proposal_product_counts=sorted({p['cost']['products'] for p in proposals}),best_proposed_score=min((p['score'] for p in proposals),default=None),accepted_edit=accepted is not None,predictions=predictions,seconds=time.perf_counter()-start,scope='Exact graph export and bounded first8output factoring screen. Null doesnot implygloballyminimalgraph; noapproximateedits/refitting/search acrossallnodes yet.');(P/'NATIVE_DAG_EXPORT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
