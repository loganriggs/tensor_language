"""Exact shared coordinates and a fixed three-parent merge test, no fitting."""
from pathlib import Path
import torch,json
from sparse_path_stability_atlas_v1 import digest
from quartic_frozen_native_score_v2 import score
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
 binding=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 source=P/'PRODUCER_METRIC_OUTER32_V1_PROGRAM.pt';receipt=json.loads((P/'PRODUCER_METRIC_OUTER32_V1_RESULT.json').read_text());assert digest(source)==receipt['artifact_sha256']
 p=torch.load(source,weights_only=True);overlap=json.loads((P/'PAIRED_PRODUCER_SHARED_SPACE_V1.json').read_text());assert overlap['artifact_sha256']==digest(source)
 cross=torch.tensor(overlap['cross_gram']);u,c,vt=torch.linalg.svd(cross);v=vt.T;n=3
 a,b=p['output_readers'];aa=a@u;bb=b@v;plus=((1+c[:n])/2).sqrt();minus=((1-c[:n])/2).sqrt()
 common=(aa[:,:n]+bb[:,:n])/(2*plus);difference=(aa[:,:n]-bb[:,:n])/(2*minus)
 parents=torch.cat([common,difference,aa[:,n:],bb[:,n:]],1)
 adapters=[]
 for branch,rotation in [(0,u),(1,v)]:
  mapping=torch.zeros(64,32);mapping[:n,:n]=torch.diag(plus);mapping[n:2*n,:n]=torch.diag(minus)*(1 if branch==0 else -1)
  start=2*n if branch==0 else 2*n+32-n;mapping[start:start+32-n,n:]=torch.eye(32-n)
  adapters.append(mapping@rotation.T)
 errors=[float((parents@adapter-original).norm()/original.norm()) for adapter,original in zip(adapters,(a,b))]
 gram=torch.eye(64);start=2*n;other=start+32-n;gram[start:other,other:]=torch.diag(c[n:]);gram[other:,start:other]=torch.diag(c[n:])
 ge,gv=torch.linalg.eigh(gram);gs=(gv*ge.sqrt())@gv.T;merged_adapters=[m.clone() for m in adapters]
 for m in merged_adapters:m[n:2*n]=0
 paired=[]
 for adapter,merged,mu in zip(adapters,merged_adapters,p['outer_weights']):
  exact=gs@((adapter*mu)@adapter.T)@gs;approx=gs@((merged*mu)@merged.T)@gs
  paired.append(float((exact-approx).norm()/exact.norm()))
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 l,r,d=[state['transformer.h.16.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')]
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double();producer=((x@l.T)*(x@r.T))@d.T*p['producer_scale'];reads=producer@parents
 ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports'];den=ports['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
 def execute(maps):
  scalar=torch.stack([((reads@m).square()*mu).sum(-1) for m,mu in zip(maps,p['outer_weights'])],1)
  return scalar@p['output_writers'].T/den[:,None]
 exact=execute(adapters);merged=execute(merged_adapters);old=p['native_write'];errors.append(float((exact-old).norm()/old.norm()))
 change=float((merged-old).norm()/old.norm());ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows'];prior=next(r for r in json.loads((P/'QUARTIC_GROUP_LIFTED_NATIVE_V1.json').read_text())['effects'] if r['name']=='lifted')
 # Restore the native float32 default before the shared tail arithmetic.
 torch.set_default_dtype(torch.float32)
 effects=score([ref,old,merged],ports['pre']+ports['native_output'],rows,state['lm_head.weight'].float(),prior)
 retained=[i for i in range(64) if not n<=i<2*n]
 artifact=P/'PAIRED_COMMON_DIFFERENCE_GRAPH_V1_PROGRAM.pt'
 torch.save(dict(source_sha256=digest(source),parents=parents,rotations=torch.stack([u,v]),plus=plus,minus=minus,shared_count=n,outer_weights=p['outer_weights'],output_writers=p['output_writers'],producer_scale=p['producer_scale'],merged_native_write=merged),artifact)
 result=dict(pred_a=max(errors)<=1e-8 and effects['pred_a'],pred_b=max(paired)<=.05,pred_c=change<=.01,pred_d=effects['pred_b'] and effects['pred_c'],
  exact_errors=errors,paired_branch_relative_errors=paired,native_write_change=change,effects=effects,principal_cosines=c[:n].tolist(),
  baseline_fitted_floats=76096,exact_graph_fitted_floats=78150,merged_graph_fitted_floats=74694,shared_native_parent_floats=15925248,
  artifact_sha256=digest(artifact),scope='Exact common/difference re-encoding plus fixed deletion of three difference parents. Semantic reuse and stable identification untested; adapter costs charged, no fitting or fresh validation.')
 out=P/'PAIRED_COMMON_DIFFERENCE_GRAPH_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='effects'},indent=2));assert result['pred_a']
if __name__=='__main__':main()
