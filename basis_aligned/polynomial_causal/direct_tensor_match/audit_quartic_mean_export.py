"""Replay archived Gaussian-corrected programs and their literal scalar DAGs."""
import json
from pathlib import Path
import torch
from export_shared_output_dag import build
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);source=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);result=json.load(open(P/'NATIVE_QUARTIC_MEAN_V1.json'));panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];i,j=torch.triu_indices(4,4);rows=[]
 for rank,program in source['programs'].items():
  s={k:v.double() for k,v in program.items()};d,out=build(s['U'],s['V'],s['C']) if rank==10 else build(s['U'],s['V'],s['W'],s['Z']);one=d.constant();out=[d.linear([(n,1),(one,float(c))]) for n,c in zip(out,s['constant'])];price=d.cost(out);assert price['stored_coefficients']==sum(t.numel() for t in s.values());assert price['products']==26;errors=[];replay=[]
  for panel,y in zip(panels,targets):
   x=panel['rows'].double();q=((x@s['U'].flatten(0,1).T)*(x@s['V'].flatten(0,1).T)).reshape(len(x),4,4).sum(2);phi=q[:,i]*q[:,j];prediction=(phi@s['C'].T if rank==10 else phi@s['Z'].T@s['W'].T)+s['constant'];errors.append(float((prediction-y).norm()/y.norm()));replay.append(float((d.evaluate(out,x[:32])-prediction[:32]).norm()/prediction[:32].norm()))
  expected=next(r for r in result['records'] if r['rank']==rank);drift=max(abs(e-r['gaussian_error']) for e,r in zip(errors,expected['panels']));assert max(replay)<1e-10 and drift<1e-5;rows.append(dict(rank=rank,cost=price,archived_panel_errors=errors,graph_replay_relative_error=max(replay),archive_metric_drift=drift,maximum_degree=max(d.degrees[n] for n in out)))
 out=dict(records=rows,scope='Gaussianbias archive only; literal reachable DAG includingbias. FP32storedparameters,FP64replay. Commonvocabularyframe excluded.');(P/'QUARTIC_MEAN_EXPORT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
