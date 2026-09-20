import json
from pathlib import Path
import torch
from export_centered_dag import build
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);source=torch.load(P/'CENTERED_OUTPUT_BLOCK_V1.pt',weights_only=True);result=json.load(open(P/'CENTERED_OUTPUT_BLOCK_V1.json'));panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];rows=[]
 for rank,program in source['programs'].items():
  s={k:v.double() for k,v in program.items()};d,out=build(s);errors=[];replays=[]
  for panel,y in zip(panels,targets):
   x=panel['rows'].double();z=x-s['mu'];prediction=s['constant']+(z@s['linear_reader'].T)@s['linear_writer'].T+((z@s['quadratic_left'].T)*(z@s['quadratic_right'].T))@s['Z'].T@s['W'].T;errors.append(float((prediction-y).norm()/y.norm()));replays.append(float((d.evaluate(out,x[:32])-prediction[:32]).norm()/prediction[:32].norm()))
  original=next(r for r in result['records'] if r['rank']==rank);drift=max(abs(a-b) for a,b in zip(errors,original['full_quartic_errors']));assert max(replays)<1e-10 and drift<1e-5;assert d.cost(out)['stored_coefficients']==sum(t.numel() for t in program.values());rows.append(dict(rank=rank,graph_replay=max(replays),archive_metric_drift=drift,cost=d.cost(out),full_quartic_errors=errors))
 (P/'OUTPUT_BLOCK_ARCHIVE_AUDIT_V1.json').write_text(json.dumps(dict(records=rows,scope='IndependentFP32archive replay, including sharedoutputfactor representation andliteralprice.'),indent=2)+'\n');print(rows)
if __name__=='__main__':main()
