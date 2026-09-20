"""Independent scalar-DAG replay and pricing of selected Gaussian quadratic fits."""
import json
from pathlib import Path
import torch
from export_centered_dag import build
from audit_centered_compact import evaluate
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    result=json.loads((P/'NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.json').read_text())
    archive=torch.load(P/'NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.pt',weights_only=True)
    panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
    targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets']
    records=[]
    for winner in result['winners']:
        key=tuple(winner[k] for k in ['metric','warmstart','optimizer']);s=archive['programs'][key];dag,outputs=build(s);cost=dag.cost(outputs);errors=[];replay=[]
        for panel,y in zip(panels,targets):
            x=panel['rows'].double();pred=evaluate(s,x)
            errors.append(float((pred-y).norm()/y.norm()))
            replay.append(float((dag.evaluate(outputs,x[:16])-pred[:16]).norm()/pred[:16].norm()))
        drift=max(abs(e-r['error']) for e,r in zip(errors,winner['panels']))
        assert cost['stored_coefficients']==34560 and cost['products']==4 and max(replay)<1e-10 and drift<1e-6
        records.append(dict(key=list(key),cost=cost,panel_errors=errors,graph_replay=max(replay),metric_drift=drift))
    out=dict(records=records,scope='Independent CPU scalar DAG replay of saved FP32 programs; same diagnostic panels, no new generalization evidence.')
    (P/'GAUSSIAN_QUADRATIC_ARCHIVE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
