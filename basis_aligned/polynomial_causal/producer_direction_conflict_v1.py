"""Test native directional conflict vs scalar overshoot; no fitting or new weights."""
import json
import time
from datetime import datetime, timezone
import torch
from retained_objective_context_v1 import Contexts, P
from shared_local_fineweb_v1 import digest


@torch.no_grad()
def main():
    out=P/'PRODUCER_DIRECTION_CONFLICT_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);start=time.monotonic()
    model=Contexts();path=P/'COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt'
    assert digest(path)==json.loads((P/'COMPOSED_LAST_BLOCK_STATES_V1_RESULT.json').read_text())['artifact_sha']
    cache=torch.load(path,weights_only=True)
    z=cache['linear_parts'][:,3].double();state=cache['linear_parts'][:,2].double()
    a=torch.linalg.lstsq(model.writer,(state-z).T).solution.T
    eps=torch.finfo(torch.float32).eps
    denominator=(state.square().mean(-1)+eps)*(cache['states'][:,2].square().mean(-1)+eps).sqrt()
    original=model.decode('SPARSE_INTERACTION_EXECUTOR_V1')
    fit=model.decode('UNIFORM_PRODUCER_GRADIENT_V1')
    def apply(t):
        return torch.einsum('oih,ni,nh->no',t,z,a)/denominator[:,None]
    e=apply(original-model.tensor);delta=apply(fit-original);direct=apply(fit-model.tensor)
    rows=[]
    for group in range(-1,5):
        sl=slice(None) if group==-1 else slice(group*24,(group+1)*24)
        b=float(e[sl].square().sum());g=float(2*(e[sl]*delta[sl]).sum());h=float(delta[sl].square().sum())
        target=float(direct[sl].square().sum())
        rows.append(dict(group=group,baseline_energy=b,linear_coefficient=g,quadratic_coefficient=h,
                         linear_over_baseline=g/b,quadratic_over_baseline=h/b,
                         direct_fit_energy=target,replay_relative_error=abs(b+g+h-target)/target,
                         every_positive_step_worsens=g>0 and h>=0))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
                pred_a=all(r['every_positive_step_worsens'] and r['replay_relative_error']<=1e-10 for r in rows),
                seconds=time.monotonic()-start,
                scope='Fixed-direction native-cache diagnostic, raw12output conditional energy. No fitting, step selection, new candidate or fresh/OOD evidence.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
