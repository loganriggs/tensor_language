"""Homogeneous shared residual-direction correction with explicit FP32 executor."""
import hashlib
import json
import signal
import time
from datetime import datetime,timezone
import torch
from retained_objective_context_v1 import Contexts,P
from sparse_interaction_executor_v1 import Executor


class ProjectedExecutor:
    def __init__(self,program):
        self.base=Executor(program)
        self.q=program['residual_direction']
        self.k=program['direction_correction']
        assert self.q.shape==(1152,) and self.k.shape==(12,128)
    def __call__(self,z,a):
        return self.base(z,a)+(z@self.q)[:,None]*(a@self.k.T)
    def resident_bytes(self):
        return self.base.resident_bytes()+4*(self.q.numel()+self.k.numel())


@torch.no_grad()
def main():
    out=P/'PROJECTED_CARRIER_V1_RESULT.json';artifact=P/'PROJECTED_CARRIER_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);signal.alarm(120);start=time.monotonic();model=Contexts()
    b=torch.load(P/'PUSHFORWARD_CARRIER_V1_PROVENANCE.pt',weights_only=True)['carrier'].double()
    q=b/b.norm();base=model.decode('SPARSE_INTERACTION_EXECUTOR_V1')
    k=torch.einsum('oih,i->oh',model.tensor-base,q)
    package=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    changed=dict(package,residual_direction=q.float(),direction_correction=k.float())
    torch.save(changed,artifact)
    fit=base+torch.einsum('i,oh->oih',changed['residual_direction'].double(),changed['direction_correction'].double())
    executor=ProjectedExecutor(changed)
    generator=torch.Generator().manual_seed(170233000)
    z=torch.randn(32,1152,generator=generator);a=torch.randn(32,128,generator=generator)
    actual=executor(z,a).double();exact=torch.einsum('oih,ni,nh->no',fit,z.double(),a.double())
    replay=float((actual-exact).norm()/exact.norm())
    zeros=bool((executor(z*0,a)==0).all() and (executor(z,a*0)==0).all())
    ratios=[]
    for i in range(6):
        old=(base-model.tensor)[2*i]-(base-model.tensor)[2*i+1]
        new=(fit-model.tensor)[2*i]-(fit-model.tensor)[2*i+1]
        ratios.append(float(torch.linalg.svdvals(new)[0]/torch.linalg.svdvals(old)[0]))
    panels=[]
    for prefix,sl in [('COMPOSED_LAST_BLOCK_STATES_V1',slice(None)),('MINIMAX_FRESH_CACHE_V1',slice(24,None))]:
        path=P/(prefix+'_ARTIFACT.pt')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==json.loads((P/(prefix+'_RESULT.json')).read_text())['artifact_sha']
        cache=torch.load(path,weights_only=True)
        z=cache['linear_parts'][sl,3].double();state=cache['linear_parts'][sl,2].double()
        a=torch.linalg.lstsq(model.writer,(state-z).T).solution.T
        eps=torch.finfo(torch.float32).eps
        denom=(state.square().mean(-1)+eps)*(cache['states'][sl,2].square().mean(-1)+eps).sqrt()
        values=[torch.einsum('oih,ni,nh->no',t-model.tensor,z,a)/denom[:,None] for t in (base,fit)]
        old,new=[float(v.square().sum()) for v in values]
        panels.append(dict(prefix=prefix,baseline_energy=old,candidate_energy=new,relative_improvement=1-new/old))
    price=artifact.stat().st_size/(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt').stat().st_size
    result=dict(utc=datetime.now(timezone.utc).isoformat(),execution_replay=replay,zero_ports=zeros,
                scalar_contrast_norm_ratios=ratios,coefficient_error=float((fit-model.tensor).norm()/model.tensor.norm()),
                panels=panels,price_ratio=price,artifact_bytes=artifact.stat().st_size,
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),resident_bytes=executor.resident_bytes(),
                pred_a=replay<=1e-6 and zeros,pred_b=price<=1.01,
                pred_c=all(p['relative_improvement']>=.05 for p in panels),pred_d=max(ratios)<=1+1e-6,
                seconds=time.monotonic()-start,scope='Weights-generated fixed residual-direction exact correction; homogeneous bilinear program, no native-data fitting or adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
