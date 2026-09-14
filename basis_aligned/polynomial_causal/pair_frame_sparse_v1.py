"""Fixed pair sum/difference readers with normalized-row sparse support."""
import hashlib
import json
import signal
import time
import numpy as np
import torch
from retained_objective_context_v1 import Contexts,P
from sparse_interaction_executor_v1 import Executor


@torch.no_grad()
def main():
    out=P/'PAIR_FRAME_SPARSE_V1_RESULT.json';artifact=P/'PAIR_FRAME_SPARSE_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);signal.alarm(120);start=time.monotonic();model=Contexts()
    old=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    o=torch.zeros(12,12,dtype=torch.float64)
    for p in range(6):
        o[2*p:2*p+2,2*p:2*p+2]=torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64)/2**.5
    h=old['head'].double()
    core=torch.einsum('po,oih,bh->pib',torch.linalg.inv(o),model.tensor,torch.linalg.inv(h))
    scaled=core/core.flatten(1).norm(dim=1)[:,None,None]
    k=old['values'].numel();indices=scaled.flatten().abs().topk(k,sorted=False).indices
    mask=torch.zeros(core.numel(),dtype=torch.bool);mask[indices]=True
    program=dict(shape=list(core.shape),mask=torch.from_numpy(np.packbits(mask.numpy(),bitorder='little')),
                 values=core.flatten()[mask].float(),output=o.float(),head=old['head'])
    torch.save(program,artifact)
    fit=model.decode('PAIR_FRAME_SPARSE_V1');base=model.decode('SPARSE_INTERACTION_EXECUTOR_V1')
    generator=torch.Generator().manual_seed(170234000)
    z=torch.randn(32,1152,generator=generator);a=torch.randn(32,128,generator=generator)
    actual=Executor(program)(z,a).double();expected=torch.einsum('oih,ni,nh->no',fit,z.double(),a.double())
    replay=float((actual-expected).norm()/expected.norm())
    def errors(t):
        delta=t-model.tensor
        sums=delta[::2]+delta[1::2];diffs=delta[::2]-delta[1::2]
        return dict(total=float(delta.norm()/model.tensor.norm()),
                    sums=float(sums.norm()/(model.tensor[::2]+model.tensor[1::2]).norm()),
                    differences=float(diffs.norm()/(model.tensor[::2]-model.tensor[1::2]).norm()))
    before,after=errors(base),errors(fit)
    price=artifact.stat().st_size/(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt').stat().st_size
    result=dict(baseline=before,candidate=after,replay=replay,occupied_count=int(mask.sum()),
                price_ratio=price,artifact_bytes=artifact.stat().st_size,
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                pred_a=replay<=1e-6,pred_b=int(mask.sum())==k and abs(price-1)<=.01,
                pred_c=after['total']<=.15 and after['differences']<=.9*before['differences'],
                seconds=time.monotonic()-start,scope='Fixed pair-reader sparse graph, common headframe, normalized weight-only rows. No native-task preservation or adoption yet.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
