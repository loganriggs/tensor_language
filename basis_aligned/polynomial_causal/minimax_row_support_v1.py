"""Weights-only minimax row support allocation; see preregistration."""
import hashlib
import json
import signal
import time
from datetime import datetime, timezone
import numpy as np
import torch
from retained_objective_context_v1 import Contexts, P


@torch.no_grad()
def main():
    out=P/'MINIMAX_ROW_SUPPORT_V1_RESULT.json'
    artifact=P/'MINIMAX_ROW_SUPPORT_V1_PROGRAM.pt'
    assert not out.exists() and not artifact.exists()
    torch.set_num_threads(2);signal.alarm(120);start=time.monotonic()
    model=Contexts()
    original=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    output,head=original['output'].double(),original['head'].double()
    core=torch.einsum('po,oih,bh->pib',torch.linalg.inv(output),model.tensor,torch.linalg.inv(head))
    rows=core.flatten(1)
    sorted_energy,order=rows.square().sort(dim=1,descending=True)
    # remaining[:, k] is the tail energy after keeping k entries in that row.
    remaining=torch.cat((sorted_energy.flip(1).cumsum(1).flip(1),
                         torch.zeros(12,1,dtype=torch.float64)),dim=1)
    budget=original['values'].numel()
    low,high=0.,float(remaining[:,0].max())
    def counts(cap):
        return (remaining>cap).sum(1)
    for _ in range(80):
        middle=(low+high)/2
        if int(counts(middle).sum())<=budget:high=middle
        else:low=middle
    kept=counts(high)
    while int(kept.sum())<budget:
        current=remaining[torch.arange(12),kept]
        current=current.masked_fill(kept==rows.shape[1],-1)
        kept[int(current.argmax())]+=1
    mask=torch.zeros_like(rows,dtype=torch.bool)
    for row,k in enumerate(kept.tolist()):mask[row,order[row,:k]]=True
    changed=dict(original,mask=torch.from_numpy(np.packbits(mask.flatten().numpy(),bitorder='little')),
                 values=rows[mask].float())
    torch.save(changed,artifact)
    fit=model.decode('MINIMAX_ROW_SUPPORT_V1')
    baseline=model.decode('SPARSE_INTERACTION_EXECUTOR_V1')
    def spectral(error):
        e=error.flatten(1)
        return float(torch.linalg.eigvalsh(e@e.T)[-1].clamp_min(0).sqrt())
    before,after=spectral(baseline-model.tensor),spectral(fit-model.tensor)
    frame=torch.einsum('op,pib,hb->oih',output,core,head)
    replay=float((frame-model.tensor).norm()/model.tensor.norm())
    frobenius=float((fit-model.tensor).norm()/model.tensor.norm())
    price=artifact.stat().st_size/(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt').stat().st_size
    result=dict(utc=datetime.now(timezone.utc).isoformat(),kept_per_row=kept.tolist(),
                deleted_row_norms=remaining[torch.arange(12),kept].sqrt().tolist(),
                original_spectral_error=before,fitted_spectral_error=after,
                spectral_relative_reduction=1-after/before,coefficient_relative_error=frobenius,
                frame_replay=replay,occupied_count=int(mask.sum()),original_occupied_count=budget,
                price_ratio=price,artifact_bytes=artifact.stat().st_size,
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                pred_a=replay<=1e-10 and int(mask.sum())==budget and abs(price-1)<=.01,
                pred_b=1-after/before>=.1,pred_c=frobenius<=.12,seconds=time.monotonic()-start,
                scope='Weights-only changed sparse support at fixed count/frames; absolute spectral-error certificate, no native transfer/adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))


if __name__=='__main__':main()
