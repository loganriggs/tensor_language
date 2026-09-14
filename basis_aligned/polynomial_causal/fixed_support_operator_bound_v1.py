"""Weights-only fixed-support lower bound; registered board10:26UTC.

Let T=O C (I tensor H^T) in the output unfolding. With fixed zero entries,
each row norm of error core is at least its deleted-target norm.
Thus ||E_output_unfold||2 >= smin(O)smin(H) max(row deleted norms).
This is an absolute all-input bilinear-error bound, not native relative error.
"""
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
from retained_objective_context_v1 import Contexts, P


@torch.no_grad()
def main():
    out=P/'FIXED_SUPPORT_OPERATOR_BOUND_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);start=time.monotonic();model=Contexts()
    package=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True)
    output,head=package['output'].double(),package['head'].double()
    oi,hi=torch.linalg.inv(output),torch.linalg.inv(head)
    core=torch.einsum('po,oih,bh->pib',oi,model.tensor,hi)
    rebuilt=torch.einsum('op,pib,hb->oih',output,core,head)
    replay=float((rebuilt-model.tensor).norm()/model.tensor.norm())
    mask=torch.from_numpy(np.unpackbits(package['mask'].numpy(),bitorder='little',
                         count=model.tensor.numel()).copy()).bool().reshape(package['shape'])
    deleted=core.masked_fill(mask,0).flatten(1)
    row_norms=deleted.norm(dim=1)
    smin_o=float(torch.linalg.svdvals(output).min());smin_h=float(torch.linalg.svdvals(head).min())
    lower=smin_o*smin_h*float(row_norms.max())
    error=(model.decode('SPARSE_INTERACTION_EXECUTOR_V1')-model.tensor).flatten(1)
    spectrum=torch.linalg.eigvalsh(error@error.T).clamp_min(0).sqrt()
    upper=float(spectrum[-1])
    remaining=1-lower/upper
    result=dict(utc=datetime.now(timezone.utc).isoformat(),frame_replay=replay,
                frame_min_singular_values=[smin_o,smin_h],deleted_core_row_norms=row_norms.tolist(),
                current_output_unfolding_singular_values=spectrum.tolist(),
                lower_bound=lower,current_upper=upper,max_possible_relative_reduction=remaining,
                pred_a=replay<=1e-10 and lower<=upper*(1+1e-10),pred_b=remaining<=.01,
                seconds=time.monotonic()-start,
                scope='Fixed mask and literal stored frames, arbitrary coefficient refits. Bound on output-unfolding spectral norm, not injective-norm optimum or relative native effect. No fit/data/adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
