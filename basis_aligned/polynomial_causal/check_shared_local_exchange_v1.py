"""Executed strongest local-optimum explanation: replace a private bank.

Same failed planted start, no access to true groups/banks in exchange proposal.
Select high-residual rows and test each bank replacement, then conditional polish.
"""
import json
import time
from pathlib import Path
import torch
from check_shared_local_subspaces_v1 import fixture
from shared_local_subspaces_v1 import bank,encode,fit,objective,parts,sweep


def main():
    torch.set_num_threads(2)
    x,g,k,r=fixture()
    tic=time.perf_counter()
    state,history=fit(x,g,k,r,2,max_sweeps=150)
    global_part,private_part=parts(state)
    residual=x-global_part-private_part
    indices=residual.square().sum(1).topk(32).indices
    proposed=bank((x-global_part)[indices],r)
    records=[]
    for j in range(k):
        locals_=[b.clone() for b in state['local_banks']]
        locals_[j]=proposed.clone()
        trial=encode(x,state['global_bank'],locals_)
        initial=objective(x,trial)
        for step in range(100):
            trial=sweep(x,trial)
            if objective(x,trial)<1e-10:break
        records.append(dict(replaced_group=j,proposal_loss=initial,
                            polished_loss=objective(x,trial),sweeps=step+1))
    result=dict(schema='shared.local.exchange.control.v1',original_loss=history[-1],
                replacements=records,best_loss=min(z['polished_loss'] for z in records),
                wall_seconds=time.perf_counter()-tic,
                scope='Post-failure algorithm discriminator, no retroactive repair of original four-start result. '
                'Only this planted fixture; native usefulness untested.')
    with Path(__file__).with_name('SHARED_LOCAL_EXCHANGE_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
