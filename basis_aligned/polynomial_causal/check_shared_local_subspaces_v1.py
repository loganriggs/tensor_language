"""Unknown-bank/group planted controls, not a native fit or global guarantee."""
import json
import time
from pathlib import Path
import torch
from shared_local_subspaces_v1 import encode, fit, objective, parts, sweep


def fixture():
    torch.manual_seed(9318)
    dt = torch.float64
    n,d,g,k,r = 320,20,2,4,2
    q = torch.linalg.qr(torch.randn(d,d,dtype=dt)).Q.T
    labels = torch.arange(n) % k
    # The common parent is stronger but private variation remains substantial.
    x = 2 * torch.randn(n,g,dtype=dt) @ q[:g]
    for j in range(k):
        ix = labels == j
        x[ix] += torch.randn(int(ix.sum()),r,dtype=dt) @ q[g+j*r:g+(j+1)*r]
    return x,g,k,r


def main():
    torch.set_num_threads(2)
    x,g,k,r = fixture()
    dt = x.dtype
    tic = time.perf_counter()
    records = []
    for seed in range(4):
        state,history = fit(x,g,k,r,seed,max_sweeps=150)
        audit = sweep(x,state)
        pred = sum(parts(state))
        # Joint projection should not depend on nonsingular bank coordinates.
        mix = torch.tensor([[1.,.3],[.2,1.1]],dtype=dt)
        remapped = encode(x,mix@state['global_bank'],[mix@b for b in state['local_banks']])
        recode_error = float((sum(parts(remapped))-pred).norm()/x.norm())
        records.append(dict(seed=seed,initial_loss=history[0],final_loss=history[-1],
                            sweeps=len(history)-1,max_loss_increase=max(b-a for a,b in zip(history,history[1:])),
                            next_sweep_gain=history[-1]-objective(x,audit),
                            recoding_prediction_error=recode_error,
                            counts=torch.bincount(state['labels'],minlength=k).tolist(),
                            recovered=history[-1]<1e-8))
        print(json.dumps(records[-1]),flush=True)
    assert max(z['recoding_prediction_error'] for z in records)<1e-10
    result=dict(schema='shared.local.solver.control.v1',starts=records,
                recovered_starts=sum(z['recovered'] for z in records),
                wall_seconds=time.perf_counter()-tic,
                scope='No planted labels/banks passed to solver. Exact-data recovery is stronger than loss plateau; '
                'one-more-sweep gain is not a global or differential stationarity certificate.')
    path=Path(__file__).with_name('SHARED_LOCAL_SUBSPACES_V1_CONTROL.json')
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
