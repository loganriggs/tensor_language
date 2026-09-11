"""Fixed nuclear penalty coefficient, varying conditional output rank.

Do not add isolated gains across blocks. Each holds all other blocks fixed.
"""
import json
from pathlib import Path
import torch
from conditional_block_svd_v1 import shrink


def spectrum_solution(s, rank, alpha=.0025):
    t = shrink(s, alpha*rank, outputs=rank)
    # Constant ||R||² omitted in both residual and penalized expressions.
    residual = t.square().sum() - 2*(t*s[:rank]).sum()
    loss = residual + alpha*t.sum().square()
    return dict(residual=float(residual),loss=float(loss),
                active=int((t>0).sum()),singular=t.tolist())


def control():
    torch.set_default_dtype(torch.float64)
    s = torch.tensor([10.,9.,8.,7.,6.,5.,4.,3.,2.,1.])
    rows=[spectrum_solution(s,r) for r in (1,2,4,8,10)]
    assert all(b['loss']<=a['loss'] for a,b in zip(rows,rows[1:]))
    actual=shrink(s,.01)
    reference=rows[2]
    err=abs(reference['loss']-float((actual.square()-2*actual*s[:4]).sum()+.0025*actual.sum().square()))
    assert err<1e-12
    # Changing rank must not silently keep the old per-component penalty.
    wrong=shrink(s,.01,outputs=8)
    correct=torch.tensor(rows[3]['singular'])
    assert float((wrong-correct).norm())>1e-3
    report=dict(rank4_replay_error=err,capacity_monotonic=True,
                live_penalty_tripwire=float((wrong-correct).norm()),rows=rows)
    with Path(__file__).with_name('BLOCK_CONDITIONAL_CAPACITY_V1_CONTROL.json').open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':control()
