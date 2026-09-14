"""Three fixed circuit branches: exact Boolean interaction decomposition.

Mask bits S=1, R=2, O=4 indicate removed writes. These operations analyze
interventions on frozen weights; no surrogate weights are fitted to text.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import torch


def design():
    return torch.tensor([[float((row&col)==col) for col in range(8)] for row in range(8)],dtype=torch.float64)


def coefficients(values):
    """values[mask,...]; return constant, singles, pairs and triple coefficients."""
    if values.shape[0]!=8:raise ValueError('All eight corners are required')
    c=values.clone()
    for bit in [1,2,4]:
        for mask in range(8):
            if mask&bit:c[mask]-=c[mask^bit]
    return c


def reconstruct(c):
    return (design().to(c)@c.reshape(8,-1)).reshape_as(c)


def identifiability_control():
    p=Path(__file__).resolve().parent;out=p/'EVEN_VALUE_FACTORIAL_V1_IDENTIFIABILITY.json'
    assert not out.exists()
    d=design();known=[0,1,2,3,4,7];missing=[5,6]
    nulls=torch.linalg.solve(d,torch.eye(8,dtype=torch.float64)[:,missing])
    # A counterexample changes hidden pair/triple interactions but leaves all
    # previously measured corners unchanged, so linearity cannot fill them in.
    observed_change=d[known]@nulls
    test=torch.arange(24,dtype=torch.float64).reshape(8,3).square()
    recovered=reconstruct(coefficients(test))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),known_masks=known,missing_masks=missing,
                known_design_rank=int(torch.linalg.matrix_rank(d[known])),
                full_design_rank=int(torch.linalg.matrix_rank(d)),null_directions=nulls.tolist(),
                observed_change_maxabs=float(observed_change.abs().max()),
                mobius_control_maxabs=float((test-recovered).abs().max()),
                scope='Existing six measured corners cannot identify S/O, R/O and triple interactions separately. Two additional native corners are required; no missing result is imputed.')
    result['pred_a']=result['known_design_rank']==6 and result['full_design_rank']==8 and result['observed_change_maxabs']==0 and result['mobius_control_maxabs']==0
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':identifiability_control()
