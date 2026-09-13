"""Exact entry-pruning curves in native and full HOSVD interaction frames.

No rank reduction; all full orthogonal adapters and packed edge IDs are priced.
This is a fixed-frame baseline, not optimized sparse Tucker or causal validation.
"""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build

P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);started=time.perf_counter();t,ids=build()
    bases=[]
    for axis in range(3):
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1)
        bases.append(torch.linalg.eigh(flat@flat.T)[1].flip(1))
    core=torch.einsum('ao,oid->aid',bases[0].T,t)
    core=torch.einsum('bi,aid->abd',bases[1].T,core)
    core=torch.einsum('cd,abd->abc',bases[2].T,core)
    replay=torch.einsum('oa,abc->obc',bases[0],core)
    replay=torch.einsum('ib,obc->oic',bases[1],replay)
    replay=torch.einsum('dc,oic->oid',bases[2],replay)
    error=float((replay-t).norm()/t.norm());assert error<1e-10
    total=float(t.square().sum());results=[]
    dense_bytes=4*t.numel()
    for name,coefficients,adapter in [('native',t,0),('full_hosvd',core,sum(q.numel() for q in bases))]:
        energies=coefficients.flatten().square().sort(descending=True).values
        assert abs(float(energies.sum())/total-1)<1e-10
        cumulative=energies.cumsum(0);cells=[]
        for tolerance in [.5,.2,.1,.05,.02]:
            count=int(torch.searchsorted(cumulative,(1-tolerance**2)*total))+1
            actual=max(0,1-float(cumulative[count-1]/total))**.5
            stored=4*adapter+8*count
            cells.append(dict(relative_error_ceiling=tolerance,retained_entries=count,
                retained_fraction=count/t.numel(),actual_relative_error=actual,
                total_bytes=stored,bytes_over_dense=stored/dense_bytes))
        results.append(dict(frame=name,adapter_floats=adapter,curves=cells))
    result=dict(schema='interaction.sparse.core.baseline.v1',tensor_shape=list(t.shape),
        token_ids=ids,dense_tensor_bytes=dense_bytes,full_reconstruction_error=error,
        frames=results,wall_seconds=time.perf_counter()-started,
        scope='Best entry support in each fixed orthogonal frame for selected-output mixed T. '
        'One uint32 packed edge ID plus FP32 coefficient per entry; adapters fully priced. '
        'No learned rotations, native-domain preservation or whole-model compression claim.')
    with (P/'INTERACTION_SPARSE_CORE_BASELINE_V1_RESULT.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
