"""Numerical unfolding bounds on Tucker reader counts, with adapter costs."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build();total=float(t.square().sum())
    spectra=[];bases=[]
    for axis in range(3):
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1)
        eig,q=torch.linalg.eigh(flat@flat.T)
        spectra.append(eig.flip(0).clamp_min(0));bases.append(q.flip(1))
    rows=[]
    for tol in (.02,.05,.1):
        ranks=[int(torch.searchsorted(e.cumsum(0),(1-tol*tol)*total))+1 for e in spectra]
        # Separate necessary ranks are not jointly sufficient: execute truncation.
        core=t
        for axis,(q,r) in enumerate(zip(bases,ranks)):
            core=torch.tensordot(q[:,:r].T,core,dims=([1],[axis])).movedim(0,axis)
        residual=max(0.,1-float(core.square().sum())/total)**.5
        adapters=sum(int(q.shape[0])*r for q,r in zip(bases,ranks))
        cost=core.numel()+adapters
        # A mode retained at full rank can use its native coordinates for free.
        minimal_adapters=sum(int(q.shape[0])*r for q,r in zip(bases,ranks) if r<q.shape[0])
        rows.append(dict(tolerance=tol,necessary_mode_ranks=ranks,truncated_hosvd_error=residual,
                         dense_core_scalars=core.numel(),dense_adapter_scalars=adapters,
                         dense_tucker_scalars=cost,over_dense_tensor=cost/t.numel(),
                         omit_full_rank_rotations_scalars=core.numel()+minimal_adapters,
                         omit_full_rank_rotations_ratio=(core.numel()+minimal_adapters)/t.numel()))
    result=dict(shape=list(t.shape),token_ids=ids,rows=rows,seconds=time.perf_counter()-start,
        scope='Necessary per-mode ranks from exact numerical unfolding spectra under coefficient Frobenius norm and arbitrary independent input ports. Any multilinear shared-reader graph has unfolding rank no larger than its reader count. Not bounds for sparse full-rank arithmetic graphs, producer-constrained inputs or behavioral fidelity. Separate ranks not jointly sufficient; tested dense HOSVD cost includes adapters.')
    (P/'INTERACTION_READER_NODE_BOUNDS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
