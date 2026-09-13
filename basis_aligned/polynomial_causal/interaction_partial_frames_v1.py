"""All fixed HOSVD mode subsets, including literal adapter and sparse-storage costs."""
import itertools,json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build

P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    start=time.perf_counter()
    t,ids=build()
    bases=[]
    for axis in range(3):
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1)
        bases.append(torch.linalg.eigh(flat@flat.T)[1])
    n=t.numel(); energy=float(t.square().sum()); rows=[]
    for mask in itertools.product([False,True],repeat=3):
        core=t.clone(); adapters=0
        for axis,enabled in enumerate(mask):
            if enabled:
                core=torch.tensordot(bases[axis].T,core,dims=([1],[axis])).movedim(0,axis)
                adapters+=bases[axis].numel()
        assert abs(float(core.square().sum())/energy-1)<1e-10
        replay=core
        for axis,enabled in enumerate(mask):
            if enabled:replay=torch.tensordot(bases[axis],replay,dims=([1],[axis])).movedim(0,axis)
        err=float((replay-t).norm()/t.norm());assert err<1e-10
        cumulative=core.flatten().square().sort(descending=True).values.cumsum(0)
        cells=[]
        for tol in [.5,.2,.1,.05,.02]:
            k=int(torch.searchsorted(cumulative,(1-tol*tol)*energy))+1
            bitmap=4*adapters+4*k+(n+7)//8
            coo=4*adapters+8*k
            cells.append(dict(tolerance=tol,entries=k,retained_fraction=k/n,
                bitmap_bytes=bitmap,coo_bytes=coo,best_bytes_ratio=min(bitmap,coo)/(4*n)))
        rows.append(dict(rotated_axes=[i for i,v in enumerate(mask) if v],adapter_floats=adapters,
            reconstruction_error=err,curves=cells))
    result=dict(shape=list(t.shape),token_ids=ids,dense_bytes=4*n,frames=rows,
        seconds=time.perf_counter()-start,
        scope='Fixed weight-only frames, no rank truncation, no learned rotations or text fitting. Storage only; runtime and behavioral preservation unmeasured.')
    (P/'INTERACTION_PARTIAL_FRAMES_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    for j,tol in enumerate([.5,.2,.1,.05,.02]):
        best=min(rows,key=lambda r:r['curves'][j]['best_bytes_ratio'])
        print(tol,best['rotated_axes'],best['curves'][j])

if __name__=='__main__':main()
