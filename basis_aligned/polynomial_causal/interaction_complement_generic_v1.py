"""Post-result generic matrix control for compact-complement storage gains."""
import json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);start=time.perf_counter();rows=[]
    for seed in (61401,61402,61403):
        gen=torch.Generator().manual_seed(seed)
        x=torch.randn(1152,1536,dtype=torch.float64,generator=gen)
        eig=torch.linalg.eigvalsh(x@x.T).clamp_min(0);total=float(eig.sum())
        for tol in (.02,.05,.1):
            k=int(torch.searchsorted(eig.cumsum(0),tol**2*total,right=True))
            adapter=k*1152-k*(k-1)//2;core=(1152-k)*1536
            rows.append(dict(seed=seed,tolerance=tol,discarded=k,
                             storage_saving=1-(adapter+core)/(1152*1536)))
    native=json.loads((P/'INTERACTION_COMPLEMENT_ADAPTER_V1_RESULT.json').read_text())['rows']
    comparison=[]
    for r in native:
        controls=[c for c in rows if c['tolerance']==r['tolerance']]
        comparison.append(dict(tolerance=r['tolerance'],model_discarded=r['discarded'],model_storage_saving=r['storage_saving'],
                               generic_discarded_range=[min(c['discarded'] for c in controls),max(c['discarded'] for c in controls)],
                               generic_saving_range=[min(c['storage_saving'] for c in controls),max(c['storage_saving'] for c in controls)]))
    result=dict(controls=rows,comparison=comparison,seconds=time.perf_counter()-start,
                scope='Three fixed iidGaussian matrices with same1152x1536shape and Frobenius-relative tolerance. Generic dimension/control, not null preserving native tensor marginals or circuit semantics.')
    (P/'INTERACTION_COMPLEMENT_GENERIC_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
