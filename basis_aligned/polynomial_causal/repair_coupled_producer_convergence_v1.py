"""Reset CG line search only for the four stalled native starts; same objective."""
import hashlib,json,time
from pathlib import Path
import torch
from coupled_producer_pymanopt_v1 import fit
P=Path(__file__).resolve().parent


def cached(name):
    r=json.loads((P/name).read_text());c=r['cache'];assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
    return r,torch.load(c['path'],weights_only=True,map_location='cpu')


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter()
    receipt,state=cached('COUPLED_PRODUCER_NATIVE_V1_RESULT.json')
    _,inputs=cached('COUPLED_PRODUCER_PREPARE_V1_RESULT.json')
    rows=[];points=[]
    for h,row in enumerate(receipt['heads']):
        for j,old in enumerate(row['fits']):
            if old['converged']:continue
            head=inputs['heads'][h];initial=state['heads'][h]['points'][j]
            e,report=fit(initial,head['g'],head['m'],head['s'],head['sharing_ceiling'],head['influence_ceiling'],seconds=10)
            rows.append(dict(head=h,start=j,old_stationarity=old['tangent_stationarity'],
                original_score=old['score'],**report,
                subspace_overlap_with_original=float(torch.linalg.svdvals(e.T@initial).square().mean())))
            points.append(dict(head=h,start=j,point=e));print(json.dumps(rows[-1]),flush=True)
    artifact=Path('/dev/shm/bilin18_coupled_producer_convergence_repair_v1.pt');assert not artifact.exists();torch.save(points,artifact)
    result=dict(all_stalled_starts_now_converged=all(x['converged'] for x in rows),
        repaired_starts=len(rows),previously_converged=36-len(rows),rows=rows,
        maximum_score_change=max(abs(x['score']-x['original_score']) for x in rows),
        cache=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),bytes=artifact.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-start,body_forwards=0,corpus_access=False,
        scope='Same-objective continuation with fresh CG line search. Original B miss retained. No replacement of originally selected best frames or C score.')
    with (P/'COUPLED_PRODUCER_CONVERGENCE_REPAIR_V1_RESULT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
