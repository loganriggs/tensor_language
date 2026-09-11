"""CPU native Grassmann fit from hash-verified GPU-prepared weight matrices."""
import os,json,hashlib,time,signal
from pathlib import Path
import torch
from coupled_producer_pymanopt_v1 import fit
from coupled_producer_routing_objective_v1 import trace_quotient
from folded_normalized_router_v1 import rotary
from joint_qk_position_space_v1 import ports
P=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not torch.cuda.is_initialized(),'CPU-only fit must not initialize CUDA'
    binding=json.loads((P/'COUPLED_PRODUCER_NATIVE_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    assert json.loads((P/'COUPLED_PRODUCER_PYMANOPT_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN'):
        print(json.dumps(dict(gpu_accessed=False,heads=9,starts=4,weight=.5,tolerance=1e-7)));return
    prepared=json.loads((P/'COUPLED_PRODUCER_PREPARE_V1_RESULT.json').read_text());assert all(prepared['predictions'].values())
    cache=prepared['cache'];assert digest(cache['path'])==cache['sha256']
    data=torch.load(cache['path'],weights_only=True,map_location='cpu')
    import pymanopt,inspect
    from pymanopt.optimizers import ConjugateGradient
    from pymanopt.manifolds import Grassmann
    execution=dict(script_sha256=digest(__file__),binding_sha256=digest(P/'COUPLED_PRODUCER_NATIVE_V1_BINDING.json'),
        prepared_cache_sha256=cache['sha256'],pymanopt_version=pymanopt.__version__,
        solver_sources={inspect.getfile(cls):digest(inspect.getfile(cls)) for cls in [ConjugateGradient,Grassmann]})
    with (P/'COUPLED_PRODUCER_NATIVE_V1_EXECUTION.json').open('x') as f:
        json.dump(execution,f,indent=2);f.write('\n')
    output=P/'COUPLED_PRODUCER_NATIVE_V1_RESULT.json';assert not output.exists();signal.alarm(2700)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);started=time.perf_counter()
    q1,k1,q2,k2=data['weights'];positions=torch.arange(511);train=positions%2==0;validation=~train
    rotations=rotary(511,128).T@torch.stack([rotary(int(s),128) for s in positions])
    rows=[];saved=[];errors=[]
    checkpoint=Path('/dev/shm/bilin18_coupled_producer_native_v1_progress.pt')
    for h,head in enumerate(data['heads']):
        fits=[];points=[]
        starts=[('routing',head['routing']),('producer',head['producer'])]
        for seed in [1409,1423]:
            gen=torch.Generator().manual_seed(seed+100*h)
            starts.append((f'random_{seed+100*h}',torch.linalg.qr(torch.randn(256,17,generator=gen)).Q))
        for label,initial in starts:
            e,report=fit(initial,head['g'],head['m'],head['s'],head['sharing_ceiling'],head['influence_ceiling'])
            fits.append(dict(label=label,**report));points.append(e)
            errors.extend([report['orthogonality_error'],max(0.,report['initial_score']-report['score'])])
            print(json.dumps(dict(head=h,**fits[-1])),flush=True)
        best=max(range(4),key=lambda i:fits[i]['score']);point=points[best]
        raw=head['basis']@point;old=head['basis']@head['routing']
        stability=[float(torch.linalg.svdvals(point.T@other).square().mean()) for other in points]
        sharing=float(trace_quotient(point,head['m'],head['g'])[0]/17)
        oldsharing=float(trace_quotient(head['routing'],head['m'],head['g'])[0]/17)
        base=(q1[h]@q1[h].T,q2[h]@q2[h].T,q1[h]@q2[h].T)
        grams=tuple(rotations.transpose(-1,-2)@g@rotations for g in base)
        actual=ports(grams,k1[h],k2[h],raw);prior=ports(grams,k1[h],k2[h],old)
        def statistics(x):return dict(discovery_touch=float(x['touch'][train].mean()),
            validation_touch=float(x['touch'][validation].mean()),validation_min_touch=float(x['touch'][validation].min()),
            validation_inside=float(x['inside'][validation].mean()),validation_mixed=float(x['mixed'][validation].mean()))
        row=dict(head=h,fits=fits,best=best,subspace_stability=stability,sharing=sharing,old_sharing=oldsharing,
            normalized_routing_influence=float(trace_quotient(point,head['s'],torch.eye(256))[0]/head['influence_ceiling']),
            normalized_sharing=sharing/head['sharing_ceiling'],actual=statistics(actual),prior=statistics(prior))
        rows.append(row);saved.append(dict(points=torch.stack(points),frame=raw,best=best))
        temporary=checkpoint.with_suffix('.tmp');torch.save(dict(heads=saved,results=rows),temporary);temporary.replace(checkpoint)
        print(json.dumps({k:v for k,v in row.items() if k!='fits'}),flush=True)
    old_receipt=json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_RESULT.json').read_text())
    sharing_receipt=json.loads((P/'MLP16_PRODUCER_OVERLAP_V1_RESULT.json').read_text())
    for row in rows:
        h=row['head'];errors.extend([abs(row['prior']['validation_touch']-old_receipt['heads'][h]['validation_mean_touch']),
            abs(row['old_sharing']-sharing_receipt['folded'][h]['mean_squared_cosine'])])
    mean=lambda values:sum(values)/len(values)
    newshare=mean([x['sharing'] for x in rows]);oldshare=mean([x['old_sharing'] for x in rows])
    newtouch=mean([x['actual']['validation_touch'] for x in rows]);oldtouch=mean([x['prior']['validation_touch'] for x in rows])
    valid=max(errors)<1e-8 and all(f['orthogonality_error']<1e-10 and f['score']>=f['initial_score']-1e-10 for x in rows for f in x['fits'])
    artifact=Path('/dev/shm/bilin18_coupled_producer_native_v1.pt');assert not artifact.exists();torch.save(dict(heads=saved,input_cache_sha256=cache['sha256']),artifact)
    result=dict(predictions={'pred_a_instrument':valid,
        'pred_b_all_converged':valid and all(f['converged'] for x in rows for f in x['fits']),
        'pred_c_joint_tradeoff':valid and newshare>=2*oldshare and newtouch>=.8*oldtouch,
        'pred_d_subspace_stability':valid and all(min(x['subspace_stability'])>=.99 for x in rows)},
        heads=rows,summary=dict(old_sharing=oldshare,new_sharing=newshare,sharing_ratio=newshare/oldshare,
            old_validation_touch=oldtouch,new_validation_touch=newtouch,touch_retention=newtouch/oldtouch),
        maximum_instrument_error=max(errors),wall_seconds=time.perf_counter()-started,execution=execution,
        cache=dict(path=str(artifact),sha256=digest(artifact),bytes=artifact.stat().st_size,ephemeral=True),
        price=dict(body_forwards=0,corpus_access=False,source_frame_numbers=9*1152*17,cpu_fits=36),
        scope='Fixed balanced weight-only coupled subspace objective. Influence surrogate optimized, actual joint-QK touch checked on held-out distances. No semantic identity or normalized behavioral effect evidence.')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='heads'},indent=2),flush=True)


if __name__=='__main__':main()
