"""Fixed subspace midpoint: post-result tradeoff audit, not a new heldout test."""
import json,hashlib,time
from pathlib import Path
import torch
from coupled_producer_routing_objective_v1 import trace_quotient
from folded_normalized_router_v1 import rotary
from joint_qk_position_space_v1 import ports
P=Path(__file__).resolve().parent


def cached(name):
    r=json.loads((P/name).read_text());c=r['cache'];assert hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()==c['sha256']
    return r,torch.load(c['path'],weights_only=True,map_location='cpu')


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter()
    original,state=cached('COUPLED_PRODUCER_NATIVE_V1_RESULT.json')
    _,inputs=cached('COUPLED_PRODUCER_PREPARE_V1_RESULT.json')
    q1,k1,q2,k2=inputs['weights'];positions=torch.arange(511);validation=positions%2==1
    rotations=rotary(511,128).T@torch.stack([rotary(int(s),128) for s in positions])
    rows=[];frames=[];errors=[]
    for h,head in enumerate(inputs['heads']):
        a=head['routing'];b=state['heads'][h]['points'][original['heads'][h]['best']]
        u,c,vh=torch.linalg.svd(a.T@b,full_matrices=False)
        midpoint=(a@u+b@vh.T)/(2+2*c).sqrt()[None,:]
        errors.append(float((midpoint.T@midpoint-torch.eye(17)).abs().max()))
        # Midpoint is the leading eigenspace of the sum of the projectors.
        projected=(a@(a.T@midpoint)+b@(b.T@midpoint))
        errors.append(float((projected-midpoint*(1+c)[None,:]).norm()))
        raw=head['basis']@midpoint;sharing=float(trace_quotient(midpoint,head['m'],head['g'])[0]/17)
        grams=tuple(rotations.transpose(-1,-2)@g@rotations for g in
            (q1[h]@q1[h].T,q2[h]@q2[h].T,q1[h]@q2[h].T))
        actual=ports(grams,k1[h],k2[h],raw)
        row=dict(head=h,sharing=sharing,validation_position_touch=float(actual['touch'][validation].mean()),
            discovery_position_touch=float(actual['touch'][~validation].mean()),
            validation_inside=float(actual['inside'][validation].mean()),
            validation_mixed=float(actual['mixed'][validation].mean()),
            smallest_endpoint_principal_cosine=float(c[-1]))
        rows.append(row);frames.append(raw)
    share=sum(r['sharing'] for r in rows)/9;touch=sum(r['validation_position_touch'] for r in rows)/9
    prior=original['summary'];artifact=Path('/dev/shm/bilin18_coupled_producer_midpoint_v1.pt');assert not artifact.exists()
    torch.save(dict(frames=torch.stack(frames),source_result_sha256=hashlib.sha256((P/'COUPLED_PRODUCER_NATIVE_V1_RESULT.json').read_bytes()).hexdigest()),artifact)
    result=dict(instrument_passed=max(errors)<1e-10,heads=rows,
        summary=dict(sharing=share,touch=touch,sharing_over_original=share/prior['old_sharing'],touch_retention=touch/prior['old_validation_touch'],
            meets_same_numerical_tradeoff_bars=share>=2*prior['old_sharing'] and touch>=.8*prior['old_validation_touch']),
        maximum_instrument_error=max(errors),body_forwards=0,corpus_access=False,wall_seconds=time.perf_counter()-start,
        cache=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),bytes=artifact.stat().st_size,ephemeral=True),
        redteam='Original half-weight objective selected one tradeoff; a fixed midpoint tests a different feasible candidate. No weight or position sweep. Original C failure unchanged.',
        scope='Post-result weight-only constructive audit on previously inspected position split. Not an optimized joint objective, fresh validation, semantic circuit or normalized-effect result.')
    with (P/'COUPLED_PRODUCER_MIDPOINT_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
