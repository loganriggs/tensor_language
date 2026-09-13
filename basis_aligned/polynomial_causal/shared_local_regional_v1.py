"""Frozen-reader effects on the existing composed regional circuit cache."""
import json,time
from pathlib import Path
import torch
from audit_fullu_output_functions_v1 import CK
from joint_quadratic_fit_v1 import product_cross
from shared_local_fineweb_v1 import digest

P=Path(__file__).resolve().parent;EPS=torch.finfo(torch.float32).eps


def main():
    torch.set_num_threads(2);started=time.perf_counter()
    receipt=json.loads((P/'COMPOSED_LAST_BLOCK_STATES_V1_RESULT.json').read_text())
    cache_path=P/'COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt'
    assert digest(cache_path)==receipt['artifact_sha']
    data=torch.load(cache_path,map_location='cpu',weights_only=True)
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'][:24]+json.loads((P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json').read_text())['rows']
    ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids']] for r in rows])
    assert ids.shape==(120,4)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True);u=sd['lm_head.weight'].double()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    bias=sd['transformer.h.17.mlp.Down_bias'].double()
    states=data['states'].double();branch=states-data['linear_parts'].double()-bias
    rho=(states.square().mean(-1,keepdim=True)+EPS).sqrt()
    native_readers=u[ids]
    raw=torch.einsum('nsi,nki->nsk',states/rho,native_readers)
    def margins(raw):
        capped=30*torch.tanh(raw/30)
        return torch.stack((capped[...,0]-capped[...,1],capped[...,2]-capped[...,3]),-1)
    reference=margins(raw);cached=data['readouts'].double()
    absolute_replay=float((reference-cached).abs().max())
    replay_effects=[]
    for group in range(5):
        sl=slice(group*24,(group+1)*24)
        for arm in [1,2]:
            a=reference[sl,arm,0]-reference[sl,0,0];b=cached[sl,arm,0]-cached[sl,0,0]
            replay_effects.append(float((a-b).norm()/b.norm().clamp_min(1e-30)))
    instrument=absolute_replay<=1e-4 and max(replay_effects)<=.05
    assert instrument
    metric=d@product_cross(l,r,l,r)@d.T;root=torch.linalg.cholesky((metric+metric.T)/2)
    mean=u.mean(0);coordinate=(u-mean)@root
    _,vectors=torch.linalg.eigh(coordinate.T@coordinate);vectors=vectors.flip(1)
    results=[]
    for width,rank in [(64,78),(128,167)]:
        path=P/f'FULLU_SHARED_LOCAL_FIT_V1_G{width}_PROGRAM.pt'
        program=torch.load(path,map_location='cpu',weights_only=True)
        flat_ids=ids.flatten()
        readers=program['global_codes'][flat_ids].double()@program['global_reader'].double()+program['mean'].double()
        for k,bank in enumerate(program['local_readers']):
            ix=program['labels'][flat_ids]==k
            readers[ix]+=program['local_codes'][flat_ids[ix]].double()@bank.double()
        q=vectors[:,:rank]
        bank=torch.linalg.solve_triangular(root.T,q,upper=True).T
        global_readers=((u[flat_ids]-mean)@root@q)@bank+mean
        cells=[];saved={}
        for name,rr in [('grouped',readers),('matched_global',global_readers)]:
            rr=rr.reshape(120,4,1152)
            route_raw={'quadratic_only':raw+torch.einsum('nsi,nki->nsk',branch/rho,rr-native_readers),
                       'whole_unembedding':torch.einsum('nsi,nki->nsk',states/rho,rr)}
            for route,v in route_raw.items():
                outcome=margins(v);saved[name+'/'+route]=outcome.tolist()
                for group in range(5):
                    sl=slice(group*24,(group+1)*24)
                    for arm,variant in [(1,'full'),(2,'compact'),(3,'mlp_only')]:
                        ref=reference[sl,arm]-reference[sl,0];pred=outcome[sl,arm]-outcome[sl,0]
                        delta=pred-ref
                        cells.append(dict(program=name,route=route,group=group,variant=variant,
                            target_effect_relative_error=float(delta[:,0].norm()/ref[:,0].norm().clamp_min(1e-30)),
                            maximum_target_effect_error=float(delta[:,0].abs().max()),
                            maximum_control_effect_error=float(delta[:,1].abs().max()),
                            control_effect_relative_error=float(delta[:,1].norm()/ref[:,1].norm().clamp_min(1e-30)),
                            opposite_target_signs=int((pred[:,0]*ref[:,0]<0).sum()),
                            reference_target_zeros=int((ref[:,0]==0).sum()),predicted_target_zeros=int((pred[:,0]==0).sum()),
                            maximum_background_margin_drift=float((outcome[sl,0]-reference[sl,0]).abs().max())))
        primary=[c for c in cells if c['program']=='grouped' and c['route']=='quadratic_only' and c['variant'] in ['full','compact']]
        result={'pred_a':instrument,'pred_b':all(c['target_effect_relative_error']<=.1 for c in primary),
            'pred_c':all(c['maximum_control_effect_error']<=1e-4 for c in primary),
            'global_width':width,'program_sha256':digest(path),'cells':cells,'outcomes':saved}
        results.append(result)
    result=dict(schema='shared.local.regional.v1',native_outcomes=reference.tolist(),
        native_margin_max_absolute_replay=absolute_replay,native_target_effect_replay_errors=replay_effects,
        configurations=results,cache_sha256=digest(cache_path),body_forwards=0,fit_steps=0,
        wall_seconds=time.perf_counter()-started,
        scope='Historical conditional regional circuit preservation; original states, normalization and generators supplied.')
    with (P/'SHARED_LOCAL_REGIONAL_V1_RESULT.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['native_outcomes','configurations']},indent=2))
    for c in results:print(json.dumps({k:v for k,v in c.items() if k not in ['cells','outcomes']},indent=2))


if __name__=='__main__':main()
