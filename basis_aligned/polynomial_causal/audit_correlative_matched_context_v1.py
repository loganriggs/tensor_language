"""Exact four-corner additive residual and paired finite-panel transfer uncertainty."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent


def additive_residual(y):
    """Last axis order: both-only, neither-only, both-either, neither-either."""
    v=np.array([1.,-1.,-1.,1.])
    contrast=y@v
    return contrast,contrast[...,None]*v/4


def main():
    path=P/'CORRELATIVE_MATCHED_CONTEXT_V1_RESULT.json';data=json.loads(path.read_text())
    assert data['predictions']['pred_a_instrument']
    maps=torch.load(P/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',weights_only=True,map_location='cpu')
    rng=np.random.default_rng(9111294)
    # Independent least-squares oracle, including exact additive and interacting fixtures.
    design=np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]],dtype=float)
    planted=rng.normal(size=(32,4));contrast,residual=additive_residual(planted)
    projected=planted-(design@np.linalg.lstsq(design,planted.T,rcond=None)[0]).T
    oracle=float(np.max(np.abs(projected-residual)));assert oracle<1e-12
    additive=design@np.array([2.,3.,7.]);assert additive_residual(additive)[0]==0
    assert additive_residual(np.array([0.,0.,0.,1.]))[0]==1
    reports={}
    for frame in ('A1','A2'):
        e=data['reports'][frame+'_either'];o=data['reports'][frame+'_only']
        layers={};stacked_mixed_sq=0.;stacked_cue_sq=0.;certified=0
        for layer,block in sorted(maps.items()):
            key=str(layer);corners=[];errors=[]
            for context in (o,e):
                base=np.array(context['native_scalars']['base'][key]);donor=np.array(context['native_scalars']['donor'][key])
                base_is_both=np.array([r['base_answer']==' and' for r in context['rows']])
                corners.extend([np.where(base_is_both,base,donor),np.where(base_is_both,donor,base)])
                # Each side error is a maximum across its batch, hence safely bounds each row.
                errors.extend([context['native_scalar_abs_errors'][s][key] for s in ('base','donor')])
            table=np.stack(corners,axis=-1);d,res=additive_residual(table)
            floor=np.maximum(np.abs(d)-sum(errors),0)
            wnorm=float(block['writer'].double().norm());cue=np.stack([table[:,1]-table[:,0],table[:,3]-table[:,2]],axis=-1)
            mixed_sq=float(np.sum(d*d))*wnorm**2;cue_sq=float(np.sum(cue*cue))*wnorm**2
            stacked_mixed_sq+=mixed_sq;stacked_cue_sq+=cue_sq;certified+=int(np.sum(floor>0))
            layers[key]={'mixed_difference':d.tolist(),'max_abs_mixed_difference':float(np.max(np.abs(d))),
                'four_corner_bridge_error_sum':sum(errors),'rows_above_bridge_error':int(np.sum(floor>0)),
                'minimum_additive_scalar_l2':float(np.linalg.norm(res)),
                'bridge_adjusted_scalar_l2_lower_bound':float(np.linalg.norm(floor)/2),
                'writer_norm':wnorm,'minimum_additive_write_l2':float(np.linalg.norm(res))*wnorm,
                'context_interaction_over_cue_l2':float(np.linalg.norm(d)/np.linalg.norm(cue)) if np.linalg.norm(cue)>0 else None}
        samples=rng.integers(0,16,size=(4000,16))
        raw={}
        for name,r in [('only',o),('either',e)]:
            base=np.array(r['base_margin']);donor=np.array(r['donor_margin']);den=base+donor
            assert np.all(den>1e-6)
            q=(base-np.array(r['scalar_swap_margin']))/den
            h=(base-np.array(r['full_head_margin']))/den
            values=q[samples].mean(1)/h[samples].mean(1)
            raw[name]=values
        deficit=raw['only']-raw['either']
        reports[frame]={'layers':layers,'scalar_rows':16*len(layers),'rows_above_bridge_error':certified,
            'minimum_additive_stacked_write_l2':float(np.sqrt(stacked_mixed_sq)/2),
            'mixed_write_over_cue_write_l2':float(np.sqrt(stacked_mixed_sq/stacked_cue_sq)),
            'normalized_transfer_ci95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in raw.items()},
            'paired_normalized_transfer_deficit_ci95':np.quantile(deficit,[.025,.975]).tolist()}
    result={'schema':'correlative.matched_context.audit.v1','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'seed':9111294,'bootstrap_draws':4000,'oracle_max_abs_error':oracle,'reports':reports,
        'scope':'Post-result diagnostic. Additive fixed-coordinate falsifier on four corners, not rejection of nonlinear encodings or universal abstraction; paired bootstrap over16authored reporter groups, not population/training OOD guarantee. Stacked writes use direct-sum layer norm, not final-logit norm.'}
    out=P/'CORRELATIVE_MATCHED_CONTEXT_AUDIT_V1_RESULT.json'
    with out.open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps({'oracle_error':oracle,'reports':{n:{k:v for k,v in r.items() if k!='layers'} for n,r in reports.items()}},indent=2))


if __name__=='__main__':main()
