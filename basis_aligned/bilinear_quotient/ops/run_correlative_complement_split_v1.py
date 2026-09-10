#!/usr/bin/env python3
# BQGATE: fixed projector/complement;30forwards480seq; no fit/head/rank/gain.
"""pred_a instrument; pred_b native/fullset; pred_c doubleinterchange;
pred_d selective mean removal; pred_e additive endpoint effects.
All native weights required; complement3314coordinates not an extracted program.
"""
import json,os,sys,time
from pathlib import Path
import torch
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
import correlative_native_folded_executor_v1 as C
from circuit_endpoint_capability_v1 import summarize
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'CORRELATIVE_COMPLEMENT_SPLIT_V1_BINDING.json';OUT=POLY/'CORRELATIVE_COMPLEMENT_SPLIT_V1_RESULT.json'


def serial(t):return t.detach().cpu().tolist()
def margin(a):return a[:,0]-a[:,1]
def center(z):return z-z.mean(-1,keepdim=True)
def bridge(a,b):
    a=a.double();b=b.double();return {'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30))}


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    rows_data=json.loads((POLY/'CORRELATIVE_RECOMBINED_ROWS_V1.json').read_text())
    blocks=torch.load(POLY/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',map_location='cpu',weights_only=True)
    artifact=torch.load(POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)
    orth=max(float((q.T@q-torch.eye(1)).abs().max()) for q in artifact['q'].values())
    assert len(blocks)==14 and len(artifact['units'])==26 and orth<=1e-5
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':30,'sequences':480,'projector_rank':14,'complement_dimension':3314,'orthogonality_error':orth}));return
    assert not OUT.exists();start=time.perf_counter();backend=P.Bilin18TorchBackend.load('cuda');q={k:v.to('cuda') for k,v in artifact['q'].items()}
    counts=[0,0];reports={};worst_scalar=0.;finite=True
    def count(_m,args):
        counts[0]+=1;counts[1]+=args[0].shape[0]
        if counts[0]>30 or counts[1]>480:raise RuntimeError('Frozen body cap exceeded')
    hook=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    with torch.inference_mode():
        try:
            for name in ('A1','A2','C'):
                rows=rows_data['panels'][name];base=g.batch_of(rows,'base');donor=g.batch_of(rows,'donor')
                da,dz,dc=C.forward(backend,donor,blocks)
                ba,bz,bc=C.forward(backend,base,blocks)
                na,nz=g.forward_units(backend,base,return_logits=True)
                fa,fz=g.forward_units(backend,base,units=artifact['units'],donor_cache=dc['cache'],return_logits=True)
                pa,pz=g.forward_units(backend,base,units=artifact['units'],donor_cache=dc['cache'],q=q,return_logits=True)
                ca,cz=g.forward_units(backend,base,units=artifact['units'],donor_cache=dc['cache'],q=q,complement=True,return_logits=True)
                xa,xz,xc=C.forward(backend,base,blocks,dc['scalars'])
                means={(rid,u):artifact['mu'][u] for rid in base.row_ids for u in artifact['units']}
                ma,mz=g.forward_units(backend,base,units=artifact['units'],donor_cache=means,return_logits=True)
                pma,pmz=g.forward_units(backend,base,units=artifact['units'],donor_cache=means,q=q,return_logits=True)
                cma,cmz=g.forward_units(backend,base,units=artifact['units'],donor_cache=means,q=q,complement=True,return_logits=True)
                b=margin(na);d=margin(da);den=b+d;positive=bool((den>1e-6).all());effects={}
                gold=center(fz.double()-nz.double());norm=gold.norm().clamp_min(1e-30)
                for arm,a,z in [('full',fa,fz),('projector',pa,pz),('complement',ca,cz)]:
                    m=margin(a);err=center(z.double()-fz.double())
                    effects[arm]={'raw_recovery':float(((b-m)/den).mean()) if positive else None,
                        'margin':serial(m),'effect_squared_per_row':serial(center(z.double()-nz.double()).square().sum(-1)),
                        'full_error_squared_per_row':serial(err.square().sum(-1)),
                        'full_effect_error':float(err.norm()/norm)}
                target=torch.tensor(base.answer_ids,device='cuda')
                base_ce=torch.nn.functional.cross_entropy(nz,target,reduction='none');removal={}
                for arm,z in [('full',mz),('projector',pmz),('complement',cmz)]:
                    damage=torch.nn.functional.cross_entropy(z,target,reduction='none')-base_ce
                    removal[arm]={'mean_ce_damage':float(damage.mean()),'mean_absolute_ce_change':float(damage.abs().mean()),'ce_damage_per_row':serial(damage)}
                interaction=center(fz.double()-pz.double()-cz.double()+nz.double())
                worst_scalar=max(worst_scalar,max(e['scaled_error'] for cap in (dc,bc,xc) for e in cap['errors'].values()))
                finite=finite and all(bool(torch.isfinite(z).all()) for z in (dz,bz,nz,fz,pz,cz,xz,mz,pmz,cmz))
                reports[name]={'rows':rows,'capability':summarize(serial(b),serial(d)),'positive_denominators':positive,
                    'native_base_margin':serial(b),'native_donor_margin':serial(d),'effects':effects,'removal':removal,
                    'native_bridge':bridge(bz,nz),'projector_scalar_bridge':bridge(pz,xz),
                    'interaction_relative':float(interaction.norm()/norm),'interaction_squared_per_row':serial(interaction.square().sum(-1))}
        finally:hook.remove()
    valid=lambda b:b['max_abs']<=1e-3 and b['relative_l2']<=1e-5
    instrument=counts==[30,480] and finite and worst_scalar<=1 and all(valid(r['native_bridge']) and valid(r['projector_scalar_bridge']) for r in reports.values())
    capable=all(r['capability']['both_endpoints_correct']==16 and r['positive_denominators'] and r['effects']['full']['raw_recovery']>=.8 for r in reports.values())
    double=capable and all(reports[n]['effects']['projector']['raw_recovery']>=.8 and abs(reports[n]['effects']['complement']['raw_recovery'])<=.23 for n in ('A1','A2')) and abs(reports['C']['effects']['projector']['raw_recovery'])<=.23 and reports['C']['effects']['complement']['raw_recovery']>=.8
    removal=capable and all(reports[n]['removal']['projector']['mean_ce_damage']>=.5 and reports[n]['removal']['complement']['mean_absolute_ce_change']<=.1 for n in ('A1','A2')) and reports['C']['removal']['complement']['mean_ce_damage']>=.5 and reports['C']['removal']['projector']['mean_absolute_ce_change']<=.1
    predictions={'pred_a_instrument':bool(instrument),'pred_b_native_and_fullset':bool(capable),'pred_c_double_interchange':bool(double),'pred_d_selective_mean_removal':bool(removal),'pred_e_additive_endpoint_effects':bool(capable and all(r['interaction_relative']<=.10 for r in reports.values()))}
    result={'schema':'correlative.complement_split.v1','predictions':predictions,'reports':reports,'worst_scalar_scaled_error':worst_scalar,'q_orthogonality_error':orth,
        'price':{'body_forwards':counts[0],'sequences':counts[1],'projector_total_rank':14,'complement_dimension':3314,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'weight_saving':0},
        'runner_sha256':digest(RUNNER),'binding_sha256':digest(BINDING),'wall_seconds':time.perf_counter()-start,
        'scope':'Fixed exact head-space split; complement is broad opaque background, not a discovered semantic computation. Reused short rows, no new fit or training OOD claim.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'price':result['price'],'wall_seconds':result['wall_seconds'],
        'reports':{n:{'capability':r['capability']['both_endpoints_correct'],'recovery':{a:e['raw_recovery'] for a,e in r['effects'].items()},'removal':{a:{k:v for k,v in e.items() if k!='ce_damage_per_row'} for a,e in r['removal'].items()},'interaction':r['interaction_relative']} for n,r in reports.items()}},indent=2))


if __name__=='__main__':main()
