#!/usr/bin/env python3
# BQGATE: frozen routing/value operands;28forwards448seq; no fit/head/rank/gain changes.
"""pred_a bridge; pred_b native/full reference; pred_c route/pred_d value sufficiency;
pred_e additive endpoints. Full native background remains charged, zero savings.
"""
import json,os,sys,time
from pathlib import Path
import torch
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import circuit_unit_greedy as g
import circuit_fast_screen_producer as P
import correlative_native_folded_executor_v1 as C
import correlative_factor_executor_v1 as F
from circuit_endpoint_capability_v1 import summarize
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'CORRELATIVE_ROUTE_VALUE_V1_BINDING.json';OUT=POLY/'CORRELATIVE_ROUTE_VALUE_V1_RESULT.json'


def center(z):return z-z.mean(-1,keepdim=True)
def serial(z):return z.detach().cpu().tolist()
def margin(a):return a[:,0]-a[:,1]
def bridge(a,b):
    a=a.double();b=b.double();return {'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30))}


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    data=json.loads((POLY/'CORRELATIVE_RECOMBINED_ROWS_V1.json').read_text());blocks=torch.load(POLY/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',weights_only=True,map_location='cpu')
    for rows in data['panels'].values():
        assert len(rows)==16 and all(len(r['base_ids'])==len(r['donor_ids']) for r in rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':28,'sequences':448,'port_blocks':len(blocks)}));return
    assert not OUT.exists();start=time.perf_counter();backend=P.Bilin18TorchBackend.load('cuda');counts=[0,0];reports={};finite=True;worst_scalar=0.;worst_product=0.
    def count(_m,args):
        counts[0]+=1;counts[1]+=args[0].shape[0]
        if counts[0]>28 or counts[1]>448:raise RuntimeError('Frozen body cap exceeded')
    hook=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    with torch.inference_mode():
        try:
            for name in ('A1','A2','P','C'):
                rows=data['panels'][name];base=g.batch_of(rows,'base');donor=g.batch_of(rows,'donor')
                da,dz,dc=F.forward(backend,donor,blocks)
                ba,bz,bc=F.forward(backend,base,blocks)
                na,nz=g.forward_units(backend,base,return_logits=True)
                ja,jz,jc=F.forward(backend,base,blocks,dc['factors'],'joint')
                ca,cz,cc=C.forward(backend,base,blocks,dc['scalars'])
                ra,rz,rc=F.forward(backend,base,blocks,dc['factors'],'route')
                va,vz,vc=F.forward(backend,base,blocks,dc['factors'],'value')
                b=margin(na);d=margin(da);den=b+d
                scale=float(torch.quantile(den.double(),.5)) if name=='A1' else reports['A1']['native_separation_scale']
                capability=summarize(serial(b),serial(d));positive=bool((den>1e-6).all())
                norm=center(jz.double()-nz.double()).norm().clamp_min(1e-30);effects={}
                for arm,a,z in [('joint',ja,jz),('route',ra,rz),('value',va,vz)]:
                    m=margin(a);err=center(z.double()-jz.double())
                    effects[arm]={'raw_recovery':float(((b-m)/den).mean()) if name in ('A1','A2') and positive else None,
                        'normalized_movement':float((m-b).abs().mean())/scale,
                        'full_effect_relative_error':float(err.norm()/norm),'margin':serial(m),
                        'effect_squared_per_row':serial(center(z.double()-nz.double()).square().sum(-1)),
                        'full_error_squared_per_row':serial(err.square().sum(-1))}
                interaction=center(jz.double()-rz.double()-vz.double()+nz.double())
                terms={}
                # Natural-state product-rule terms are descriptive; every intervention above recomputes its own live factors.
                for layer in blocks:
                    pd,ud=dc['factors'][layer]['route'],dc['factors'][layer]['value']
                    pb,ub=bc['factors'][layer]['route'],bc['factors'][layer]['value']
                    route=((pd-pb)*ub).sum((1,2));value=(pb*(ud-ub)).sum((1,2));cross=((pd-pb)*(ud-ub)).sum((1,2))
                    joint=dc['scalars'][layer]-bc['scalars'][layer]
                    product_err=float(((route+value+cross-joint).abs()/(1e-4+1e-5*joint.abs())).max());worst_product=max(worst_product,product_err)
                    terms[str(layer)]={k:serial(v) for k,v in [('route',route),('value',value),('cross',cross),('joint',joint)]}
                worst_scalar=max(worst_scalar,max(v for cap in (bc,dc,jc,rc,vc) for v in cap['errors'].values()),max(e['scaled_error'] for e in cc['errors'].values()))
                finite=finite and all(bool(torch.isfinite(z).all()) for z in (dz,bz,nz,jz,cz,rz,vz))
                reports[name]={'rows':rows,'native_base_margin':serial(b),'native_donor_margin':serial(d),'capability':capability,
                    'positive_denominators':positive,'native_separation_scale':scale,'effects':effects,
                    'native_bridge':bridge(bz,nz),'joint_scalar_bridge':bridge(jz,cz),
                    'interaction_relative':float(interaction.norm()/norm),'interaction_squared_per_row':serial(interaction.square().sum(-1)),
                    'natural_scalar_product_terms':terms}
        finally:hook.remove()
    valid=lambda b:b['max_abs']<=1e-3 and b['relative_l2']<=1e-5
    instrument=counts==[28,448] and finite and worst_scalar<=1 and worst_product<=1 and all(valid(r['native_bridge']) and valid(r['joint_scalar_bridge']) for r in reports.values())
    general=all(r['capability']['both_endpoints_correct']==16 for r in reports.values()) and all(reports[n]['positive_denominators'] and reports[n]['effects']['joint']['raw_recovery']>=.8 for n in ('A1','A2')) and all(reports[n]['effects']['joint']['normalized_movement']<=.23 for n in ('P','C'))
    def sufficient(arm):
        return general and all(reports[n]['effects'][arm]['raw_recovery']>=.8 and reports[n]['effects'][arm]['full_effect_relative_error']<=.15 for n in ('A1','A2')) and all(reports[n]['effects'][arm]['normalized_movement']<=.23 for n in ('P','C'))
    predictions={'pred_a_instrument':bool(instrument),'pred_b_native_and_joint_reference':bool(general),
        'pred_c_route_sufficient':bool(sufficient('route')),'pred_d_value_sufficient':bool(sufficient('value')),
        'pred_e_additive_endpoint_effects':bool(general and all(reports[n]['interaction_relative']<=.10 for n in ('A1','A2')))}
    result={'schema':'correlative.route_value.v1','predictions':predictions,'reports':reports,'worst_scalar_scaled_error':worst_scalar,'worst_product_scaled_error':worst_product,
        'price':{'body_forwards':counts[0],'sequences':counts[1],'native_parameters':sum(p.numel() for p in backend.model.parameters()),'weight_saving':0,'port_core_coefficients':76032},
        'wall_seconds':time.perf_counter()-start,'runner_sha256':digest(RUNNER),'binding_sha256':digest(BINDING),
        'scope':'Previously tested new word combinations; fixed conditional interface with native routing/value producers and full background. Factor sufficiency, not independent extraction or new OOD test.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'price':result['price'],'wall_seconds':result['wall_seconds'],
        'reports':{n:{'capability':r['capability']['both_endpoints_correct'],'interaction':r['interaction_relative'],'effects':{arm:{k:v for k,v in e.items() if k in ('raw_recovery','normalized_movement','full_effect_relative_error')} for arm,e in r['effects'].items()}} for n,r in reports.items()}},indent=2))


if __name__=='__main__':main()
