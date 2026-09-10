#!/usr/bin/env python3
# BQGATE: five frozen source hypotheses on new combinations;36forwards576seq, no fitting.
"""pred_a bridges<=1e-3abs/1e-5rel andscalar-scaled<=1;
pred_b first/pred_c local: E and recovery>=.8/effecterror<=.15/P,C<=.23;
pred_d E andinteraction<=.10; pred_e nativecorrect/fullrecovery>=.8/P,C<=.23.
Full native inputs/routing/background remain charged; no model weight updates.
"""
import json,os,sys,time
from pathlib import Path
import torch
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import circuit_unit_greedy as g
import circuit_fast_screen_producer as P
import correlative_native_folded_executor_v1 as C
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'CORRELATIVE_VALUE_SOURCE_V1_BINDING.json';OUT=POLY/'CORRELATIVE_VALUE_SOURCE_V1_RESULT.json'


def components(blocks,keep):
    return {layer:{**block,'ports':[{**port,
             'local_value_reader':port['local_value_reader'] if keep=='local' else torch.zeros_like(port['local_value_reader']),
             'first_value_reader':port['first_value_reader'] if keep=='first' else torch.zeros_like(port['first_value_reader'])}
             for port in block['ports']]} for layer,block in blocks.items()}


def bridge(a,b):
    a=a.double();b=b.double()
    return {'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30))}


def center(a):return a-a.mean(-1,keepdim=True)


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    data=json.loads((POLY/'CORRELATIVE_RECOMBINED_ROWS_V1.json').read_text());panels=data['panels']
    blocks=torch.load(POLY/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',weights_only=True,map_location='cpu')
    first,local=components(blocks,'first'),components(blocks,'local')
    map_error=max(float((first[l]['ports'][i][k]+local[l]['ports'][i][k]-p[k]).abs().max())
                  for l,b in blocks.items() for i,p in enumerate(b['ports']) for k in ('local_value_reader','first_value_reader'))
    zeros=all(bool((first[l]['ports'][i]['local_value_reader']==0).all()) and bool((local[l]['ports'][i]['first_value_reader']==0).all()) for l,b in blocks.items() for i,p in enumerate(b['ports']))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'forwards':36,'sequences':576,'map_sum_error':map_error,'zeros_correct':zeros,'provenance':data['provenance']}));return
    assert not OUT.exists();start=time.perf_counter();backend=P.Bilin18TorchBackend.load('cuda');counts=[0,0];reports={};finite=True
    def count(_m,args):
        counts[0]+=1;counts[1]+=args[0].shape[0]
        if counts[0]>36 or counts[1]>576:raise RuntimeError('Frozen body cap exceeded')
    counter=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    with torch.inference_mode():
        try:
            for name in ('A1','A2','P','C'):
                rows=panels[name];base=g.batch_of(rows,'base');donor=g.batch_of(rows,'donor')
                da,dz,dc=C.forward(backend,donor,blocks)
                ba,bz,bc=C.forward(backend,base,blocks)
                na,nz=g.forward_units(backend,base,return_logits=True)
                fulla,fullz,fullc=C.forward(backend,base,blocks,dc['scalars'])
                _a,_z,fd=C.forward(backend,donor,first)
                _a,_z,ld=C.forward(backend,donor,local)
                fa,fz,fc=C.forward(backend,base,first,fd['scalars'])
                la,lz,lc=C.forward(backend,base,local,ld['scalars'])
                summed={layer:fd['scalars'][layer]+ld['scalars'][layer] for layer in blocks}
                ja,jz,jc=C.forward(backend,base,blocks,summed)
                scalar_sum=max(float(((summed[l]-dc['scalars'][l]).abs()/(1e-4+1e-5*dc['scalars'][l].abs())).max()) for l in blocks)
                full_scalar_worst=max(e['scaled_error'] for cap in (dc,bc,fullc,jc) for e in cap['errors'].values())
                base_margin=na[:,0]-na[:,1];donor_margin=da[:,0]-da[:,1];den=base_margin+donor_margin
                a1_scale=float(den.median()) if name=='A1' else reports['A1']['native_separation_scale']
                # torch.median chooses the lower middle value for even n; use the
                # original protocol's midpoint median explicitly for its scale.
                if name=='A1':a1_scale=float(torch.quantile(den.double(),.5))
                gold=center(fullz.double()-nz.double());norm=gold.norm().clamp_min(1e-30)
                effects={}
                for arm,af,z in [('full',fulla,fullz),('first',fa,fz),('local',la,lz)]:
                    margin=af[:,0]-af[:,1]
                    error=center(z.double()-fullz.double())
                    effects[arm]={'raw_recovery':float(((base_margin-margin)/den).mean()) if name in ('A1','A2') and bool((den>1e-6).all()) else None,
                                  'normalized_movement':float((margin-base_margin).abs().mean())/a1_scale if a1_scale>1e-6 else None,
                                  'full_effect_relative_error':float(error.norm()/norm),
                                  'base_oriented_margin':margin.cpu().tolist(),
                                  'centered_effect_squared_per_row':center(z.double()-nz.double()).square().sum(-1).cpu().tolist(),
                                  'full_effect_error_squared_per_row':error.square().sum(-1).cpu().tolist()}
                interaction=center(fullz.double()-fz.double()-lz.double()+nz.double())
                targets=torch.tensor(base.answer_ids,device='cuda')
                ce={arm:torch.nn.functional.cross_entropy(z.float(),targets,reduction='none').cpu().tolist()
                    for arm,z in [('native',nz),('full',fullz),('first',fz),('local',lz)]}
                reports[name]={'rows':rows,'native_separation_scale':a1_scale,'effects':effects,
                               'native_base_margin':base_margin.cpu().tolist(),'native_donor_margin':donor_margin.cpu().tolist(),
                               'native_correct_both_sides':bool((base_margin>0).all() and (donor_margin>0).all()),
                               'positive_recovery_denominators':bool((den>1e-6).all()),
                               'native_bridge':bridge(bz,nz),'split_joint_bridge':bridge(jz,fullz),
                               'component_sum_worst_scaled_error':scalar_sum,'full_scalar_worst_scaled_error':full_scalar_worst,
                               'nonlinear_interaction_relative':float(interaction.norm()/norm),
                               'interaction_squared_per_row':interaction.square().sum(-1).cpu().tolist(),'base_answer_ce':ce}
                finite=finite and all(bool(torch.isfinite(z).all()) for z in (dz,bz,nz,fullz,fz,lz,jz))
        finally:counter.remove()
    valid=lambda b:b['max_abs']<=1e-3 and b['relative_l2']<=1e-5
    instrument=counts==[36,576] and finite and zeros and map_error==0 and all(valid(r['native_bridge']) and valid(r['split_joint_bridge']) and r['component_sum_worst_scaled_error']<=1 and r['full_scalar_worst_scaled_error']<=1 for r in reports.values())
    general=all(r['native_correct_both_sides'] for r in reports.values()) and all(reports[n]['positive_recovery_denominators'] and reports[n]['effects']['full']['raw_recovery']>=.8 for n in ('A1','A2')) and all(reports[n]['effects']['full']['normalized_movement']<=.23 for n in ('P','C'))
    def sufficient(arm):
        return general and all(reports[n]['effects'][arm]['raw_recovery']>=.8 and reports[n]['effects'][arm]['full_effect_relative_error']<=.15 for n in ('A1','A2')) and all(reports[n]['effects'][arm]['normalized_movement']<=.23 for n in ('P','C'))
    predictions={'pred_a_instrument':bool(instrument),'pred_b_first_value_sufficient':bool(sufficient('first')),
                 'pred_c_local_value_sufficient':bool(sufficient('local')),
                 'pred_d_additive_effects':bool(general and all(reports[n]['nonlinear_interaction_relative']<=.10 for n in ('A1','A2'))),
                 'pred_e_full_interface_generalizes':bool(general)}
    result={'schema':'correlative.value_source.v1','predictions':predictions,'reports':reports,'data_provenance':data['provenance'],
            'price':{'body_forwards':counts[0],'sequences':counts[1],'native_parameters':sum(p.numel() for p in backend.model.parameters()),'weight_saving':0,'full_core_scalars':76032,'hypothetical_single_source_core_scalars':46080},
            'binding_sha256':digest(BINDING),'runner_sha256':digest(RUNNER),'wall_seconds':time.perf_counter()-start,
            'scope':'New factor combinations relative to original source banks, fixed learned interface; native routing/background retained; no training-OOD or independent upstream extraction'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'price':result['price'],'wall_seconds':result['wall_seconds'],
                      'reports':{n:{'native_correct':r['native_correct_both_sides'],'effects':{arm:{k:v for k,v in e.items() if k in ('raw_recovery','normalized_movement','full_effect_relative_error')} for arm,e in r['effects'].items()},'interaction':r['nonlinear_interaction_relative']} for n,r in reports.items()}},indent=2))


if __name__=='__main__':main()
