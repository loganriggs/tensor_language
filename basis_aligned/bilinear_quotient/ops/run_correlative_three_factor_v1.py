#!/usr/bin/env python3
# BQGATE: fixed QK1/QK2/value lattice through both saved branches;54forwards864seq.
"""pred_a instrumentation/parent bridges; pred_b native double dissociation;
pred_c opposite stored score-half dependence (abs>=.20, A1/A2 difference<=.15);
pred_d value conditional contribution>=.50 in all own-behavior panels.
No fits, head/rank/dose selection. QK labels are not canonical semantic units.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
import correlative_three_factor_executor_v1 as C
from circuit_endpoint_capability_v1 import summarize
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'CORRELATIVE_THREE_FACTOR_V1_RESULT.json';BINDING=POLY/'CORRELATIVE_THREE_FACTOR_V1_BINDING.json'


def serial(x):return x.detach().cpu().tolist()
def margin(x):return x[:,0]-x[:,1]
def center(x):return x-x.mean(-1,keepdim=True)
def bridge(a,b):
    a=a.double();b=b.double();return {'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-30))}


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    panels=json.loads((POLY/'CORRELATIVE_RECOMBINED_ROWS_V1.json').read_text())['panels']
    artifact=torch.load(POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)
    for n in ('A1','A2','C'):
        assert len(panels[n])==16
        for row in panels[n]:assert len(row['base_ids'])==len(row['donor_ids']) and row['base_semantic_position']==row['donor_semantic_position']
    orth=max(float((q.T@q-torch.eye(1)).abs().max()) for q in artifact['q'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':54,'sequences':864,'orthogonality_error':orth,'masks':list(range(1,8))}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();backend=P.Bilin18TorchBackend.load('cuda')
    parent=json.loads((POLY/'CORRELATIVE_COMPLEMENT_SPLIT_V1_RESULT.json').read_text())
    route_parent=json.loads((POLY/'CORRELATIVE_ROUTE_VALUE_V1_RESULT.json').read_text())
    counts=[0,0];reports={};worst_relative=0.;worst_scaled=0.;finite=True
    q={k:v.to('cuda') for k,v in artifact['q'].items()}
    def count(_m,args):
        counts[0]+=1;counts[1]+=len(args[0])
        if counts[0]>54 or counts[1]>864:raise RuntimeError('Frozen body price exceeded')
    hook=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    try:
        with torch.inference_mode():
            for name in ('A1','A2','C'):
                rows=panels[name];base=g.batch_of(rows,'base');donor=g.batch_of(rows,'donor')
                da,dz,dc=C.forward(backend,donor,artifact);ba,bz,bc=C.forward(backend,base,artifact)
                caps=[dc,bc];b=margin(ba);d=margin(da);den=b+d
                branches={}
                for branch in ('P','R'):
                    afs={0:ba};zs={0:bz.double()};arms={}
                    for mask in range(1,8):
                        aa,zz,cc=C.forward(backend,base,artifact,dc['factors'],mask,branch)
                        afs[mask]=aa;zs[mask]=zz.double();caps.append(cc)
                        arms[str(mask)]={'margin':serial(margin(aa)),
                            'raw_recovery':float(((b-margin(aa))/den).mean())}
                    refa,refz=g.forward_units(backend,base,units=artifact['units'],donor_cache=dc['cache'],q=q,complement=branch=='R',return_logits=True)
                    full_effect=center(zs[7]-zs[0]);fullnorm=full_effect.norm().clamp_min(1e-30)
                    interactions={};summed=torch.zeros_like(full_effect)
                    for mask in range(1,8):
                        term=torch.zeros_like(full_effect)
                        for sub in range(8):
                            if sub&mask==sub:term+=(-1)**(mask.bit_count()-sub.bit_count())*zs[sub]
                        term=center(term);summed+=term
                        interactions[str(mask)]={'relative_l2':float(term.norm()/fullnorm),
                                                'squared_per_row':serial(term.square().sum(-1))}
                        err=center(zs[mask]-zs[7]);arms[str(mask)]['full_branch_error']=float(err.norm()/fullnorm)
                    conditional={}
                    for label,bit in [('qk1',1),('qk2',2),('value',4)]:
                        per=(margin(afs[7^bit])-margin(afs[7]))/den
                        conditional[label]={'mean':float(per.mean()),'per_row':serial(per)}
                    role='projector' if branch=='P' else 'complement'
                    replay=abs(arms['7']['raw_recovery']-parent['reports'][name]['effects'][role]['raw_recovery'])
                    prior_replay={}
                    if branch=='P' and name in ('A1','A2'):
                        for key,mask in [('route',3),('value',4)]:
                            prior_replay[key]=abs(arms[str(mask)]['raw_recovery']-route_parent['reports'][name]['effects'][key]['raw_recovery'])
                    higher=center(zs[7]-zs[1]-zs[2]-zs[4]+2*zs[0])
                    branches[branch]={'arms':arms,'conditional_dependencies':conditional,'interactions':interactions,
                        'full_effect_squared_per_row':serial(full_effect.square().sum(-1)),
                        'higher_order_relative':float(higher.norm()/fullnorm),
                        'higher_order_squared_per_row':serial(higher.square().sum(-1)),
                        'mobius_sum_relative_error':float((summed-full_effect).norm()/fullnorm),
                        'full_native_bridge':bridge(zs[7],refz),'parent_recovery_error':replay,'prior_operand_replay_errors':prior_replay}
                    finite=finite and all(bool(torch.isfinite(z).all()) for z in zs.values())
                for cap in caps:
                    worst_relative=max(worst_relative,max(x['relative_l2'] for x in cap['errors'].values()))
                    worst_scaled=max(worst_scaled,max(x['max_scaled'] for x in cap['errors'].values()))
                reports[name]={'capability':summarize(serial(b),serial(d)),'positive_cue_denominators':bool((den>1e-6).all()),
                    'native_base_margin':serial(b),'native_donor_margin':serial(d),'branches':branches}
    finally:hook.remove()
    good=lambda x:x['max_abs']<=1e-3 and x['relative_l2']<=1e-5
    instrument=counts==[54,864] and finite and orth<=1e-5 and worst_relative<=1e-5 and worst_scaled<=1 and all(
        good(b['full_native_bridge']) and b['parent_recovery_error']<=1e-3 and
        all(x<=1e-3 for x in b['prior_operand_replay_errors'].values()) and b['mobius_sum_relative_error']<=1e-10
        for r in reports.values() for b in r['branches'].values())
    recovery=lambda n,b:reports[n]['branches'][b]['arms']['7']['raw_recovery']
    capable=all(r['capability']['both_endpoints_correct']==16 and r['positive_cue_denominators'] for r in reports.values())
    dissociation=capable and all(recovery(n,'P')>=.8 and abs(recovery(n,'R'))<=.23 for n in ('A1','A2')) and recovery('C','R')>=.8 and abs(recovery('C','P'))<=.23
    own={n:reports[n]['branches']['R' if n=='C' else 'P']['conditional_dependencies'] for n in reports}
    differences={n:v['qk1']['mean']-v['qk2']['mean'] for n,v in own.items()}
    different=instrument and dissociation and differences['A1']*differences['A2']>0 and differences['A1']*differences['C']<0 and all(abs(x)>=.20 for x in differences.values()) and abs(differences['A1']-differences['A2'])<=.15
    value=instrument and dissociation and all(v['value']['mean']>=.50 for v in own.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_parent_dissociation':bool(dissociation),
                 'pred_c_distinct_stored_score_halves':bool(different),'pred_d_value_dependence':bool(value)}
    result={'schema':'correlative.three_factor.v1','predictions':predictions,'reports':reports,
            'score_dependency_difference':differences,'worst_factor_relative_error':worst_relative,'worst_factor_scaled_error':worst_scaled,
            'q_orthogonality_error':orth,'price':{'body_forwards':counts[0],'sequences':counts[1],
            'native_parameters':sum(p.numel() for p in backend.model.parameters()),'native_weight_saving':0},
            'runner_sha256':digest(RUNNER),'binding_sha256':digest(BINDING),'wall_seconds':time.perf_counter()-tic,
            'scope':'Stored full-head factor ports, not canonical semantic factors or specific weight-entry identification. Reused authored rows, native background and all weights retained. Parent removal failure unchanged.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'price':result['price'],'wall_seconds':result['wall_seconds'],
                     'own_dependencies':{n:{k:v['mean'] for k,v in values.items()} for n,values in own.items()},
                     'score_dependency_difference':differences,'worst_factor_relative_error':worst_relative,
                     'worst_factor_scaled_error':worst_scaled},indent=2))


if __name__=='__main__':main()
