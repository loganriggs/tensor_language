#!/usr/bin/env python3
# BQGATE: fixed e/all36 ports, fresh lexical/cue controls,36forwards576seq.
"""pred_a instrument/replay; pred_b native capability96/96; pred_c fresh
recovery>=.80; pred_d P/G/C preservation<=.10; pred_e target zeroCE>=.10
and G/C absCE<=.10. No fit/direction/layer/gain rescue. Native weights retained.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
from scalar_write_network_executor_v1 import ScalarWriteNetwork,bridge
from gerund_scalar_network_v1 import controls
from circuit_endpoint_capability_v1 import summarize
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'GERUND_FRESH_TRANSFER_V1_RESULT.json';BIND=POLY/'GERUND_FRESH_TRANSFER_V1_BINDING.json'
def serial(x):return x.detach().cpu().tolist()

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    rows=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels'];tiny=controls()
    assert set(rows)=={'A1','A2','P','G','C','R'} and all(len(v)==16 for v in rows.values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'gpu_accessed':False,'model_loaded':False,'body_forwards':36,'sequences':576,'controls':tiny}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2)
    backend=P.Bilin18TorchBackend.load('cuda')
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    engine=ScalarWriteNetwork(backend,e,36,576);reports={};replay={}
    previous=json.loads((POLY/'GERUND_SCALAR_NETWORK_V1_RESULT.json').read_text())['reports']['A1']
    try:
        with torch.inference_mode():
            for name,panel in rows.items():
                b=engine.body(panel,'base',label=name+'_base');d=engine.body(panel,'donor',label=name+'_donor')
                noop=engine.body(panel,'base','all','donor',b['scalars'],name+'_noop')
                engine.bridges[name+'_noop']=bridge(noop['z'],b['z'])
                swapped=engine.body(panel,'base','all','donor',d['scalars'],name+'_swap')
                zb=engine.body(panel,'base','all','zero',label=name+'_zero_base');zd=engine.body(panel,'donor','all','zero',label=name+'_zero_donor')
                bm=b['af'][:,0]-b['af'][:,1];dm=d['af'][:,0]-d['af'][:,1];den=bm+dm
                sm=swapped['af'][:,0]-swapped['af'][:,1];recovery=(bm-sm)/den
                ba=torch.tensor([r['base_answer_id'] for r in panel],device='cuda');da=torch.tensor([r['donor_answer_id'] for r in panel],device='cuda')
                bce=F.cross_entropy(b['z'],ba,reduction='none');dce=F.cross_entropy(d['z'],da,reduction='none')
                sce=F.cross_entropy(swapped['z'],ba,reduction='none')-bce
                bzero=F.cross_entropy(zb['z'],ba,reduction='none')-bce;dzero=F.cross_entropy(zd['z'],da,reduction='none')-dce
                reports[name]={'capability':summarize(serial(bm),serial(dm)),'positive_denominators':bool((den>1e-6).all()),
                    'paired_x0_equal':bool(torch.equal(b['x0'],d['x0'])),'base_margins':serial(bm),'donor_margins':serial(dm),
                    'swap':{'raw_recovery':float(recovery.mean()),'raw_recovery_per_row':serial(recovery),
                            'mean_absolute_ce_change':float(sce.abs().mean()),'mean_ce_change':float(sce.mean()),'ce_change_per_row':serial(sce)},
                    'zero_removal':{'mean_ce_damage':float((bzero.mean()+dzero.mean())/2),'mean_absolute_ce_change':float((bzero.abs().mean()+dzero.abs().mean())/2),
                                    'base_ce_change_per_row':serial(bzero),'donor_ce_change_per_row':serial(dzero)}}
                if name=='R':
                    replay['swap']=bridge(recovery,torch.tensor(previous['arms']['all_swap']['raw_recovery_per_row'],device='cuda'))
                    for side,ce in [('base',bzero),('donor',dzero)]:replay['zero_'+side]=bridge(ce,torch.tensor(previous['arms']['all_zero_'+side]['ce_change_per_row'],device='cuda'))
    finally:engine.close()
    instrument=engine.valid() and len(replay)==3 and all(v['max_abs']<=1e-3 for v in replay.values())
    capable=all(v['capability']['both_endpoints_correct']==16 and v['paired_x0_equal'] for v in reports.values()) and all(reports[k]['positive_denominators'] for k in ['A1','A2','G','C'])
    transfer=instrument and capable and all(reports[k]['swap']['raw_recovery']>=.8 for k in ['A1','A2'])
    preserve=instrument and capable and all(reports[k]['swap']['mean_absolute_ce_change']<=.1 for k in ['P','G','C']) and all(abs(reports[k]['swap']['raw_recovery'])<=.1 for k in ['G','C'])
    removal=instrument and capable and all(reports[k]['zero_removal']['mean_ce_damage']>=.1 for k in ['A1','A2']) and all(reports[k]['zero_removal']['mean_absolute_ce_change']<=.1 for k in ['G','C'])
    result={'schema':'gerund.fresh_transfer.v1','predictions':{'pred_a_instrument':instrument,'pred_b_native_capability':capable,'pred_c_fresh_transfer':transfer,'pred_d_preservation':preserve,'pred_e_selective_removal':removal},
            'reports':reports,'bridges':engine.bridges,'all_port_scalar_checks':engine.scalar_checks,'replay':replay,'controls':tiny,
            'price':{'body_forwards':engine.counts[0],'sequences':engine.counts[1],'hook_visits':engine.visits,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'native_weight_saving':0},
            'runner_sha256':digest(RUNNER),'binding_sha256':digest(BIND),'wall_seconds':time.perf_counter()-tic,
            'scope':'Fixed native distributed intervention on new lexical/cue panels; closer G grammatical control. C/R reused. No independent producer, pretraining-OOD or scalar-only closure claim.'}
    atomic_create_json(OUT,result);print(json.dumps({'predictions':result['predictions'],'summary':{k:{'capability':v['capability']['both_endpoints_correct'],'recovery':v['swap']['raw_recovery'],'swap_absCE':v['swap']['mean_absolute_ce_change'],'zero_CE':v['zero_removal']['mean_ce_damage'],'zero_absCE':v['zero_removal']['mean_absolute_ce_change']} for k,v in reports.items()},'wall_seconds':result['wall_seconds']}))

if __name__=='__main__':main()
