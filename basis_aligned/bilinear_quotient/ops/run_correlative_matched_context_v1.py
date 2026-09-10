#!/usr/bin/env python3
# BQGATE: frozen26head14scalar interface;24forwards384seq; no fit or updates.
"""pred_a valid; pred_b native/ceiling; pred_c neutral>=.8;
pred_d neutral>=.8 and matched deficit>=.15; pred_e context movement<=.23.
All545902902native parameters remain; no structural saving.
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
BINDING=POLY/'CORRELATIVE_MATCHED_CONTEXT_V1_BINDING.json'
OUT=POLY/'CORRELATIVE_MATCHED_CONTEXT_V1_RESULT.json'


def margin(af):return af[:,0]-af[:,1]
def serial(t):return t.detach().cpu().tolist()


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    data=json.loads((POLY/'CORRELATIVE_MATCHED_CONTEXT_ROWS_V1.json').read_text())
    blocks=torch.load(POLY/'CORRELATIVE_WEIGHT_PORTS_V1_MAPS.pt',weights_only=True,map_location='cpu')
    artifact=torch.load(POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    assert len(blocks)==14 and len(artifact['units'])==26 and len(data['matching_checks'])==64
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':24,'sequences':384,'matched_pairs':64}));return
    assert not OUT.exists();start=time.perf_counter();backend=P.Bilin18TorchBackend.load('cuda');counts=[0,0];reports={}
    finite=True;worst_scalar=0.
    def count(_m,args):
        counts[0]+=1;counts[1]+=args[0].shape[0]
        if counts[0]>24 or counts[1]>384:raise RuntimeError('Frozen body cap exceeded')
    hook=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    with torch.inference_mode():
        try:
            for frame in ('A1','A2'):
                native={}
                for context in ('only','either'):
                    rows=data['panels'][context][frame]
                    native[context]={side:C.forward(backend,g.batch_of(rows,side),blocks) for side in ('base','donor')}
                for context in ('only','either'):
                    rows=data['panels'][context][frame];base=g.batch_of(rows,'base')
                    ba,bz,bc=native[context]['base'];da,dz,dc=native[context]['donor']
                    na,nz=g.forward_units(backend,base,return_logits=True)
                    ha,hz=g.forward_units(backend,base,units=artifact['units'],donor_cache=dc['cache'],q=None,return_logits=True)
                    qa,qz,qc=C.forward(backend,base,blocks,dc['scalars'])
                    other='either' if context=='only' else 'only'
                    xa,xz,xc=C.forward(backend,base,blocks,native[other]['base'][2]['scalars'])
                    b=margin(na);d=margin(da);den=b+d;h=margin(ha);q=margin(qa);x=margin(xa)
                    scale=float(torch.quantile(den.double(),.5))
                    positive=bool((den>1e-6).all())
                    raw_h=float(((b-h)/den).mean()) if positive else None
                    raw_q=float(((b-q)/den).mean()) if positive else None
                    normalized=raw_q/raw_h if positive and raw_h>1e-6 else None
                    err=(bz.double()-nz.double())
                    for cap in (bc,dc,qc,xc):worst_scalar=max(worst_scalar,max(e['scaled_error'] for e in cap['errors'].values()))
                    finite=finite and all(bool(torch.isfinite(z).all()) for z in (bz,dz,nz,hz,qz,xz))
                    reports[frame+'_'+context]={
                        'rows':rows,'base_margin':serial(b),'donor_margin':serial(d),
                        'full_head_margin':serial(h),'scalar_swap_margin':serial(q),'context_swap_margin':serial(x),
                        'full_head_raw_recovery':raw_h,'scalar_raw_recovery':raw_q,'scalar_normalized_recovery':normalized,
                        'native_correct_count':int(((b>0)&(d>0)).sum()),'positive_denominators':positive,
                        'context_movement':float((x-b).abs().mean())/scale if scale>1e-6 else None,
                        'native_separation_scale':scale,'native_bridge_abs':float(err.abs().max()),
                        'native_bridge_rel':float(err.norm()/nz.double().norm().clamp_min(1e-30)),
                        'native_scalars':{side:{str(l):serial(native[context][side][2]['scalars'][l]) for l in blocks} for side in ('base','donor')},
                        'native_scalar_abs_errors':{side:{str(l):native[context][side][2]['errors'][l]['maxabs'] for l in blocks} for side in ('base','donor')},
                        'context_effect_squared_per_row':serial((xz.double()-nz.double()).square().sum(-1))}
        finally:hook.remove()
    instrument=counts==[24,384] and finite and worst_scalar<=1 and all(r['native_bridge_abs']<=1e-3 and r['native_bridge_rel']<=1e-5 for r in reports.values())
    ceiling=all(r['native_correct_count']==16 and r['positive_denominators'] and r['full_head_raw_recovery']>=.8 for r in reports.values())
    neutral=ceiling and all(reports[f+'_only']['scalar_normalized_recovery']>=.8 for f in ('A1','A2'))
    deficits={f:reports[f+'_only']['scalar_normalized_recovery']-reports[f+'_either']['scalar_normalized_recovery'] if reports[f+'_only']['scalar_normalized_recovery'] is not None and reports[f+'_either']['scalar_normalized_recovery'] is not None else None for f in ('A1','A2')}
    predictions={'pred_a_instrument':bool(instrument),'pred_b_capability_and_ceiling':bool(ceiling),
        'pred_c_neutral_transfer':bool(neutral),'pred_d_matched_inner_cue_deficit':bool(neutral and all(v is not None and v>=.15 for v in deficits.values())),
        'pred_e_same_answer_context_invariance':bool(ceiling and all(r['context_movement'] is not None and r['context_movement']<=.23 for r in reports.values()))}
    result={'schema':'correlative.matched_context.v1','predictions':predictions,'reports':reports,'normalized_transfer_deficits':deficits,
        'worst_scalar_scaled_error':worst_scalar,'price':{'body_forwards':counts[0],'sequences':counts[1],'native_parameters':sum(p.numel() for p in backend.model.parameters()),'weight_saving':0,'port_core_coefficients':76032},
        'runner_sha256':digest(RUNNER),'binding_sha256':digest(BINDING),'wall_seconds':time.perf_counter()-start,
        'scope':data['scope']}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'deficits':deficits,'price':result['price'],'wall_seconds':result['wall_seconds'],
        'reports':{n:{k:r[k] for k in ('native_correct_count','full_head_raw_recovery','scalar_raw_recovery','scalar_normalized_recovery','context_movement')} for n,r in reports.items()}},indent=2))


if __name__=='__main__':main()
