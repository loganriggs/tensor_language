#!/usr/bin/env python3
# BQGATE: fixed two-port first-value factorial,15forwards240seq.
"""pred_a native/static-cache/noop/replay instruments; pred_b first-value gate
transfer>=.8,error<=.2,remaining-stream magnitude<=.2 on A1/A2; pred_c all
lexical endpoints capable, recovery>=.8 both frames,G absCE<=.1. No fitting.
TOKEN_CONTEXT_BROADCAST_V1_PREREGISTRATION.md fixes all arms and nulls.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
from scalar_write_network_executor_v1 import ScalarWriteNetwork,bridge
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'TOKEN_CONTEXT_BROADCAST_V1_RESULT.json';BIND=POLY/'TOKEN_CONTEXT_BROADCAST_V1_BINDING.json'
def serial(x):return x.detach().cpu().tolist()
def center(x):return x.double()-x.double().mean(-1,keepdim=True)
def small_bridge(x):return x['max_abs']<=1e-3 and x['relative_l2']<=1e-5


def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    panels=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    for name in ['A1','A2','G']:
        rows=panels[name];assert len(rows)==16
        assert all(len(rows[i]['base_ids'])==len(rows[(i+1)%16]['base_ids']) and [j for j,(a,b) in enumerate(zip(rows[i]['base_ids'],rows[(i+1)%16]['base_ids'])) if a!=b]==[3] for i in range(16))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=15,sequences=240,changed_position=3)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    saved=torch.load(POLY/'TOKEN_CONTEXT_SOURCE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)
    engine=ScalarWriteNetwork(backend,e,15,240);reports={};checks={};visits=0
    def body(rows,label,values=None):
        nonlocal visits
        capture={};ix=torch.arange(16,device='cuda');pos=torch.tensor([r['base_semantic_position'] for r in rows],device='cuda')
        def first(_m,args,out):
            nonlocal visits
            visits+=1;capture['v0']=out[1].detach().clone()
            return out if values is None else (out[0],values.to(out[1]))
        def last(_m,args):
            nonlocal visits
            visits+=1;capture['u17']=args[0][ix,pos].detach().double().clone()
        handles=[model.transformer.h[0].attn.register_forward_hook(first),model.transformer.h[17].mlp.register_forward_pre_hook(last)]
        try:result=engine.body(rows,'base',label=label)
        finally:
            for h in handles:h.remove()
        result.update(capture);return result
    try:
        for name in ['A1','A2','G']:
            rows=panels[name];donor_rows=rows[1:]+rows[:1];s=saved[name]
            assert s['row_ids']==[r['row_id'] for r in rows]
            kp=s['context_reader'].cuda();base=body(rows,name+'_base');donor=body(donor_rows,name+'_donor')
            value=body(rows,name+'_value',donor['v0']);other=body(donor_rows,name+'_other',base['v0']);noop=body(rows,name+'_noop',base['v0'])
            ids=torch.tensor([r['base_ids'] for r in rows],device='cuda');x0=F.rms_norm(model.transformer.wte(ids),(1152,));lam=model.transformer.h[0].lambdas
            v0=model.transformer.h[0].attn.c_v(F.rms_norm(lam[0]*x0+lam[1]*x0,(1152,))).view_as(base['v0'])
            checks[name+'_static_value']=bridge(v0,base['v0']);checks[name+'_noop']=bridge(noop['z'],base['z'])
            checks[name+'_saved_u_max_abs']=float((base['u17']-s['u17'].cuda()).abs().max())
            tau={k:(kp*v['u17']).sum(-1) for k,v in [('base',base),('donor',donor),('value',value),('other',other)]}
            d=tau['donor']-tau['base'];checks[name+'_gate_norm']=float(d.norm())
            zref=center(donor['z']-base['z']);reference_squared=zref.square().sum(-1);arms={}
            answer=torch.tensor([r['base_answer_id'] for r in rows],device='cuda');ce0=F.cross_entropy(base['z'],answer,reduction='none')
            for label,arm in [('value',value),('other',other)]:
                change=tau[label]-tau['base'];err=change-d;ce=F.cross_entropy(arm['z'],answer,reduction='none')-ce0
                zerr=center(arm['z']-base['z'])-zref
                arms[label]=dict(gate_transfer=float((change*d).sum()/d.square().sum()),gate_error=float(err.norm()/d.norm()),gate_magnitude=float(change.norm()/d.norm()),
                    gate_cross_per_row=serial(change*d),gate_reference_squared_per_row=serial(d.square()),gate_error_squared_per_row=serial(err.square()),gate_change_squared_per_row=serial(change.square()),
                    mean_absolute_ce=float(ce.abs().mean()),ce_change_per_row=serial(ce),full_effect_error=float(zerr.norm()/zref.norm()),full_error_squared_per_row=serial(zerr.square().sum(-1)))
            inter=center(donor['z']-value['z']-other['z']+base['z'])
            panel=dict(base_grammar_capability=int((base['af'][:,0]>base['af'][:,1]).sum()),donor_grammar_capability=int((donor['af'][:,0]>donor['af'][:,1]).sum()),
                gates={k:serial(v) for k,v in tau.items()},arms=arms,full_reference_squared_per_row=serial(reference_squared),interaction_squared_per_row=serial(inter.square().sum(-1)),interaction=float(inter.norm()/zref.norm()))
            if name!='G':
                alt=answer.roll(-1,0);ix=torch.arange(16,device='cuda')
                margin=lambda a:a['z'][ix,answer]-a['z'][ix,alt]
                mb,md,mv,mo=map(margin,[base,donor,value,other]);den=mb-md;cap=(mb>0)&(md<0)
                panel['lexical']=dict(both_correct=int(cap.sum()),base_margin=serial(mb),donor_margin=serial(md),denominators=serial(den),all_denominators_positive=bool((den>1e-6).all()),
                    value_recovery=serial((mb-mv)/den) if bool((den>1e-6).all()) else None,other_recovery=serial((mb-mo)/den) if bool((den>1e-6).all()) else None)
            reports[name]=panel
    finally:engine.close()
    valid=engine.valid() and visits==30 and all(r['base_grammar_capability']==16 and r['donor_grammar_capability']==16 for r in reports.values())
    valid=valid and all(small_bridge(checks[n+'_static_value']) and small_bridge(checks[n+'_noop']) and checks[n+'_saved_u_max_abs']<=1e-5 and checks[n+'_gate_norm']>1e-4 for n in reports)
    gate=valid and all(reports[n]['arms']['value']['gate_transfer']>=.8 and reports[n]['arms']['value']['gate_error']<=.2 and reports[n]['arms']['other']['gate_magnitude']<=.2 for n in ['A1','A2'])
    lexical=valid and all(reports[n]['lexical']['both_correct']==16 and reports[n]['lexical']['all_denominators_positive'] and sum(reports[n]['lexical']['value_recovery'])/16>=.8 for n in ['A1','A2']) and reports['G']['arms']['value']['mean_absolute_ce']<=.1
    result=dict(schema='token.context_broadcast.v1',predictions={'pred_a_instrument':valid,'pred_b_shared_context_source':gate,'pred_c_lexical_behavior':lexical},reports=reports,checks=checks,engine_bridges=engine.bridges,
        runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],native_parameters=sum(p.numel() for p in model.parameters()),native_weight_saving=0),
        scope='Cross-layer first-value source screen. Known token-to-V0 producer exact; all contextual consumers remain native. Gate and lexical capability/transfer judged separately, no circuit promotion.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=result['predictions'],reports={n:dict(value_gate=r['arms']['value']['gate_transfer'],value_gate_error=r['arms']['value']['gate_error'],other_gate_magnitude=r['arms']['other']['gate_magnitude'],lexical=r.get('lexical'),interaction=r['interaction']) for n,r in reports.items()},wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
