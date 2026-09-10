#!/usr/bin/env python3
# BQGATE: fixed e reads/all18 MLP consumers,28forwards448seq, no fitting.
"""pred_a instrument/replay; pred_b live recovery>=.8; pred_c blocking
loss>=.20 both targets; pred_d blocked scalar closure<=.10; pred_e G base-zero
damage reduction>=50%. See frozen GERUND_MLP_CONSUMER_V1_PREREGISTRATION.md.
"""
import json, os, signal, sys, time
from pathlib import Path
RUNNER=Path(__file__).resolve(); ROOT=RUNNER.parents[3]; POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
from scalar_write_network_executor_v1 import ScalarWriteNetwork, bridge
from bilinear_scalar_consumer_v1 import folded_delta, context_reader, controls
from induction_context_transport_v2 import digest
from circuit_endpoint_capability_v1 import summarize
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'GERUND_MLP_CONSUMER_V1_RESULT.json'; BIND=POLY/'GERUND_MLP_CONSUMER_V1_BINDING.json'
def serial(x): return x.detach().cpu().tolist()
def centered(x): return x-x.mean(-1,keepdim=True)


def main():
    binding=json.loads(BIND.read_text()); assert all(digest(p)==h for p,h in binding.items())
    tiny=controls(); rows=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    names=['A1','A2','G','C']; assert all(len(rows[k])==16 for k in names)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=28,sequences=448,controls=tiny))); return
    assert not OUT.exists(); signal.alarm(900); tic=time.perf_counter(); torch.set_num_threads(2)
    backend=P.Bilin18TorchBackend.load('cuda'); model=backend.model; torch.set_grad_enabled(False)
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',weights_only=True,map_location='cpu')['e'].cuda()
    engine=ScalarWriteNetwork(backend,e,28,448); reports={}; checks={}; readers=[]; consumer_visits=0
    g_row=rows['G'][0]; agreement=model.lm_head.weight[g_row['base_answer_id']].float()-model.lm_head.weight[g_row['base_foil_id']].float(); agreement=agreement/agreement.norm()
    for layer,block in enumerate(model.transformer.h):
        m=block.mlp; assert not m.config.gated
        l,r,d=m.Left.weight.float(),m.Right.weight.float(),m.Down.weight.float()
        ke=context_reader(l,r,d,e.float(),e.float()).double(); kg=context_reader(l,r,d,e.float(),agreement).double()
        readers.append(dict(layer=layer,context_reader_cosine=float(F.cosine_similarity(ke,kg,dim=0)),
                            e_reader_norm=float(ke.norm()),agreement_reader_norm=float(kg.norm())))
    previous=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_RESULT.json').read_text())['reports']

    def body(panel,side,label,group=None,mode=None,donor=None,s0=None):
        nonlocal consumer_visits
        ix=torch.arange(16,device='cuda'); pos=torch.tensor([x[side+'_semantic_position'] for x in panel],device='cuda')
        saved={}; handles=[]
        def callback(layer):
            def hook(m,args,out):
                nonlocal consumer_visits
                consumer_visits+=1; u=args[0][ix,pos]; scalar=u.double()@e; saved[layer]=scalar.detach().clone()
                if s0 is None: return out
                delta=(s0[layer]-scalar).to(u); ef=e.to(u)
                l,r,d=m.Left.weight.to(u),m.Right.weight.to(u),m.Down.weight.to(u)
                change=folded_delta(l,r,d,ef,u,delta)
                direct=u+delta[:,None]*ef
                direct=F.linear(F.linear(direct,l)*F.linear(direct,r),d)+m.Down_bias
                checks[label+'_fold_'+str(layer)]=bridge(out[ix,pos]+change,direct)
                changed=out.clone(); changed[ix,pos]=out[ix,pos]+change
                return changed
            return hook
        for layer,block in enumerate(model.transformer.h): handles.append(block.mlp.register_forward_hook(callback(layer)))
        try: result=engine.body(panel,side,group,mode,donor,label)
        finally:
            for h in handles:h.remove()
        result['input_scalars']=saved; return result

    try:
        with torch.inference_mode():
            for name in names:
                panel=rows[name]; b=body(panel,'base',name+'_base'); d=body(panel,'donor',name+'_donor')
                live=body(panel,'base',name+'_live_swap','all','donor',d['scalars'])
                zero=body(panel,'base',name+'_live_zero','all','zero')
                blocked=body(panel,'base',name+'_blocked_swap','all','donor',d['scalars'],b['input_scalars'])
                blocked_zero=body(panel,'base',name+'_blocked_zero','all','zero',s0=b['input_scalars'])
                noop=body(panel,'base',name+'_noop','all','donor',b['scalars'],b['input_scalars'])
                checks[name+'_noop']=bridge(noop['z'],b['z'])
                bm=b['af'][:,0]-b['af'][:,1]; dm=d['af'][:,0]-d['af'][:,1]; den=bm+dm
                answers=torch.tensor([r['base_answer_id'] for r in panel],device='cuda')
                bce=F.cross_entropy(b['z'],answers,reduction='none')
                prediction=engine.read(b['h'].double()+(((d['h'].double()-b['h'].double())@e)[:,None]*e))
                arms={}
                for label,arm in [('live_swap',live),('blocked_swap',blocked),('live_zero',zero),('blocked_zero',blocked_zero)]:
                    margin=arm['af'][:,0]-arm['af'][:,1]; recovery=(bm-margin)/den
                    ce=F.cross_entropy(arm['z'],answers,reduction='none')-bce
                    item=dict(raw_recovery=float(recovery.mean()),raw_recovery_per_row=serial(recovery),
                              mean_ce=float(ce.mean()),mean_absolute_ce=float(ce.abs().mean()),ce_change_per_row=serial(ce))
                    if label.endswith('swap'):
                        error=centered(prediction.double()-arm['z'].double()).square().sum(-1)
                        effect=centered(arm['z'].double()-b['z'].double()).square().sum(-1)
                        item.update(scalar_prediction_error=float((error.sum()/effect.sum().clamp_min(1e-30)).sqrt()),
                                    error_squared_per_row=serial(error),effect_squared_per_row=serial(effect))
                    arms[label]=item
                checks[name+'_replay_swap']=bridge(torch.tensor(arms['live_swap']['raw_recovery_per_row']),torch.tensor(previous[name]['swap']['raw_recovery_per_row']))
                checks[name+'_replay_zero']=bridge(torch.tensor(arms['live_zero']['ce_change_per_row']),torch.tensor(previous[name]['zero_removal']['base_ce_change_per_row']))
                reports[name]=dict(capability=summarize(serial(bm),serial(dm)),paired_x0_equal=bool(torch.equal(b['x0'],d['x0'])),arms=arms,
                    recovery_loss=arms['live_swap']['raw_recovery']-arms['blocked_swap']['raw_recovery'])
    finally: engine.close()
    capable=all(v['capability']['both_endpoints_correct']==16 and v['paired_x0_equal'] for v in reports.values())
    numeric=all(v['relative_l2']<=1e-5 if '_fold_' in k else v['max_abs']<=1e-3 if '_replay_' in k else v['max_abs']<=1e-3 and v['relative_l2']<=1e-5 for k,v in checks.items())
    instrument=engine.valid() and capable and numeric and consumer_visits==28*18
    live_ok=instrument and all(reports[k]['arms']['live_swap']['raw_recovery']>=.8 for k in ['A1','A2'])
    essential=live_ok and all(reports[k]['recovery_loss']>=.2 for k in ['A1','A2'])
    closure=live_ok and all(reports[k]['arms']['blocked_swap']['scalar_prediction_error']<=.1 for k in ['A1','A2'])
    ga=reports['G']['arms']; mediation=instrument and ga['live_zero']['mean_ce']>=.1 and ga['blocked_zero']['mean_ce']<=.5*ga['live_zero']['mean_ce']
    result=dict(schema='gerund.mlp_consumer.v1',predictions={'pred_a_instrument':instrument,'pred_b_live_target':live_ok,'pred_c_essential_consumers':essential,'pred_d_conditional_closure':closure,'pred_e_agreement_damage_mediation':mediation},
                reports=reports,checks=checks,engine_bridges=engine.bridges,scalar_checks=engine.scalar_checks,context_readers=readers,controls=tiny,
                price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],output_hook_visits=engine.visits,consumer_hook_visits=consumer_visits,native_parameters=sum(p.numel() for p in model.parameters()),native_weight_saving=0),
                runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,
                scope='Conditional post-RMS scalar-read intervention. Other normalized inputs and attention stay live. Reused rows; no independent extraction, OOD or selective-removal promotion.')
    atomic_create_json(OUT,result)
    print(json.dumps(dict(predictions=result['predictions'],summary={k:dict(recovery_loss=v['recovery_loss'],arms={a:{m:x[m] for m in ['raw_recovery','mean_ce']} for a,x in v['arms'].items()},blocked_prediction_error=v['arms']['blocked_swap']['scalar_prediction_error']) for k,v in reports.items()},wall_seconds=result['wall_seconds'])))


if __name__=='__main__': main()
