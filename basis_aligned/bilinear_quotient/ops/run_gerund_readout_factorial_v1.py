#!/usr/bin/env python3
# BQGATE: native numerator/RMS factorial and token moment state,12forwards192seq.
"""pred_a replay/bridges; pred_b norm-only error<=.10; pred_c numerator-only
error<=.10; pred_d true norm repairs projection<=.10; pred_e exact token/moment
scores<=1e-3 and composition<=1e-10. Fixed protocol, no fitting/rank changes.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
from scalar_write_network_executor_v1 import ScalarWriteNetwork,bridge
from selected_reader_response_v1 import initialize,step
import quadratic_readout_state_v1 as Q
from induction_context_transport_v2 import digest
from circuit_endpoint_capability_v1 import summarize
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'GERUND_READOUT_FACTORIAL_V1_RESULT.json';BIND=POLY/'GERUND_READOUT_FACTORIAL_V1_BINDING.json'
def serial(x):return x.detach().cpu().tolist()
def center(x):return x-x.mean(-1,keepdim=True)


def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items());tiny=Q.controls()
    rows=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'gpu_accessed':False,'model_loaded':False,'body_forwards':12,'sequences':192,'controls':tiny}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;U=model.lm_head.weight.double()
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    program={k:v.cuda() for k,v in torch.load(POLY/'SELECTED_READER_RESPONSE_V1_PROGRAM.pt',map_location='cpu',weights_only=True).items()}
    g_row=rows['G'][0];g=U[g_row['base_answer_id']]-U[g_row['base_foil_id']];g=g/g.norm();V=torch.stack([e,g]);W=V.T@torch.linalg.inv(V@V.T)
    engine=ScalarWriteNetwork(backend,e,12,192);mlp=model.transformer.h[17].mlp
    L,R,D=mlp.Left.weight.double(),mlp.Right.weight.double(),mlp.Down.weight.double();le,re=L@e,R@e;quadratic=D@(le*re)
    def read(numerator_state,norm_state):
        rho=(norm_state.double().square().mean(-1)+Q.EPS32).sqrt()
        return 30*torch.tanh((numerator_state.double()@U.T)/(30*rho[:,None]))
    def body(panel,side,label,delta=None):
        ix=torch.arange(16,device='cuda');pos=torch.tensor([v[side+'_semantic_position'] for v in panel],device='cuda');cap={};handles=[]
        if delta is not None:
            def edit(_m,args):
                x=args[0].clone();x[ix,pos]=(x[ix,pos].double()+delta[:,None]*e).to(x);return (x,)
            handles.append(mlp.register_forward_pre_hook(edit))
        def capture(_m,args,out):cap['u']=args[0][ix,pos].detach().clone()
        handles.append(mlp.register_forward_hook(capture))
        try:result=engine.body(panel,side,label=label)
        finally:
            for h in handles:h.remove()
        result.update(cap);return result
    reports={};checks={};saved={};previous=json.loads((POLY/'NATIVE_RESPONSE_GATE_V1_RESULT.json').read_text())['reports']
    try:
        for name in ['A1','A2','G','C']:
            panel=rows[name];base=body(panel,'base',name+'_base');donor=body(panel,'donor',name+'_donor')
            u=base['u'].double();delta=(donor['u'].double()-u)@e;live=body(panel,'base',name+'_edit',delta)
            h0,h1=base['h'].double(),live['h'].double();pred,_=step(program,initialize(program,u),delta);hp=h0+pred@W.T
            z={'base':read(h0,h0),'full':read(h1,h1),'numerator':read(h1,h0),'norm':read(h0,h1),'projection':read(hp,hp),'projection_true_norm':read(hp,h1)}
            checks[name+'_native_base']=bridge(z['base'],base['z']);checks[name+'_native_full']=bridge(z['full'],live['z'])
            bm=base['af'][:,0]-base['af'][:,1];dm=donor['af'][:,0]-donor['af'][:,1];den=bm+dm
            answers=torch.tensor([v['base_answer_id'] for v in panel],device='cuda');foils=torch.tensor([v['base_foil_id'] for v in panel],device='cuda');ix=torch.arange(16,device='cuda')
            basece=F.cross_entropy(base['z'],answers,reduction='none');nativece=F.cross_entropy(live['z'],answers,reduction='none')-basece
            recovery=(bm-(live['af'][:,0]-live['af'][:,1]))/den
            checks[name+'_replay_ce']=bridge(nativece,torch.tensor(previous[name]['native_ce_per_row'],device='cuda'))
            checks[name+'_replay_recovery']=abs(float(recovery.mean())-previous[name]['native_task_recovery'])
            effect=center(live['z'].double()-base['z'].double());effectsq=effect.square().sum(-1);arms={}
            for label,scores in z.items():
                if label=='base':continue
                err=center(scores-live['z'].double()).square().sum(-1);ce=F.cross_entropy(scores,answers,reduction='none')-basece
                arms[label]=dict(effect_error=float((err.sum()/effectsq.sum()).sqrt()),error_squared_per_row=serial(err),effect_squared_per_row=serial(effectsq),mean_ce=float(ce.mean()),ce_per_row=serial(ce),
                                 task_recovery=float(((bm.double()-(scores[ix,answers]-scores[ix,foils]))/den.double()).mean()))
            checks[name+'_replay_projection']=abs(arms['projection']['effect_error']-previous[name]['schemes']['exact']['full_effect_error'])
            interaction=center(z['full']-z['numerator']-z['norm']+z['base']).square().sum(-1)
            linear=((u@R.T)*le+(u@L.T)*re)@D.T
            readers=U[torch.stack([answers,foils],-1)];state=Q.compile_state(h0,linear,quadratic,readers)
            token_scores=Q.evaluate(state,delta);native_scores=torch.stack([live['z'][ix,answers],live['z'][ix,foils]],-1).double()
            checks[name+'_token_max_abs']=float((token_scores-native_scores).abs().max())
            checks[name+'_margin_max_abs']=float(((token_scores[:,0]-token_scores[:,1])-(native_scores[:,0]-native_scores[:,1])).abs().max())
            checks[name+'_composition_max_abs']=float((Q.evaluate(Q.advance(state,delta/2),delta/2)-token_scores).abs().max())
            reports[name]=dict(capability=summarize(serial(bm),serial(dm)),arms=arms,native_ce_per_row=serial(nativece),
                interaction_relative=float((interaction.sum()/effectsq.sum()).sqrt()),interaction_squared_per_row=serial(interaction),effect_squared_per_row=serial(effectsq))
            saved[name]=dict(row_ids=[v['row_id'] for v in panel],h_base=h0.cpu(),h_edited=h1.cpu(),h_projected=hp.cpu(),u_base=u.cpu(),delta=delta.cpu(),linear=linear.cpu(),quadratic=quadratic.cpu(),token_state={k:v.cpu() for k,v in state.items()},token_ids=torch.stack([answers,foils],-1).cpu())
    finally:engine.close()
    a=engine.valid() and all(v['capability']['both_endpoints_correct']==16 for v in reports.values())
    for name in reports:
        a=a and all(checks[name+k]['max_abs']<=1e-3 and checks[name+k]['relative_l2']<=1e-5 for k in ['_native_base','_native_full']) and checks[name+'_replay_ce']['max_abs']<=1e-3 and checks[name+'_replay_recovery']<=1e-3 and checks[name+'_replay_projection']<=1e-3
    norm=a and all(reports[k]['arms']['norm']['effect_error']<=.1 for k in ['A1','A2'])
    numerator=a and all(reports[k]['arms']['numerator']['effect_error']<=.1 for k in ['A1','A2'])
    repair=a and all(reports[k]['arms']['projection_true_norm']['effect_error']<=.1 for k in ['A1','A2'])
    token=a and all(checks[k+'_token_max_abs']<=1e-3 and checks[k+'_margin_max_abs']<=1e-3 and checks[k+'_composition_max_abs']<=1e-10 for k in reports)
    artifact=POLY/'GERUND_READOUT_FACTORIAL_V1_STATES.pt';assert not artifact.exists();torch.save(saved,artifact)
    result=dict(schema='gerund.readout_factorial.v1',predictions={'pred_a_instrument':a,'pred_b_norm_only':norm,'pred_c_numerator_only':numerator,'pred_d_true_norm_repair':repair,'pred_e_token_moment_program':token},
                reports=reports,checks=checks,engine_bridges=engine.bridges,controls=tiny,states_sha256=digest(artifact),
                price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],native_parameters=sum(p.numel() for p in model.parameters()),state_artifact_bytes=artifact.stat().st_size(),token_program_coefficients_per_context=11,native_weight_saving=0),
                runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,
                scope='Fixed grammatical MLP17 local response. Numerator/RMS factorial and conditional token-score program; no independent state producer, full-CE program or behavioral extraction claim.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=result['predictions'],summary={k:dict(errors={a:x['effect_error'] for a,x in v['arms'].items()},interaction=v['interaction_relative']) for k,v in reports.items()},wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
