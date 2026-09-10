#!/usr/bin/env python3
# BQGATE: fixed two-reader context transfer,13forwards193seq, no refit.
"""pred_a instrument; pred_b reference context error<=.10; pred_c cyclic
context error<=.10; pred_d reference full-vocab effect error<=.10 both targets.
Fixed protocol NATIVE_RESPONSE_GATE_V1_PREREGISTRATION.md.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
from scalar_write_network_executor_v1 import ScalarWriteNetwork
from selected_reader_response_v1 import initialize,step
from bilinear_scalar_consumer_v1 import controls
from induction_context_transport_v2 import digest
from circuit_endpoint_capability_v1 import summarize
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'NATIVE_RESPONSE_GATE_V1_RESULT.json';BIND=POLY/'NATIVE_RESPONSE_GATE_V1_BINDING.json'
def serial(x):return x.detach().cpu().tolist()
def center(x):return x-x.mean(-1,keepdim=True)
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))


def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items());tiny=controls()
    rows=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    prototype=json.loads((POLY/'GERUND_SHARED_READER_V1_ROWS.json').read_text())['panels']['A1'][:1]
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'gpu_accessed':False,'model_loaded':False,'body_forwards':13,'sequences':193,'controls':tiny}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    program={k:v.cuda() for k,v in torch.load(POLY/'SELECTED_READER_RESPONSE_V1_PROGRAM.pt',map_location='cpu',weights_only=True).items()}
    row=rows['G'][0];g=model.lm_head.weight[row['base_answer_id']].double()-model.lm_head.weight[row['base_foil_id']].double();g=g/g.norm()
    readers=torch.stack([e,g]);writer=readers.T@torch.linalg.inv(readers@readers.T)
    engine=ScalarWriteNetwork(backend,e,13,193);mlp=model.transformer.h[17].mlp
    l,r,d=mlp.Left.weight.double(),mlp.Right.weight.double(),mlp.Down.weight.double()
    f64=lambda u:((u@l.T)*(u@r.T))@d.T
    reports={};checks={'writer_identity_max_abs':float((readers@writer-torch.eye(2,device='cuda')).abs().max())}

    def body(panel,side,label,delta=None):
        ix=torch.arange(len(panel),device='cuda');pos=torch.tensor([v[side+'_semantic_position'] for v in panel],device='cuda');cap={};handles=[]
        if delta is not None:
            def edit(_m,args):
                x=args[0].clone();x[ix,pos]=(x[ix,pos].double()+delta[:,None]*e).to(x);return (x,)
            handles.append(mlp.register_forward_pre_hook(edit))
        def capture(_m,args,out):cap['u']=args[0][ix,pos].detach().clone();cap['m']=out[ix,pos].detach().clone()
        handles.append(mlp.register_forward_hook(capture))
        try:result=engine.body(panel,side,label=label)
        finally:
            for h in handles:h.remove()
        result.update(cap);return result

    try:
        proto=body(prototype,'base','prototype');su=proto['u'].double()@e
        tau_reference=initialize(program,proto['u'].double())-2*su[:,None]*program['a']
        for name in ['A1','A2','G','C']:
            panel=rows[name];b=body(panel,'base',name+'_base');d_native=body(panel,'donor',name+'_donor')
            u=b['u'].double();s=u@e;delta=(d_native['u'].double()-u)@e
            live=body(panel,'base',name+'_edit',delta)
            full=live['m'].double()-b['m'].double();selected=full@readers.T
            exact,_=step(program,initialize(program,u),delta)
            direct=(f64(u+delta[:,None]*e)-f64(u))@readers.T
            checks[name+'_fp64']=rel(exact,direct)
            checks[name+'_native_scaled']=float(((selected-exact).abs()/(.01+1e-4*exact.abs())).max())
            checks[name+'_effect_norm']=float(full.norm())
            tau=initialize(program,u)-2*s[:,None]*program['a']
            bm=b['af'][:,0]-b['af'][:,1];dm=d_native['af'][:,0]-d_native['af'][:,1];den=bm+dm
            answers=torch.tensor([v['base_answer_id'] for v in panel],device='cuda');foils=torch.tensor([v['base_foil_id'] for v in panel],device='cuda');ix=torch.arange(16,device='cuda')
            basece=F.cross_entropy(b['z'],answers,reduction='none');livece=F.cross_entropy(live['z'],answers,reduction='none')-basece
            live_effect=center(live['z'].double()-b['z'].double());live_sq=live_effect.square().sum(-1)
            schemes={}
            for key,t in [('exact',tau),('reference',tau_reference.expand_as(tau)),('cyclic',tau.roll(-1,0))]:
                pred=delta[:,None]*(t+2*s[:,None]*program['a'])+delta[:,None].square()*program['a']
                z=engine.read(b['h'].double()+pred@writer.T)
                ce=F.cross_entropy(z,answers,reduction='none')-basece
                err=(pred-selected).square().sum(-1);target_sq=selected.square().sum(-1)
                ferr=center(z.double()-live['z'].double()).square().sum(-1)
                schemes[key]=dict(selected_response_error=float((err.sum()/target_sq.sum().clamp_min(1e-30)).sqrt()),
                    selected_error_squared_per_row=serial(err),selected_reference_squared_per_row=serial(target_sq),
                    full_effect_error=float((ferr.sum()/live_sq.sum().clamp_min(1e-30)).sqrt()),full_error_squared_per_row=serial(ferr),full_effect_squared_per_row=serial(live_sq),
                    mean_ce=float(ce.mean()),ce_change_per_row=serial(ce),ce_prediction_mae=float((ce-livece).abs().mean()),
                    task_recovery=float(((bm-(z[ix,answers]-z[ix,foils]))/den).mean()))
            reports[name]=dict(capability=summarize(serial(bm),serial(dm)),delta_per_row=serial(delta),context_gate_per_row=serial(tau),schemes=schemes,
                native_mean_ce=float(livece.mean()),native_ce_per_row=serial(livece),
                native_task_recovery=float(((bm-(live['af'][:,0]-live['af'][:,1]))/den).mean()))
    finally:engine.close()
    capable=all(v['capability']['both_endpoints_correct']==16 for v in reports.values())
    numeric=checks['writer_identity_max_abs']<=1e-10 and all(checks[k+'_fp64']<=1e-10 and checks[k+'_native_scaled']<=1 and checks[k+'_effect_norm']>1e-4 for k in reports)
    a=engine.valid() and capable and numeric
    b=a and all(reports[k]['schemes']['reference']['selected_response_error']<=.1 for k in ['A1','A2'])
    c=a and all(reports[k]['schemes']['cyclic']['selected_response_error']<=.1 for k in ['A1','A2'])
    d_ok=a and all(reports[k]['schemes']['reference']['full_effect_error']<=.1 for k in ['A1','A2'])
    result=dict(schema='native.response_gate.v1',predictions={'pred_a_instrument':a,'pred_b_reference_context':b,'pred_c_lexical_context':c,'pred_d_full_effect':d_ok},
                checks=checks,engine_bridges=engine.bridges,controls=tiny,reports=reports,reference_row_id=prototype[0]['row_id'],reference_gate=serial(tau_reference),
                price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],native_parameters=sum(p.numel() for p in model.parameters()),extra_program_writer_reference_coefficients=4612,native_weight_saving=0),
                runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,
                scope='Fixed two-reader local response on native contexts. Initial scalar/background and context producers remain native. Reused texts; no OOD/extraction/removal promotion.')
    atomic_create_json(OUT,result)
    print(json.dumps(dict(predictions=result['predictions'],summary={k:dict(native_recovery=v['native_task_recovery'],native_ce=v['native_mean_ce'],schemes={s:dict(selected_error=x['selected_response_error'],full_error=x['full_effect_error'],ce_mae=x['ce_prediction_mae']) for s,x in v['schemes'].items()}) for k,v in reports.items()},wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
