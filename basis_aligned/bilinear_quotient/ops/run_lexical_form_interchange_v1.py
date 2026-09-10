#!/usr/bin/env python3
# BQGATE: fixed lexical/form commands,23forwards368seq; no fit or site selection.
"""pred_a instrument/replay/fold; pred_b lexical reuse>=.8 and form drift<=.1;
pred_c form reuse>=.8 and lexical drift<=.1; pred_d joint four-token error<=.2,
additive full effect error<=.1; pred_e G agreement margin meanabs<=.1.
LEXICAL_FORM_INTERCHANGE_V1_PREREGISTRATION.md fixes the complete semantics.
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
OUT=POLY/'LEXICAL_FORM_INTERCHANGE_V1_RESULT.json';BIND=POLY/'LEXICAL_FORM_INTERCHANGE_V1_BINDING.json'
def serial(x):return x.detach().cpu().tolist()
def center(x):return x.double()-x.double().mean(-1,keepdim=True)
def measure(error,reference):
    error=error.double();reference=reference.double();en=error.square().sum(-1);rn=reference.square().sum(-1)
    return dict(relative_l2=float(en.sum().sqrt()/rn.sum().sqrt().clamp_min(1e-30)),reference_norm=float(rn.sum().sqrt()),error_squared_per_row=serial(en),reference_squared_per_row=serial(rn))
def recovery(before,after,patch):
    den=after-before;ok=bool((den>1e-6).all());v=(patch-before)/den if ok else None
    return dict(denominators=serial(den),valid=ok,per_row=None if v is None else serial(v),mean=None if v is None else float(v.mean()))
def axes(s):return torch.stack((s[:,2]-s[:,0],s[:,3]-s[:,1]),-1),torch.stack((s[:,1]-s[:,0],s[:,3]-s[:,2]),-1)


def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    panels=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    H=torch.tensor([[1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],[1,-1,-1,1]],dtype=torch.float64);assert torch.equal(H@H.T,4*torch.eye(4,dtype=torch.float64))
    for n in ['A1','A2','G']:
        rows=panels[n];assert len(rows)==16
        assert all(len(rows[i]['base_ids'])==len(rows[(i+1)%16]['base_ids']) and [j for j,(a,b) in enumerate(zip(rows[i]['base_ids'],rows[(i+1)%16]['base_ids'])) if a!=b]==[3] for i in range(16))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=23,sequences=368,hadamard_control=True)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;H=H.cuda()
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    old=json.loads((POLY/'TOKEN_CONTEXT_BROADCAST_V1_RESULT.json').read_text());engine=ScalarWriteNetwork(backend,e,23,368)
    reports={};checks={};saved={};visits=0
    def body(rows,side,label,values=None,scalars=None):
        nonlocal visits
        capture={}
        def first(_m,args,out):
            nonlocal visits
            visits+=1;capture['v0']=out[1].detach().clone()
            if values is None:return out
            v=out[1].clone();v[:,3]=values[:,3].to(v);return out[0],v
        hook=model.transformer.h[0].attn.register_forward_hook(first)
        try:result=engine.body(rows,side,label=label,group=None if scalars is None else 'all',mode=None if scalars is None else 'donor',donor=scalars)
        finally:hook.remove()
        result.update(capture);return result
    try:
        for name in ['A1','A2']:
            rows=panels[name];rot=rows[1:]+rows[:1];a={}
            for label,rs,side in [('B',rows,'base'),('X',rot,'base'),('Y',rows,'donor'),('Z',rot,'donor')]:a[label]=body(rs,side,name+'_'+label)
            a['LB']=body(rows,'base',name+'_LB',values=a['X']['v0']);a['LY']=body(rows,'donor',name+'_LY',values=a['Z']['v0'])
            a['FB']=body(rows,'base',name+'_FB',scalars=a['Y']['scalars']);a['FX']=body(rot,'base',name+'_FX',scalars=a['Y']['scalars'])
            a['J']=body(rows,'base',name+'_J',values=a['X']['v0'],scalars=a['Y']['scalars'])
            noop=body(rows,'base',name+'_noop',values=a['B']['v0'],scalars=a['B']['scalars']);checks[name+'_noop']=bridge(noop['z'],a['B']['z'])
            ids=torch.tensor([[r['base_answer_id'],r['base_foil_id'],rot[i]['base_answer_id'],rot[i]['base_foil_id']] for i,r in enumerate(rows)],device='cuda')
            s={k:v['z'].gather(1,ids).double() for k,v in a.items()};lex={};form={}
            for k,v in s.items():lex[k],form[k]=axes(v)
            U=model.lm_head.weight[ids].double();components=torch.einsum('ab,nbd->nad',H,U)/4;products=components@model.transformer.h[17].mlp.Down.weight.double()
            checks[name+'_reader_reconstruction']=float((torch.einsum('ba,nbd->nad',H,components)-U).abs().max())
            checks[name+'_product_reconstruction']=float((torch.einsum('ba,nbd->nad',H,products)-U@model.transformer.h[17].mlp.Down.weight.double()).abs().max())
            transfers={
                'LB':recovery(lex['B'][:,0],lex['X'][:,0],lex['LB'][:,0]),'LY':recovery(lex['Y'][:,1],lex['Z'][:,1],lex['LY'][:,1]),
                'FB':recovery(form['B'][:,0],form['Y'][:,0],form['FB'][:,0]),'FX':recovery(form['X'][:,1],form['Z'][:,1],form['FX'][:,1])}
            preservation={
                'LB':measure(form['LB']-form['B'],form['Y']-form['B']),'LY':measure(form['LY']-form['Y'],form['Z']-form['X']),
                'FB':measure(lex['FB']-lex['B'],lex['X']-lex['B']),'FX':measure(lex['FX']-lex['X'],lex['Z']-lex['Y'])}
            if transfers['LB']['valid']:checks[name+'_lexical_replay']=max(abs(x-y) for x,y in zip(transfers['LB']['per_row'],old['reports'][name]['lexical']['value_recovery']))
            else:checks[name+'_lexical_replay']=1e30
            joint=measure(center(s['J']-s['B'])-center(s['Z']-s['B']),center(s['Z']-s['B']))
            addition=measure(center(a['J']['z']-a['LB']['z']-a['FB']['z']+a['B']['z']),center(a['J']['z']-a['B']['z']))
            capability={k:int((s[k].argmax(-1)==i).sum()) for k,i in [('B',0),('X',2),('Y',1),('Z',3)]}
            reports[name]=dict(native_capability=capability,transfers=transfers,preservation=preservation,joint=joint,addition=addition,joint_correct=int((s['J'].argmax(-1)==3).sum()),selected_scores={k:serial(v) for k,v in s.items()})
            saved[name]=dict(token_ids=ids.cpu(),reader_components=components.cpu(),product_readers=products.cpu(),row_ids=[r['row_id'] for r in rows],states={k:dict(h=v['h'].cpu(),selected_scores=s[k].cpu()) for k,v in a.items()})
        rows=panels['G'];rot=rows[1:]+rows[:1];g={}
        g['B']=body(rows,'base','G_B');g['X']=body(rot,'base','G_X');g['L']=body(rows,'base','G_L',values=g['X']['v0'])
        ids=torch.tensor([[r['base_answer_id'],r['base_foil_id']] for r in rows],device='cuda');scores={k:v['z'].gather(1,ids).double() for k,v in g.items()}
        ce={k:F.cross_entropy(v['z'],ids[:,0],reduction='none') for k,v in g.items()};dc=ce['L']-ce['B'];dm=(scores['L'][:,0]-scores['L'][:,1])-(scores['B'][:,0]-scores['B'][:,1])
        checks['G_ce_replay']=abs(float(dc.abs().mean())-old['reports']['G']['arms']['value']['mean_absolute_ce'])
        reports['G']=dict(selected_scores={k:serial(v) for k,v in scores.items()},ce_change_per_row=serial(dc),mean_absolute_ce=float(dc.abs().mean()),margin_change_per_row=serial(dm),mean_absolute_margin=float(dm.abs().mean()),
            natural_margin_change_per_row=serial((scores['X'][:,0]-scores['X'][:,1])-(scores['B'][:,0]-scores['B'][:,1])),common_score_change_per_row=serial((scores['L']-scores['B']).mean(-1)))
        saved['G']=dict(token_ids=ids.cpu(),states={k:dict(h=v['h'].cpu(),selected_scores=scores[k].cpu()) for k,v in g.items()})
    finally:engine.close()
    valid=engine.valid() and visits==23 and checks['G_ce_replay']<=1e-6
    for n in ['A1','A2']:
        valid=valid and checks[n+'_noop']['max_abs']<=1e-3 and checks[n+'_noop']['relative_l2']<=1e-5 and checks[n+'_reader_reconstruction']<=1e-10 and checks[n+'_product_reconstruction']<=1e-10 and checks[n+'_lexical_replay']<=1e-5
    capable=all(all(v==16 for v in reports[n]['native_capability'].values()) for n in ['A1','A2'])
    def branch(labels):
        return valid and capable and all(reports[n]['transfers'][k]['valid'] and reports[n]['transfers'][k]['mean']>=.8 and reports[n]['preservation'][k]['reference_norm']>1e-4 and reports[n]['preservation'][k]['relative_l2']<=.1 for n in ['A1','A2'] for k in labels)
    joint=valid and capable and all(reports[n]['joint_correct']==16 and reports[n]['joint']['reference_norm']>1e-4 and reports[n]['joint']['relative_l2']<=.2 and reports[n]['addition']['reference_norm']>1e-4 and reports[n]['addition']['relative_l2']<=.1 for n in ['A1','A2'])
    artifact=POLY/'LEXICAL_FORM_INTERCHANGE_V1_STATES.pt';assert not artifact.exists();torch.save(saved,artifact)
    result=dict(schema='lexical.form_interchange.v1',predictions={'pred_a_instrument':valid,'pred_b_lexical_reuse_selectivity':branch(['LB','LY']),'pred_c_form_reuse_selectivity':branch(['FB','FX']),'pred_d_joint_independent_effects':joint,'pred_e_agreement_margin':valid and reports['G']['mean_absolute_margin']<=.1},reports=reports,checks=checks,engine_bridges=engine.bridges,
        artifact_sha256=digest(artifact),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],native_parameters=sum(p.numel() for p in model.parameters()),artifact_bytes=artifact.stat().st_size,native_weight_saving=0),
        scope='Fixed lexical/form command reuse and composition screen on opened rows. Exact reader basis and folded coefficients do not independently extract native consumers. Original failed G probability preservation remains separate.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=result['predictions'],targets={n:dict(capability=reports[n]['native_capability'],recovery={k:v['mean'] for k,v in reports[n]['transfers'].items()},preservation={k:v['relative_l2'] for k,v in reports[n]['preservation'].items()},joint_error=reports[n]['joint']['relative_l2'],addition_error=reports[n]['addition']['relative_l2']) for n in ['A1','A2']},G_margin=reports['G']['mean_absolute_margin'],G_ce=reports['G']['mean_absolute_ce'],wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
