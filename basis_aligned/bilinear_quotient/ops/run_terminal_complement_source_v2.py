#!/usr/bin/env python3
# BQGATE: fixed last-MLP complement clamp,16forwards256seq; no fitting.
"""pred_a instrument/noop/replay/fold; pred_b MLP complement alone lexical-drift
relative error<=.20 all4cells; pred_c carried complement alone<=.20 all4cells.
Null neither source sufficient. Price16forwards256seq, original model retained.
TERMINAL_COMPLEMENT_SOURCE_V2_PREREGISTRATION.md fixes semantics.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
from scalar_write_network_executor_v2 import ScalarWriteNetwork,bridge
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
from run_lexical_form_interchange_v1 import axes,measure
OUT=POLY/'TERMINAL_COMPLEMENT_SOURCE_V2_RESULT.json';BIND=POLY/'TERMINAL_COMPLEMENT_SOURCE_V2_BINDING.json'
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    panels=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    for name in ['A1','A2']:
        rows=panels[name];assert len(rows)==16
        assert all(len(rows[i]['base_ids'])==len(rows[(i+1)%16]['base_ids']) and [j for j,(a,b) in enumerate(zip(rows[i]['base_ids'],rows[(i+1)%16]['base_ids'])) if a!=b]==[3] for i in range(16))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=16,sequences=256,extra_capture_visits=32,physical_readouts=16)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;mlp=model.transformer.h[17].mlp
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    old=torch.load(POLY/'LEXICAL_FORM_INTERCHANGE_V1_STATES.pt',map_location='cpu',weights_only=True)
    engine=ScalarWriteNetwork(backend,e,16,256);visits=[0,0,0,0];checks={};reports={};saved={}
    def perp(x):
        x=x.double();return x-(x@e)[...,None]*e
    def body(rows,side,label,scalars=None,restore=None,own_cache=None):
        batch=g.batch_of(rows,side);ix=torch.arange(16,device='cuda');pos=torch.tensor(batch.semantic_positions,device='cuda');capture={}
        def inp(_m,args):visits[0]+=1;capture['u']=args[0][ix,pos].detach().clone()
        def product(_m,args):visits[1]+=1;capture['products']=args[0][ix,pos].detach().clone()
        def clamp(_m,args,out):
            visits[2]+=1;capture['native_m']=out[ix,pos].detach().clone()
            if restore is None:return out
            live=out[ix,pos].double();updated=perp(restore)+(live@e)[:,None]*e
            y=out.clone();y[ix,pos]=updated.to(y);return y
        def down_out(_m,args,out):visits[3]+=1;capture['down_out']=out[ix,pos].detach().clone()
        def first(_m,args,out):
            capture['v0']=out[1].detach().clone()
            if own_cache is None:return out
            v=out[1].clone();v[:,3]=own_cache[:,3].to(v);return out[0],v
        hooks=[mlp.register_forward_pre_hook(inp),mlp.Down.register_forward_pre_hook(product),mlp.Down.register_forward_hook(down_out),mlp.register_forward_hook(clamp),model.transformer.h[0].attn.register_forward_hook(first)]
        try:r=engine.body(rows,side,label=label,group=None if scalars is None else 'all',mode=None if scalars is None else 'donor',donor=scalars)
        finally:
            for hook in hooks:hook.remove()
        r.update(capture);return r
    try:
        for name in ['A1','A2']:
            rows=panels[name];rot=rows[1:]+rows[:1];arms={}
            for label,rs,side in [('B',rows,'base'),('X',rot,'base'),('Y',rows,'donor')]:arms[label]=body(rs,side,name+'_'+label)
            for label,rs,base in [('FB',rows,'B'),('FX',rot,'X')]:
                arms[label]=body(rs,'base',name+'_'+label,scalars=arms['Y']['scalars'])
                arms[label+'_clamp']=body(rs,'base',name+'_'+label+'_clamp',scalars=arms['Y']['scalars'],restore=arms[base]['writes'][17,1])
            noop=body(rows,'base',name+'_noop',scalars=arms['B']['scalars'],own_cache=arms['B']['v0']);checks[name+'_noop_readout']=bridge(noop['z'],arms['B']['z'])
            ids=old[name]['token_ids'].cuda();C=old[name]['reader_components'].cuda();Cp=perp(C);fold=Cp@mlp.Down.weight.double();bias=Cp@mlp.Down_bias.double()
            reports[name]={};saved[name]=dict(token_ids=ids.cpu(),reader_components=C.cpu(),perpendicular_product_readers=fold.cpu(),row_ids=old[name]['row_ids'],states={})
            for label,v in arms.items():
                score=v['z'].gather(1,ids).double();m=v['writes'][17,1]
                if label in ['B','X','Y','FB','FX']:
                    checks[name+'_'+label+'_replay_state']=bridge(v['h'],old[name]['states'][label]['h'].cuda())
                    checks[name+'_'+label+'_replay_readout']=bridge(score,old[name]['states'][label]['selected_scores'].cuda())
                    predicted=torch.einsum('ncp,np->nc',fold,v['products'].double())+bias
                    actual=torch.einsum('ncd,nd->nc',Cp,m.double())
                    exact_m=v['products'].double()@mlp.Down.weight.double().T+mlp.Down_bias.double()
                    exact_reader=torch.einsum('ncd,nd->nc',Cp,exact_m)
                    err=(predicted-exact_reader).abs();checks[name+'_'+label+'_fold']=dict(max_abs=float(err.max()),max_scaled=float((err/(1e-9+1e-10*exact_reader.abs())).max()))
                    checks[name+'_'+label+'_native_down_exact']=float(((v['down_out']+mlp.Down_bias)-v['native_m']).abs().max())
                    native_reader=torch.einsum('ncd,nd->nc',Cp,v['native_m'].double())
                    saved[name].setdefault('rounding_diagnostics',{})[label]=dict(native_down_reader_error=float((predicted-native_reader).abs().max()),post_scalar_reader_error=float((predicted-actual).abs().max()),scalar_cast_reader_change=float((actual-native_reader).abs().max()))
                saved[name]['states'][label]=dict(h=v['h'].cpu(),r=v['raw17'].cpu(),m=m.cpu(),u=v['u'].cpu(),products=v['products'].cpu(),native_m=v['native_m'].cpu(),selected_scores=score.cpu())
            for label,base in [('FB','B'),('FX','X')]:
                v0=arms[base];v1=arms[label];s=v1['h'].double()@e
                ls0=axes(v0['z'].gather(1,ids))[0];ls1=axes(v1['z'].gather(1,ids))[0];ref=ls1-ls0;cell={}
                for i,j in [(0,0),(0,1),(1,0),(1,1)]:
                    ri=[v0,v1][i]['raw17'];mj=[v0,v1][j]['writes'][17,1]
                    h=s[:,None]*e+perp(ri)+perp(mj);z=engine.read(h);score=z.gather(1,ids).double()
                    cell[str(i)+str(j)]=measure(axes(score)[0]-ls0-ref,ref)
                    if (i,j)==(1,1):
                        checks[name+'_'+label+'_full_state']=bridge(h,v1['h']);checks[name+'_'+label+'_full_readout']=bridge(z,v1['z'])
                    if (i,j)==(1,0):
                        checks[name+'_'+label+'_clamp_state']=bridge(h,arms[label+'_clamp']['h']);checks[name+'_'+label+'_clamp_readout']=bridge(z,arms[label+'_clamp']['z'])
                reports[name][label]=cell
    finally:engine.close()
    valid=engine.valid() and visits==[16,16,16,16]
    for k,v in checks.items():
        if k.endswith('_native_down_exact'):valid=valid and v==0
        elif k.endswith('_fold'):valid=valid and v['max_scaled']<=1
        elif k.endswith('_state'):valid=valid and v['relative_l2']<=1e-5
        else:valid=valid and v['max_abs']<=1e-3 and v['relative_l2']<=1e-5
    def passes(corner):return valid and all(reports[n][l][corner]['reference_norm']>1e-4 and reports[n][l][corner]['relative_l2']<=.20 for n in ['A1','A2'] for l in ['FB','FX'])
    artifact=POLY/'TERMINAL_COMPLEMENT_SOURCE_V2_STATES.pt';assert not artifact.exists();torch.save(saved,artifact)
    result=dict(schema='terminal.complement_source.v2',predictions={'pred_a_instrument':bool(valid),'pred_b_mlp_complement_sufficient':bool(passes('01')),'pred_c_carried_complement_sufficient':bool(passes('10'))},reports=reports,checks=checks,engine_bridges=engine.bridges,scalar_checks=engine.scalar_checks,visits=visits,
        artifact_sha256=digest(artifact),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],physical_readouts=16,fp64_down_control_vectors=160,native_parameters=sum(p.numel() for p in model.parameters()),artifact_bytes=artifact.stat().st_size,native_weight_saving=0),scope='Conditional complementary-source sufficiency on opened rows. Full native contexts and weights retained. Old capability, selectivity and joint-choice failures unchanged.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=result['predictions'],errors={n:{l:{c:v['relative_l2'] for c,v in cs.items()} for l,cs in cells.items()} for n,cells in reports.items()},wall_seconds=result['wall_seconds'])))
if __name__=='__main__':main()
