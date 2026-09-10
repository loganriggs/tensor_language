#!/usr/bin/env python3
# BQGATE: all36 module-output context-source screens,117forwards1872seq.
"""pred_a instrument/no-op/all-chain/causal-zero; pred_b one common source
transfer>=.50,error<=.50 on both target frames,G absCE<=.10. Nomination only.
Fixed TOKEN_CONTEXT_SOURCE_V1_PREREGISTRATION.md; no fitting or rank changes.
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
from circuit_endpoint_capability_v1 import summarize
from circuit_fast_screen_managed_runner import atomic_create_json
from bilinear_scalar_consumer_v1 import controls
OUT=POLY/'TOKEN_CONTEXT_SOURCE_V1_RESULT.json';BIND=POLY/'TOKEN_CONTEXT_SOURCE_V1_BINDING.json'
SITES=[(l,k) for l in range(18) for k in range(2)]
def key(site):return ('attn' if site[1]==0 else 'mlp')+'_'+str(site[0])
def serial(x):return x.detach().cpu().tolist()


def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items());tiny=controls()
    rows=json.loads((POLY/'GERUND_FRESH_TRANSFER_V1_ROWS.json').read_text())['panels']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':117,'sequences':1872,'controls':tiny}));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model
    e=torch.load(POLY/'GERUND_SHARED_READER_V1_COMPONENT.pt',map_location='cpu',weights_only=True)['e'].cuda()
    engine=ScalarWriteNetwork(backend,e,117,1872);m=model.transformer.h[17].mlp
    L,R,D=m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double();le,re=L@e,R@e
    reports={};checks={};artifacts={};capture_visits=0
    def body(panel,label,sites=(),cache=None,cycle=False):
        nonlocal capture_visits
        ix=torch.arange(16,device='cuda');pos=torch.tensor([v['base_semantic_position'] for v in panel],device='cuda');capture={};handles=[]
        def callback(l,k):
            def hook(_m,args,out):
                nonlocal capture_visits
                capture_visits+=1;y=out[0] if k==0 else out
                if (l,k) in sites:
                    y=y.clone();target=cache[l,k].roll(-1,0) if cycle else cache[l,k];y[ix,pos]=target.to(y)
                capture[l,k]=y[ix,pos].detach().clone()
                if (l,k)==(17,1):capture['u17']=args[0][ix,pos].detach().clone()
                return (y,out[1]) if k==0 else y
            return hook
        for l,block in enumerate(model.transformer.h):
            handles.extend([block.attn.register_forward_hook(callback(l,0)),block.mlp.register_forward_hook(callback(l,1))])
        try:result=engine.body(panel,'base',label=label)
        finally:
            for h in handles:h.remove()
        result['cache']={site:capture[site] for site in SITES};result['u17']=capture['u17'].double();return result
    try:
        for name in ['A1','A2','G']:
            panel=rows[name];assert len(panel)==16
            ids=torch.tensor([[v['base_answer_id'],v['base_foil_id']] for v in panel],device='cuda')
            V=model.lm_head.weight[ids[:,0]].double()-model.lm_head.weight[ids[:,1]].double()
            c=V@D;k=(c*re)@L+(c*le)@R;a=(c*(le*re)).sum(-1);kp=k-2*a[:,None]*e
            pullback=model.transformer.h[17].lambdas[0].double()*(kp@model.transformer.h[16].mlp.Down.weight.double())
            base=body(panel,name+'_base');cache=base['cache'];u=base['u17'];tau=(kp*u).sum(-1);donor_tau=(kp*u.roll(-1,0)).sum(-1);difference=donor_tau-tau
            checks[name+'_gate_norm']=float(difference.norm());checks[name+'_self_orthogonal_max_abs']=float((kp@e).abs().max())
            checks[name+'_final_tokens_match']=all(panel[i]['base_ids'][-1]==panel[(i+1)%16]['base_ids'][-1] for i in range(16))
            noop=body(panel,name+'_noop',SITES,cache);checks[name+'_noop']=bridge(noop['z'],base['z'])
            all_sites=body(panel,name+'_all',SITES,cache,True);all_tau=(kp*all_sites['u17']).sum(-1)
            checks[name+'_all_chain_scaled']=float(((all_tau-donor_tau).abs()/(1e-3+1e-5*donor_tau.abs())).max())
            checks[name+'_all_chain_error']=float((all_tau-donor_tau).norm()/difference.norm().clamp_min(1e-30))
            baseline_ce=F.cross_entropy(base['z'],ids[:,0],reduction='none');arms={}
            for site in SITES:
                label=key(site);patched=body(panel,name+'_'+label,(site,),cache,True)
                change=(kp*patched['u17']).sum(-1)-tau;error=change-difference
                ce=F.cross_entropy(patched['z'],ids[:,0],reduction='none')-baseline_ce
                arms[label]=dict(gate_transfer=float((change*difference).sum()/difference.square().sum()),gate_error=float(error.norm()/difference.norm()),
                    gate_cross_per_row=serial(change*difference),gate_reference_squared_per_row=serial(difference.square()),gate_error_squared_per_row=serial(error.square()),
                    mean_ce=float(ce.mean()),mean_absolute_ce=float(ce.abs().mean()),ce_change_per_row=serial(ce))
                if site==(17,1):checks[name+'_causal_zero']=float(change.abs().max())
            margin=base['af'][:,0]-base['af'][:,1]
            reports[name]=dict(base_capability_count=int((margin>0).sum()),base_margins=serial(margin),native_gate=serial(tau),cyclic_gate=serial(donor_tau),
                gate_difference=serial(difference),arms=arms,mlp16_product_reader_norms=serial(pullback.norm(dim=1)))
            artifacts[name]=dict(row_ids=[v['row_id'] for v in panel],context_reader=kp.cpu(),mlp16_product_reader=pullback.cpu(),u17=u.cpu(),h=base['h'].cpu(),native_outputs={key(s):v.cpu() for s,v in cache.items()})
    finally:engine.close()
    valid=engine.valid() and capture_visits==117*36 and all(v['base_capability_count']==16 for v in reports.values())
    for name in reports:
        valid=valid and checks[name+'_final_tokens_match'] and checks[name+'_gate_norm']>1e-4 and checks[name+'_noop']['max_abs']<=1e-3 and checks[name+'_noop']['relative_l2']<=1e-5 and checks[name+'_all_chain_scaled']<=1 and checks[name+'_causal_zero']<=1e-7
    nominees=[key(s) for s in SITES if all(reports[n]['arms'][key(s)]['gate_transfer']>=.5 and reports[n]['arms'][key(s)]['gate_error']<=.5 for n in ['A1','A2']) and reports['G']['arms'][key(s)]['mean_absolute_ce']<=.1] if valid else []
    artifact=POLY/'TOKEN_CONTEXT_SOURCE_V1_ARTIFACT.pt';assert not artifact.exists();torch.save(artifacts,artifact)
    result=dict(schema='token.context_source.v1',predictions={'pred_a_instrument':valid,'pred_b_shared_source_nomination':valid and bool(nominees)},nominees=nominees,reports=reports,checks=checks,
        engine_bridges=engine.bridges,controls=tiny,artifact_sha256=digest(artifact),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,
        price=dict(body_forwards=engine.counts[0],sequences=engine.counts[1],native_parameters=sum(p.numel() for p in model.parameters()),artifact_bytes=artifact.stat().st_size,native_weight_saving=0),
        scope='Whole-module output source screen for lexical context-gate initialization. Internal gate transfer is not lexical behavior recovery, independent producer or semantic circuit identification.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=result['predictions'],nominees=nominees,top_sources={n:sorted([(k,v['gate_transfer'],v['gate_error']) for k,v in r['arms'].items()],key=lambda x:-x[1])[:6] for n,r in reports.items()},wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
