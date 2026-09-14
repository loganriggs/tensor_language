#!/usr/bin/env python3
# BQGATE:140bodyforwards;12captureprefixes;180seconds;no fitting.
"""pred_a post9/post10/all-vocabulary relative errors<=1e-5 EACHcase.
pred_b effect errors<=1e-5+1e-4abs(native); zeroexact; nonzero effect live.
pred_c serialized bank+8contexts+runtime<=89MB and <=.35 projected portfolio.
Null: the isolated shared interface differs at native precision or suffix output.
Price140bodyforwards,12captures,8recipients*8strengths*2,180seconds.
Original prefix/MLP9 generators, MLP10 and later suffix remain native.
"""
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from current_remainder_crossed_v1 import CurrentRemainder
from coupled_routing_value_mlp9_v1 import prepare_for_suffix
from inherited_source_positions_v1 import paired_masks

STEM='COUPLED_SHARED_EXECUTOR_NATIVE_V1'
RECIPIENTS=[0,2,4,6,8,9,10,11]
STRENGTHS=[(0,0),(-.75,.25),(.25,-.75),(.125,.875),(.875,.125),(1.25,.75),(.75,1.25),(-.5,1.5)]


def load_runtime():
    path=P/'extracted_circuits/coupled_shared_executor_v1/execute.py'
    spec=importlib.util.spec_from_file_location('coupled_shared_standalone',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'DENOMINATOR_FACTORED_NATIVE_V1_ROWS.json').read_text())['rows']
    paired_masks(rows[:8]);assert len(rows)==12 and len(STRENGTHS)==8
    assert all(len(rows[i]['ids'])==len(rows[i^1]['ids']) for i in range(12))
    runtime,runtime_path=load_runtime()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        prior=json.loads((P/'COUPLED_SHARED_EXECUTOR_V1_RESULT.json').read_text())
        assert prior['pred_a'] and prior['pred_b'] and prior['pred_c']
        portfolio=json.loads((P/'COUPLED_SHARED_WEIGHT_PORTFOLIO_V1_RESULT.json').read_text())
        assert portfolio['contexts']==8 and portfolio['best_storage_plan']['projected_mask']==0
        print('140bodyforwards;12captures;8recipients*8strengths native/standalone');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt')
    assert not out.exists() and not artifact.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();block=model.transformer.h[9];block10=model.transformer.h[10]
    graph=CurrentRemainder({k:v.cuda() for k,v in torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True).items()})
    mlp_weights=[getattr(block.mlp,k).weight.double() for k in ['Left','Right','Down']]
    mlp_bias=block.mlp.Down_bias.double();context={};captures={};count=0;records=[];effects=[]
    bank={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in torch.load(P/'extracted_circuits/coupled_shared_executor_v1/bank.pt',weights_only=True).items()}
    bank['weights']={k:v.cuda() for k,v in bank['weights'].items()}
    original_forward=block.forward;original_attention10=block10.attn.forward
    def block_pre(module,args):context['raw9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
    def attention_hook(module,args,output):
        if context['mode']=='capture':
            captures[context['row']]=dict(g=graph.routing(args[0]),v=graph.values(args[0]),z=(context['raw9']+output[0]).double(),x0=context['x0'].clone(),first=args[1].clone())
            return output
        return output[0]+context['delta'].to(output[0].dtype),output[1]
    def forwarded(x,v1,x0):
        if context['mode']=='compiled':
            context['executed']=runtime.execute(bank,context['caller'],*context['strengths'])
            return context['executed']['post_mlp9'].to(x.dtype),v1
        return original_forward(x,v1,x0)
    def block_post(module,args,output):context['post9']=output[0].detach().clone()
    def post10(module,args,output):context['post10']=output[0].detach().clone()
    def attention10(x,v1=None):
        if context['mode']=='compiled':return context['executed']['attention10'].to(x.dtype),v1
        return original_attention10(x,v1)
    handles=[block.register_forward_pre_hook(block_pre),block.attn.register_forward_hook(attention_hook),block.register_forward_hook(block_post),block10.register_forward_hook(post10)]
    block.forward=forwarded;block10.attn.forward=attention10
    def forward(i,mode):
        nonlocal count
        context.update(mode=mode,row=i);row=rows[i]
        ids=torch.tensor([row['ids']],device='cuda')
        x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None;context['x0']=x0
        for unit in model.transformer.h:x,v1=unit(x,v1,x0)
        count+=1
        return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    def readouts(scores,row):
        if row['pool']=='natural':return torch.stack([-scores.log_softmax(-1)[row['target_id']],scores[198]-scores[11]]).double().cpu()
        return torch.stack([scores[row['uk_id']]-scores[row['us_id']],scores[row['control_ids'][0]]-scores[row['control_ids'][1]]]).double().cpu()
    rel=lambda a,b:float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))
    callers=[]
    try:
        for i in range(12):forward(i,'capture')
        for i in RECIPIENTS:
            row=rows[i];r,d=captures[i],captures[i^1]
            source=graph.changes(r['g'],r['v'],d['g'],d['v'])
            changes={k:graph.write(source[k]) for k in ['routing','values','mixed']}
            program,_=prepare_for_suffix(r['z'],changes,*mlp_weights,mlp_bias)
            caller=dict(source={k:v for k,v in program.items() if k!='bias'},x0=r['x0'],first_values=r['first'])
            callers.append(caller);context['caller']=caller
            for j,strengths in enumerate(STRENGTHS):
                context['strengths']=strengths
                context['delta']=strengths[0]*changes['routing']+strengths[1]*changes['values']+(strengths[0]*strengths[1])*changes['mixed']
                native=forward(i,'native');native_state=context['post9'];native10=context['post10']
                compiled=forward(i,'compiled');compiled_state=context['post9'];compiled10=context['post10']
                rn,rc=readouts(native,row),readouts(compiled,row)
                if j==0:native_base,compiled_base=rn,rc
                ne,ce=rn-native_base,rc-compiled_base;discrepancy=(ce-ne).abs();bar=1e-5+1e-4*ne.abs()
                records.append(dict(row=i,strengths=strengths,state_error=rel(compiled_state,native_state),post10_error=rel(compiled10,native10),score_error=rel(compiled,native),native_effect=ne.tolist(),compiled_effect=ce.tolist(),effect_discrepancies=discrepancy.tolist(),effect_bars=bar.tolist(),effect_pass=bool((discrepancy<=bar).all()),zero_effect_pass=bool((ne==0).all() and (ce==0).all()) if j==0 else True,live_effect=bool(ne.abs().max()>=1e-5) if j else True))
                effects.append(dict(row=i,strengths=strengths,native=rn.tolist(),compiled=rc.tolist()))
    finally:
        block.forward=original_forward;block10.attn.forward=original_attention10
        for handle in handles:handle.remove()
    assert count==140
    payload=lambda obj:sum(v.numel()*v.element_size() for v in obj.values())
    bank_bytes=payload(bank['weights'])+payload({'lambdas':bank['lambdas'],'bias':bank['bias']})
    context_bytes=sum(payload(c['source'])+c['x0'].numel()*c['x0'].element_size()+c['first_values'].numel()*c['first_values'].element_size() for c in callers)
    runtime_bytes=runtime_path.stat().st_size;total=bank_bytes+context_bytes+runtime_bytes
    portfolio=json.loads((P/'COUPLED_SHARED_WEIGHT_PORTFOLIO_V1_RESULT.json').read_text())
    torch.save(dict(effects=effects),artifact)
    result={'pred_a':all(r['state_error']<=1e-5 and r['post10_error']<=1e-5 and r['score_error']<=1e-5 for r in records),'pred_b':all(r['effect_pass'] and r['zero_effect_pass'] and r['live_effect'] for r in records),'pred_c':total<=89_000_000 and total<=.35*portfolio['shared_projected_portfolio_bytes'],'max_state_error':max(r['state_error'] for r in records),'max_post10_error':max(r['post10_error'] for r in records),'max_score_error':max(r['score_error'] for r in records),'records':records,'bank_tensor_bytes':bank_bytes,'caller_context_bytes':context_bytes,'runtime_source_bytes':runtime_bytes,'total_executor_bytes':total,'shared_native_formula_bytes':portfolio['shared_native_portfolio_bytes'],'shared_projected_bytes':portfolio['shared_projected_portfolio_bytes'],'retained_native_model_parameters':545902902,'body_forwards':count,'seconds':time.perf_counter()-start,'artifact_sha256':digest(artifact),'source_shas':binding,'scope':'Exported shared-bank runtime installed at block9/attention10 for eight existing recipient contexts and seven off-grid coupled interventions plus zero. Original prefix/context/MLP9 generators, MLP10 and later suffix retained. Off-grid strengths are not unseen text; no semantic identification, speedup, arbitrary-text interface or whole-model compression.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_shas']}),flush=True);signal.alarm(0)


if __name__=='__main__':main()
