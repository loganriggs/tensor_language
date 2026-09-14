#!/usr/bin/env python3
# BQGATE:208bodyforwards;8captureprefixes;180seconds;no fitting.
"""pred_a post9/post10 states andall-vocabulary scores<=1e-5relative EACHcase.
pred_b baseline-subtracted readout errors<=1e-5+1e-4abs(effect); zeroexact.
pred_c combined source/attention bytes<=.75independent EACHprefix.
Null: native suffix amplifies local arithmetic error or prices erase saving.
Price208bodyforwards,8captureprefixes,4recipients25strengths,180seconds.
Original native weights, source ports, prefix and suffix remain charged.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from current_remainder_crossed_v1 import CurrentRemainder
from coupled_routing_value_mlp9_v1 import prepare_for_suffix,execute
from coupled_attention10_ports_v1 import compile_program as compile_attention,execute as execute_attention,MAPS
from inherited_source_positions_v1 import paired_masks
STEM='COUPLED_ATTENTION10_NATIVE_V1'
RECIPIENTS=[0,12,24,36]
GRID=[-1,0,.5,1,2]
STRENGTHS=[(0,0)]+[(a,b) for a in GRID for b in GRID if (a,b)!=(0,0)]


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    paired_masks(rows);assert len(rows)==48 and len(STRENGTHS)==25
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        control=json.loads((P/'COUPLED_NATIVE_SUFFIX_V1_PREPARATION_CONTROL.json').read_text())
        assert max(control['coefficient_errors'].values())<=1e-10
        port_control=json.loads((P/'COUPLED_ATTENTION10_PORTS_V1_RESULT.json').read_text())
        assert port_control['maxima']['write_error']<=1e-10 and port_control['maxima']['native_fp32_error']<=1e-5
        print('208bodyforwards;8captureprefixes;4recipients25coupledstrengths native/compiled');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');example=P/(STEM+'_EXAMPLE_PROGRAM.pt')
    assert not out.exists() and not artifact.exists() and not example.exists()
    start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();block=model.transformer.h[9];block10=model.transformer.h[10]
    graph=CurrentRemainder({k:v.cuda() for k,v in torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True).items()})
    weights=[getattr(block.mlp,k).weight.double() for k in ['Left','Right','Down']]
    bias=block.mlp.Down_bias.double();context={};captures={};count=0;records=[];prices=[];effects=[]
    original_forward=block.forward;original_attention10=block10.attn.forward
    attention_weights={name:getattr(block10.attn,name).weight for name in [*MAPS.values(),'c_proj']}
    attention_weights['lamb']=block10.attn.lamb
    payload=lambda p:sum(v.numel()*v.element_size() for v in p.values())
    def block_pre(module,args):context['raw9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
    def attention_hook(module,args,output):
        if context['mode']=='capture':
            captures[context['row']]=dict(g=graph.routing(args[0]),v=graph.values(args[0]),z=(context['raw9']+output[0]).double(),x0=context['x0'].clone(),first=args[1].clone())
            return output
        a,b=context['strengths'];changes=context['changes']
        delta=a*changes['routing']+b*changes['values']+a*b*changes['mixed']
        return output[0]+delta.to(output[0].dtype),output[1]
    def forwarded(x,v1,x0):
        if context['mode']=='compiled':return execute(context['program'],*context['strengths']).to(x.dtype),v1
        return original_forward(x,v1,x0)
    def block_post(module,args,output):context['post9']=output[0].detach().clone()
    def post10(module,args,output):context['post10']=output[0].detach().clone()
    def attention10(x,v1=None):
        if context['mode']=='compiled':return execute_attention(context['attention_program'],*context['strengths']).to(x.dtype),v1
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
        return torch.stack([scores[row['uk_id']]-scores[row['us_id']],scores[row['control_ids'][0]]-scores[row['control_ids'][1]]]).double().cpu()
    rel=lambda a,b:float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))
    try:
        for i in [j for r in RECIPIENTS for j in [r,r^1]]:forward(i,'capture')
        for i in RECIPIENTS:
            row=rows[i];r,d=captures[i],captures[i^1]
            source=graph.changes(r['g'],r['v'],d['g'],d['v'])
            changes={k:graph.write(source[k]) for k in ['routing','values','mixed']}
            program,generic=prepare_for_suffix(r['z'],changes,*weights,bias)
            attention_program=compile_attention(program,r['x0'],block10.lambdas,attention_weights,r['first'])
            context.update(program=program,changes=changes,attention_program=attention_program)
            sizes=[payload(program)+payload(attention_weights)+r['x0'].numel()*r['x0'].element_size()+r['first'].numel()*r['first'].element_size()+block10.lambdas.numel()*block10.lambdas.element_size(),payload(program)+payload(attention_program)+r['x0'].numel()*r['x0'].element_size()]
            prices.append(dict(row=i,tokens=len(row['ids']),independent_bytes=sizes[0],combined_bytes=sizes[1],ratio=sizes[1]/sizes[0]))
            if not example.exists():torch.save(dict(source={k:v.cpu() for k,v in program.items()},attention={k:v.cpu() for k,v in attention_program.items()}),example)
            for j,strengths in enumerate(STRENGTHS):
                context['strengths']=strengths
                native=forward(i,'native');native_state=context['post9'];native10=context['post10']
                compiled=forward(i,'compiled');compiled_state=context['post9'];compiled10=context['post10']
                rn,rc=readouts(native,row),readouts(compiled,row)
                if j==0:native_base,compiled_base=rn,rc
                ne,ce=rn-native_base,rc-compiled_base;discrepancy=(ce-ne).abs();bar=1e-5+1e-4*ne.abs()
                records.append(dict(row=i,strengths=strengths,state_error=rel(compiled_state,native_state),post10_error=rel(compiled10,native10),score_error=rel(compiled,native),
                                    effect_discrepancies=discrepancy.tolist(),effect_bars=bar.tolist(),effect_pass=bool((discrepancy<=bar).all()),
                                    zero_effect_pass=bool((ne==0).all() and (ce==0).all()) if j==0 else True))
                effects.append(dict(row=i,strengths=strengths,native=rn.tolist(),compiled=rc.tolist(),native_effect=ne.tolist(),compiled_effect=ce.tolist()))
    finally:
        block.forward=original_forward;block10.attn.forward=original_attention10
        for handle in handles:handle.remove()
    assert count==208
    torch.save(dict(effects=effects),artifact)
    result={'pred_a':all(r['state_error']<=1e-5 and r['post10_error']<=1e-5 and r['score_error']<=1e-5 for r in records),
            'pred_b':all(r['effect_pass'] and r['zero_effect_pass'] for r in records),
            'pred_c':all(p['ratio']<=.75 for p in prices),
            'max_state_error':max(r['state_error'] for r in records),'max_score_error':max(r['score_error'] for r in records),'max_post10_error':max(r['post10_error'] for r in records),
            'records':records,'prices':prices,'original_mlp_scalars':sum(w.numel() for w in weights)+bias.numel(),
            'example_bytes':example.stat().st_size,'example_sha256':digest(example),'artifact_sha256':digest(artifact),
            'body_forwards':count,'seconds':time.perf_counter()-start,'source_shas':binding,
            'scope':'Combined coupled MLP9 and complete attention10 programs prepared once per pristine context; native suffix tested over100 signed cases. Mixed strength tied to a*b. Context/source/MLPweights remain; no static model compression or independent role identification.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_shas']}),flush=True)


if __name__=='__main__':main()
