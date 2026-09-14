#!/usr/bin/env python3
# BQGATE:110bodyforwards;5prefixes<=247tokens;120seconds;no fitting.
"""pred_a blockstate andall-vocabulary scores<=1e-5relative EACHcase.
pred_b own-effect discrepancies<=1e-5+1e-4abs(referenceeffect) EACHreadout/case.
pred_c zero-strength baseline-subtracted effects exactlyzero,110forwards.
Null: exact conditional rational compiler fails native suffix numerical transfer.
Price110bodyforwards,5prefixes,11strengths,120seconds; context/weights charged.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from even_value_shared_graph_v1 import SharedGraph
from sro_mlp9_rational_v1 import prepare,execute
STEM='SRO_MLP9_SUFFIX_NATIVE_V1'
STRENGTHS=[[int(bool(mask&bit)) for bit in [1,2,4]] for mask in range(8)]+[[-1,.5,1.5],[.25,.75,1.25],[2,-1,0]]


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    panel=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())
    rows=[('regional',i,panel['regional'][i]) for i in [0,24,48]]+[('natural',i,panel['natural'][i]) for i in [0,16]]
    assert len(rows)==5 and all(len(row['ids'])<=247 for _,_,row in rows)
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('110bodyforwards:5prefixes11strengths native/compiled; block9 rational state then unchanged suffix');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');example=P/(STEM+'_EXAMPLE_PROGRAM.pt')
    assert not out.exists() and not artifact.exists() and not example.exists()
    start=time.perf_counter();signal.alarm(120);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();block=model.transformer.h[9]
    graph=SharedGraph({k:v.cuda() for k,v in torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True).items()})
    weights=[getattr(block.mlp,k).weight.double() for k in ['Left','Right','Down']]
    bias=block.mlp.Down_bias.double();context={};count=0;records=[];prices=[];effects=[]
    original_forward=block.forward
    def block_pre(module,args):
        context['raw9']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
    def attention_hook(module,args,output):
        # Only the direct/native path executes attention9. Sources use pristine inputs.
        first=args[1].reshape(*args[0].shape[:2],9,128)[:,:,8]
        branches=graph.state(args[0],first_values=first)
        directions=torch.stack([b@graph.p['output'].double().T for b in branches],-2)
        if context['strength']==0:
            z=(context['raw9']+output[0]).double()
            context['program']=prepare(z,directions,*weights,bias)
        a=torch.tensor(context['amplitude'],device=directions.device,dtype=directions.dtype)
        delta=(directions*a[None,None,:,None]).sum(-2)
        return output[0]-delta.to(output[0].dtype),output[1]
    def forwarded(x,v1,x0):
        if context['mode']=='compiled':
            return execute(context['program'],context['amplitude']).to(x.dtype),v1
        return original_forward(x,v1,x0)
    def block_post(module,args,output):context['post9']=output[0].detach().clone()
    handles=[block.register_forward_pre_hook(block_pre),block.attn.register_forward_hook(attention_hook),block.register_forward_hook(block_post)]
    block.forward=forwarded
    def forward(row,mode):
        nonlocal count
        context['mode']=mode
        ids=torch.tensor([row['ids']],device='cuda')
        x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
        for unit in model.transformer.h:x,v1=unit(x,v1,x0)
        count+=1
        return (30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
    def readouts(logits,pool,row):
        if pool=='regional':
            return torch.stack([logits[row['uk_id']]-logits[row['us_id']],logits[row['control_ids'][0]]-logits[row['control_ids'][1]]]).double().cpu()
        return torch.stack([-logits.log_softmax(-1)[198],logits[198]-logits[11]]).double().cpu()
    relative=lambda a,b:float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))
    try:
        for pool,i,row in rows:
            native_base=compiled_base=None
            for j,a in enumerate(STRENGTHS):
                context.update(strength=j,amplitude=a)
                native=forward(row,'native');native_state=context['post9']
                compiled=forward(row,'compiled');compiled_state=context['post9']
                rn=readouts(native,pool,row);rc=readouts(compiled,pool,row)
                if j==0:
                    native_base=rn;compiled_base=rc
                    program=context['program'];prices.append(dict(pool=pool,row=i,tokens=len(row['ids']),prepared_bytes=sum(v.numel()*v.element_size() for v in program.values())))
                    if not example.exists():torch.save({k:v.cpu() for k,v in program.items()},example)
                native_effect=rn-native_base;compiled_effect=rc-compiled_base
                discrepancy=(compiled_effect-native_effect).abs();bar=1e-5+1e-4*native_effect.abs()
                records.append(dict(pool=pool,row=i,strengths=a,state_error=relative(compiled_state,native_state),score_error=relative(compiled,native),
                                    effect_discrepancies=discrepancy.tolist(),effect_bars=bar.tolist(),effect_pass=bool((discrepancy<=bar).all()),
                                    zero_effect_pass=bool((native_effect==0).all() and (compiled_effect==0).all()) if j==0 else True))
                effects.append(dict(pool=pool,row=i,strengths=a,native=rn.tolist(),compiled=rc.tolist(),native_effect=native_effect.tolist(),compiled_effect=compiled_effect.tolist()))
    finally:
        block.forward=original_forward
        for h in handles:h.remove()
    assert count==110
    torch.save(dict(effects=effects),artifact)
    result={'pred_a':all(r['state_error']<=1e-5 and r['score_error']<=1e-5 for r in records),
            'pred_b':all(r['effect_pass'] for r in records),'pred_c':count==110 and all(r['zero_effect_pass'] for r in records),
            'max_state_error':max(r['state_error'] for r in records),'max_score_error':max(r['score_error'] for r in records),
            'records':records,'prepared_state_prices':prices,'original_mlp_scalars':sum(w.numel() for w in weights)+bias.numel(),
            'example_bytes':example.stat().st_size,'example_sha256':digest(example),'artifact_sha256':digest(artifact),
            'body_forwards':count,'seconds':time.perf_counter()-start,'source_shas':binding,
            'scope':'Exact conditional compiler prepared once per pristine prefix, used for11binary/signed strengths then native suffix. Five reused prefixes; native context/attention/source/MLPweights all charged. No speed/fullmodelcompression/newOOD claim; no target-based fitting.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_shas']}),flush=True)


if __name__=='__main__':main()
