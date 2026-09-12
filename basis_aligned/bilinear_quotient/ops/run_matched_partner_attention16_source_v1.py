#!/usr/bin/env python3
# BQGATE:16bodyforwards,128sequences,max18tokens;frozengain/sourceaudit;300sec.
"""pred_a native source/replay identities; pred_b residual-only swaps<=.1;
pred_c attention-dependent swaps<=.1; sign>=.9/live>=4 on3 active families.
Exact conditional gain program; null: source interactions cannot be separated.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from attention_source_quartic_v1 import compile_sources,evaluate_gain
from packed_quadratic_branch_v1 import execute
from quartic_frozen_native_score_v2 import score
STEM='MATCHED_PARTNER_ATTENTION16_SOURCE_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows'];sequences=[];buckets={}
    for row in rows:
        for side in ('base','donor'):
            ids=row[side+'_ids'];index=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(index)
    assert len(sequences)==128 and sum((len(v)+7)//8 for v in buckets.values())==16
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=16,sequences=128,max_length=18)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PORTS.pt');assert not out.exists() and not ap.exists();signal.alarm(300)
    tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();block=model.transformer.h[16];captured={};counts=[0,0]
    def source_hook(module,args):captured['residual']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
    def attention_hook(module,args,value):captured['attention']=value[0].detach()
    def head_input_hook(module,args):captured['heads']=args[0].detach()
    def input_hook(module,args):captured['input16']=args[0].detach()
    def output_hook(module,args,value):captured['raw_logits']=value[:,-1].detach()
    def counter(module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=16 and args[0].shape[1]<=18
    handles=[block.register_forward_pre_hook(source_hook),block.attn.register_forward_hook(attention_hook),block.attn.c_proj.register_forward_pre_hook(head_input_hook),
             block.mlp.register_forward_pre_hook(input_hook),model.lm_head.register_forward_hook(output_hook),model.transformer.h[0].attn.register_forward_pre_hook(counter)]
    cache=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True);ports=cache['ports'];state=ports['pre']+ports['native_output']
    source={k:torch.empty(128,1152) for k in ('residual','attention','heads')};checks=[]
    try:
        for length,indices in sorted(buckets.items()):
            for offset in range(0,len(indices),8):
                selected=indices[offset:offset+8];tokens=torch.tensor([sequences[i] for i in selected],device='cuda');model(tokens,tokens)
                rr=captured['residual'][:,-1];aa=captured['attention'][:,-1];xx=captured['input16'][:,-1];heads=captured['heads'][:,-1]
                reconstructed=F.rms_norm(rr+aa,(1152,));old=ports['input16'][selected].cuda()
                projected=F.linear(heads,block.attn.c_proj.weight.to(heads.dtype))
                physical=30*torch.tanh(captured['raw_logits']/30)
                manual=30*torch.tanh(F.linear(F.rms_norm(state[selected].cuda(),(1152,)),model.lm_head.weight)/30)
                for key,value in [('residual',rr),('attention',aa),('heads',heads)]:source[key][selected]=value.cpu()
                checks.extend([float((reconstructed-xx).norm()/xx.norm()),float((old-xx).norm()/old.norm()),float((projected-aa).norm()/aa.norm()),float((manual-physical).norm()/physical.norm())])
    finally:
        for hook in handles:hook.remove()
    assert counts==[16,128]
    program=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True);rr=source['residual'].double().cuda();aa=source['attention'].double().cuda()
    den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps;compiled=compile_sources(program,rr,aa);gain_errors=[]
    for gain in (0.,.5,1.,1.5):
        raw=rr+gain*aa;x=raw/(raw.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None]
        direct=execute(program,x,den);predicted=evaluate_gain(program,compiled,gain,den)
        for branch in direct:gain_errors.append(float((direct[branch]-predicted[branch]).norm()/direct[branch].norm()))
    native=evaluate_gain(program,compiled,1.,den);checks.extend(float((native[b].cpu()-cache['writes'][str(b)]).norm()/cache['writes'][str(b)].norm()) for b in (3,8))
    norm1=compiled['norm'].sum(-1);scalar_sectors=compiled['sectors']/(norm1.square()*den)[:,None,None];writer=program['writers'][:,1].double().cuda()
    residual_write=scalar_sectors[:,1,0,None]*writer[None,:];attention_write=scalar_sectors[:,1,1:].sum(-1)[:,None]*writer[None,:]
    effects=score([cache['writes']['8'],residual_write.cpu(),attention_write.cpu()],state,rows,model.lm_head.weight.detach().cpu())
    gain0=evaluate_gain(program,compiled,0.,den)[8].cpu();gain0_effects=score([cache['writes']['8'],gain0,gain0],state,rows,model.lm_head.weight.detach().cpu())
    def passes(report):return all(f['swap_relative_rms']<=.1 and f['swap_sign_agreement']>=.9 and f['swap_live']>=4 for f in report['families'] if f['family']!='quoted_control')
    result={'pred_a':max(checks)<=1e-5 and max(gain_errors)<=1e-8 and effects['pred_a'],'pred_b':passes(effects['reports'][0]),'pred_c':passes(effects['reports'][1])}
    torch.save(dict(sources=source,compiled={k:v.cpu() for k,v in compiled.items()},scalar_sectors=scalar_sectors.cpu(),rows_sha=digest(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json')),ap)
    result.update(checks=checks,gain_errors=gain_errors,effects=effects,gain0_effects=gain0_effects,counts=counts,artifact_sha=digest(ap),source_shas=binding,
                  execution_seconds=time.perf_counter()-tic,scope='Conditional attention16 input-edge/source algebra; not full-model attention ablation or semantic identification.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','effects','gain0_effects')}),flush=True)

if __name__=='__main__':main()
