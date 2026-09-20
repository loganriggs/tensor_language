"""Fresh native capture against the corrected residual6-to-attention8 fold."""
from pathlib import Path
import hashlib,json,os,time,torch
import torch.nn.functional as F
from types import MethodType
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
import sys
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows
from city_attention7_exact_upstream_v1 import execute as exact_upstream
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();stem='CITY_COUPLED_UPSTREAM_FRESH_V1';rows=json.loads((P/(stem+'_ROWS.json')).read_text())['rows'];groups,_=group_rows(rows);assert len(groups)==40
    binding=json.loads((P/(stem+'_BINDING.json')).read_text())['files'];ck=next(k for k in binding if k.endswith('pytorch_model.bin'));sd=torch.load(ck,weights_only=True,mmap=True)
    tokens=torch.tensor(sorted({int(t) for r in groups for t in r['ids']}));initial=F.rms_norm(sd['transformer.wte.weight'][tokens].float(),(1152,));first=F.linear(F.rms_norm(sd['transformer.h.0.lambdas'][0]*initial+sd['transformer.h.0.lambdas'][1]*initial,(1152,)),sd['transformer.h.0.attn.c_v.weight']);attn={'token_ids':tokens,'initial_table':initial.clone(),'first_table':first.clone(),'lambdas8':sd['transformer.h.8.lambdas'].clone(),'lambdas7':sd['transformer.h.7.lambdas'].clone(),'mixture':sd['transformer.h.7.attn.lamb'].clone()}
    for key,name in [('q1','c_q'),('k1','c_k'),('q2','c_q2'),('k2','c_k2'),('value','c_v'),('output','c_proj')]:attn[key]=sd[f'transformer.h.7.attn.{name}.weight'].clone()
    reader=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True);head=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True);program={'attention7':attn,'readers':reader,'head8':head,'mlp7_left':sd['transformer.h.7.mlp.Left.weight'],'mlp7_right':sd['transformer.h.7.mlp.Right.weight'],'mlp7_down':sd['transformer.h.7.mlp.Down.weight'],'mlp7_bias':sd['transformer.h.7.mlp.Down_bias']}
    from fastload import load_model_fast
    model=load_model_fast().eval();ids=torch.tensor([g['ids'] for g in groups]);city=groups[0]['city_position'];destination=torch.tensor([g['destination_positions'] for g in groups],dtype=torch.long)
    # The fixed windows have one destination mask per group; use the first group's mask for the native city write.
    mask=torch.zeros(40,32,dtype=torch.bool)
    for i,g in enumerate(groups):mask[i,g['destination_positions']]=True
    cache={};attn8=model.transformer.h[8].attn;old8=attn8.squared_attention
    def capture(module,q,k,v,q2,k2):
        route=((q[:,:,2]*k[:,city,None,2]).sum(-1)/128)*((q2[:,:,2]*k2[:,city,None,2]).sum(-1)/128)
        # The exact upstream operator is a city-removal intervention. Keep
        # this native receipt on the same intervention; donor replacement is
        # a different selective/composition experiment.
        cache['delta']=-F.linear(route[...,None]*v[:,city,None,2],module.c_proj.weight[:,256:384])*mask[...,None]
        return old8(q,k,v,q2,k2)
    h8=attn8.squared_attention=MethodType(capture,attn8)
    def capture_residual6(module,args):
        cache.setdefault('residual6',args[0].clone())
    h7=model.transformer.h[7].register_forward_pre_hook(capture_residual6)
    x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
    try:
        for block in model.transformer.h:x,first=block(x,first,x0)
    finally:h7.remove();attn8.squared_attention=old8
    generated=[];errors=[];outside=[]
    for i,g in enumerate(groups):
        got=exact_upstream(program,cache['residual6'][i:i+1],ids[i:i+1],city,mask[i],return_rho=False);generated.append(got);errors.append(float((got-cache['delta'][i:i+1]).norm()/cache['delta'][i:i+1].norm()));outside.append(float(got[:,~mask[i]].abs().max()))
    r={'pred_a':max(errors)<=.05,'pred_b':max(outside)==0 and bool(torch.isfinite(torch.stack(generated)).all()),'max_relative_write_error':max(errors),'mean_relative_write_error':sum(errors)/len(errors),'max_off_support':max(outside),'documents':20,'sequences':40,'rows':240,'token_table':len(tokens),'scope':'Fresh outcome-blind FineWeb native capture against corrected exact-RMS residual6-to-attention8 generator; full coupled suffix and selective/composition gates remain next.','seconds':time.perf_counter()-start,'source_shas':binding}
    torch.save({'native_delta':cache['delta'],'generated_delta':torch.cat(generated),'residual6':cache['residual6'],'token_ids':ids},P/(stem+'_ARTIFACT.pt'));(P/(stem+'_RESULT.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
