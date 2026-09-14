"""Exact source-position contributions of the reflection-odd key branch O."""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from inherited_source_positions_v1 import paired_masks


def source_channels(graph,current,first_values):
    """Return B,query,source,128 channels whose source sum equals O."""
    p=graph.p;x=current.double();batch,tokens=x.shape[:2]
    if first_values.shape not in ((batch,tokens,128),(batch,tokens,9,128),(batch,tokens,1152)):
        raise ValueError('First values must describe one head or all nine heads')
    if first_values.ndim==4:first_values=first_values[:,:,8]
    elif first_values.shape[-1]!=128:first_values=first_values.reshape(batch,tokens,9,128)[:,:,8]
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    angles=torch.outer(torch.arange(tokens,dtype=torch.float32),inv)
    co,si=angles.cos().bfloat16().to(x.device),angles.sin().bfloat16().to(x.device)
    def rotate(z):
        a,b=z.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
    nums=[x@p[k].double().T for k in ['k1','k2']]
    shared=torch.cat(nums,-1)@p['key_coordinates'];full=[];reflected=[]
    for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
        q=rotate(F.rms_norm(F.linear(current,p[qn].to(current.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double()
        key=F.linear(current,p[kn].to(current.dtype))
        den=(key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double()
        inside=shared@graph.adapters[j].T
        full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128)
        reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
    odd=(full[0]*full[1]-reflected[0]*reflected[1])/2
    odd=odd.masked_fill(~torch.ones(tokens,tokens,dtype=torch.bool,device=x.device).tril(),0)
    current_values=x@p['current_value'].double().T
    mixed=(1-p['mixture'].double())*current_values+p['mixture'].double()*first_values.double()
    return odd[...,None]*mixed[:,None,:,:]


def select_sources(channels,mask):
    mask=torch.as_tensor(mask,dtype=channels.dtype,device=channels.device)
    if mask.shape!=channels.shape[:1]+channels.shape[2:3]:
        raise ValueError('Mask must have B,source shape')
    return (channels*mask[:,None,:,None]).sum(-2)


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_SOURCE_POSITIONS_V1_CPU_CONTROL.json'
    assert not out.exists();torch.set_num_threads(2);torch.manual_seed(14091716)
    graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True))
    rows=json.loads((root/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows']
    masks=paired_masks(rows);errors=[];layouts=[];zeros=[]
    for i in [0,12,24,36]:
        tokens=len(rows[i]['ids']);current=F.rms_norm(torch.randn(1,tokens,1152),(1152,))
        first=torch.randn(1,tokens,9,128);channels=source_channels(graph,current,first)
        reference=graph.state(current,first_values=first[:,:,8])[2]
        errors.append(float((channels.sum(-2)-reference).norm()/reference.norm().clamp_min(1e-30)))
        flat=source_channels(graph,current,first.reshape(1,tokens,1152))
        layouts.append(float((flat-channels).norm()/channels.norm().clamp_min(1e-30)))
        mask=masks[i][None];city=select_sources(channels,mask);other=select_sources(channels,~mask)
        errors.append(float((city+other-reference).norm()/reference.norm().clamp_min(1e-30)))
        zeros.append(float(select_sources(channels,torch.zeros_like(mask)).abs().max()))
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(errors)<=1e-10 and max(layouts)==0 and max(zeros)==0,'source_sum_errors':errors,'layout_errors':layouts,'zero_mask_maxabs':zeros,'rows':len(rows),'mask_counts':[int(x.sum()) for x in masks],'additional_model_scalars':0,'scope':'Exact source-position partition of O with synthetic normalized current states, native weights and token-defined masks. No native causal or semantic conclusion.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='mask_counts'}))


if __name__=='__main__':main()
