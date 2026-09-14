"""Separate odd routing and mixed-value factors for exact source-swap algebra."""
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from odd_source_positions_v1 import source_channels


def source_factors(graph,query_current,key_current,value_current,first_values):
    p=graph.p;xq=query_current;xk=key_current;xv=value_current;batch,tokens=xq.shape[:2]
    if first_values.ndim==4:first_values=first_values[:,:,8]
    elif first_values.shape[-1]!=128:first_values=first_values.reshape(batch,tokens,9,128)[:,:,8]
    inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angles=torch.outer(torch.arange(tokens,dtype=torch.float32),inv);co,si=angles.cos().bfloat16().to(xq.device),angles.sin().bfloat16().to(xq.device)
    def rotate(z):a,b=z.chunk(2,-1);return torch.cat((a*co+b*si,-a*si+b*co),-1)
    nums=[xk.double()@p[k].double().T for k in ['k1','k2']];shared=torch.cat(nums,-1)@p['key_coordinates'];full=[];reflected=[]
    for j,(qn,kn) in enumerate([('q1','k1'),('q2','k2')]):
        q=rotate(F.rms_norm(F.linear(xq,p[qn].to(xq.dtype)),(128,),eps=torch.finfo(torch.float32).eps)).double();key=F.linear(xk,p[kn].to(xk.dtype));den=(key.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt().double();inside=shared@graph.adapters[j].T
        full.append(q@rotate(nums[j]/den).transpose(-1,-2)/128);reflected.append(q@rotate((nums[j]-2*inside)/den).transpose(-1,-2)/128)
    odd=(full[0]*full[1]-reflected[0]*reflected[1])/2;odd=odd.masked_fill(~torch.ones(tokens,tokens,dtype=torch.bool,device=xq.device).tril(),0)
    current_values=xv.double()@p['current_value'].double().T;mixed=(1-p['mixture'].double())*current_values+p['mixture'].double()*first_values.double()
    return odd[...,None],mixed[:,None,:,:]


def separated_channels(graph,query_current,key_current,value_current,first_values):
    routing,values=source_factors(graph,query_current,key_current,value_current,first_values);return routing*values


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_SOURCE_SWAP_INTERACTION_V1_CPU_CONTROL.json';assert not out.exists();torch.set_num_threads(2);torch.manual_seed(14091753)
    graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True));replay=[];algebra=[];mixed=[]
    for tokens in (23,24):
        x=F.rms_norm(torch.randn(1,tokens,1152),(1152,));d=F.rms_norm(torch.randn(1,tokens,1152),(1152,));f=torch.randn(1,tokens,9,128);fd=torch.randn_like(f)
        r0,v0=source_factors(graph,x,x,x,f);rd,vd=source_factors(graph,x,d,d,fd);orig=r0*v0;full=rd*vd;route=rd*v0;value=r0*vd;interaction=(rd-r0)*(vd-v0)
        replay.append(float((orig-source_channels(graph,x,f)).norm()/orig.norm().clamp_min(1e-30)));algebra.append(float((full-orig-(route-orig)-(value-orig)-interaction).norm()/(full-orig).norm().clamp_min(1e-30)));mixed.append(float(interaction.norm()))
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(replay)<=1e-10 and max(algebra)<=1e-10 and min(mixed)>1e-8,'source_helper_replay_errors':replay,'swap_expansion_errors':algebra,'mixed_term_norms':mixed,'scope':'Exact synthetic routing/value/mixed expansion of O source channels; no native behavioral conclusion or fitting.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
