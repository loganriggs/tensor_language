"""Exact current versus inherited-first factors of head9.8 mixed values."""
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from even_value_shared_graph_v1 import SharedGraph
from odd_source_swap_interaction_v1 import source_factors


def value_parts(graph,value_current,first_values):
    batch,tokens=value_current.shape[:2]
    if first_values.ndim==4:first_values=first_values[:,:,8]
    elif first_values.shape[-1]!=128:first_values=first_values.reshape(batch,tokens,9,128)[:,:,8]
    mixture=graph.p['mixture'].double();current=(1-mixture)*(value_current.double()@graph.p['current_value'].double().T);inherited=mixture*first_values.double()
    return current[:,None,:,:],inherited[:,None,:,:]


@torch.no_grad()
def main():
    root=Path(__file__).resolve().parent;out=root/'ODD_VALUE_SOURCE_SPLIT_V1_CPU_CONTROL.json';assert not out.exists();torch.set_num_threads(2);torch.manual_seed(14091757)
    graph=SharedGraph(torch.load(root/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True));factor=[];delta=[];live=[]
    for tokens in (23,24):
        x=F.rms_norm(torch.randn(1,tokens,1152),(1152,));d=F.rms_norm(torch.randn(1,tokens,1152),(1152,));f=torch.randn(1,tokens,9,128);fd=torch.randn_like(f)
        _,v=source_factors(graph,x,x,x,f);co,io=value_parts(graph,x,f);cd,id_=value_parts(graph,d,fd);full=cd+id_;current=cd+io;inherited=co+id_
        factor.append(float((co+io-v).norm()/v.norm().clamp_min(1e-30)));delta.append(float((full-v-(current-v)-(inherited-v)).norm()/(full-v).norm().clamp_min(1e-30)));live.extend([float((current-v).norm()),float((inherited-v).norm())])
    result={'utc':datetime.now(timezone.utc).isoformat(),'pred_a':max(factor)<=1e-10 and max(delta)<=1e-10 and min(live)>1e-8,'value_factor_errors':factor,'donor_delta_recomposition_errors':delta,'source_delta_norms':live,'scope':'Exact synthetic current/inherited split of mixed values; no native behavioral conclusion or fitting.'};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
