"""Batched paired head8.2 writes through the fixed head9.8 odd-value descendant.

A caller declares arms and supplies a weight-defined write function. All six
endpoint readouts share a forward. V4 also supports native8_* arms applied at actual attention8 output.
Those arms recompute MLP8 and all later pathways; odd-only reentry is skipped.
Historical hash-bound runtimes are untouched.
"""
import torch
import torch.nn.functional as F
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts

CONTROL_PAIRS=[(3797,3290),(2266,4171),(3321,3431),(17180,10912)]

@torch.no_grad()
def measure(model,graph,rows,arms,write,odd_write=None):
    assert arms[0]=='native' and len(set(arms))==len(arms)
    assert len(rows)%2==0
    current_inputs=[];state={'arm':'native'};outside=[];write_norms={a:[] for a in arms[1:]}
    def capture(module,args):
        if state['arm']=='native':current_inputs.append(args[0].detach().cpu())
    def native8_output(module,args,result):
        if not state['arm'].startswith('native8_'):return result
        i=state['index'];row=rows[i];current=current_inputs[i].to(result[0].device);donor=current_inputs[i^1].to(result[0].device)
        mask=torch.zeros(current.shape[1],dtype=torch.bool,device=current.device);mask[row['destination_positions']]=True
        delta=write(state['arm'],row,rows[i^1],current,donor,mask)
        assert delta.shape==result[0].shape and bool(torch.isfinite(delta).all())
        outside.append(float(delta[:,~mask].abs().max()));write_norms[state['arm']].append(float(delta.norm()))
        return result[0]+delta.to(result[0].dtype),result[1]
    def reentry(module,args):
        if state['arm']=='native' or state['arm'].startswith('native8_'):return
        x,_,x0=args;i=state['index'];row=rows[i]
        current=current_inputs[i].to(x.device);donor=current_inputs[i^1].to(x.device)
        mask=torch.zeros(x.shape[1],dtype=torch.bool,device=x.device)
        mask[row['destination_positions']]=True
        delta=write(state['arm'],row,rows[i^1],current,donor,mask)
        assert delta.shape==x.shape and bool(torch.isfinite(delta).all())
        outside.append(float(delta[:,~mask].abs().max()))
        write_norms[state['arm']].append(float(delta.norm()))
        hybrid=x+delta*mask[None,:,None]
        mixed=module.lambdas[0]*hybrid+module.lambdas[1]*x0
        state['changed']=F.rms_norm(mixed,(mixed.shape[-1],))
        state['raw_mixed']=module.lambdas[0]*x+module.lambdas[1]*x0
        state['delta8']=delta
        state['lambda90']=module.lambdas[0]
        state['mask']=mask
    def descendant(module,args,result):
        if state['arm']=='native' or state['arm'].startswith('native8_'):return result
        current,first=args
        routing,_=source_factors(graph,current,current,current,first)
        original,_=value_parts(graph,current,first)
        changed,_=value_parts(graph,state['changed'],first)
        delta=select_sources(routing*(changed-original),state['mask'][None])@graph.p['output'].double().T
        if odd_write is not None:
            delta=odd_write(state['arm'],graph,state['raw_mixed'],state['delta8'],state['lambda90'],state['mask'],delta)
        return result[0]+delta.to(result[0].dtype),result[1]
    handles=[model.transformer.h[8].attn.register_forward_pre_hook(capture),
             model.transformer.h[8].attn.register_forward_hook(native8_output),
             model.transformer.h[9].register_forward_pre_hook(reentry),
             model.transformer.h[9].attn.register_forward_hook(descendant)]
    values=torch.zeros(len(arms),len(rows),len(rows[0]['endpoint_pairs'])+4,dtype=torch.float64)
    count=0
    try:
        for ai,arm in enumerate(arms):
            state['arm']=arm
            for i,row in enumerate(rows):
                state['index']=i
                ids=torch.tensor([row['ids']],device='cuda')
                x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
                for block in model.transformer.h:x,first=block(x,first,x0)
                scores=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0]
                for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):
                    values[ai,i,j]=(scores[left]-scores[right]).cpu()
                count+=1
            if ai==0:assert len(current_inputs)==len(rows)
    finally:
        for handle in handles:handle.remove()
    return {'values':values,'body_forwards':count,'outside':outside,'write_norms':write_norms}
