"""Exact alternating fixed-group low-rank plus sparse correction countercheck."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from interaction_shared_write_subspaces_v1 import update,reconstruction
P=Path(__file__).resolve().parent

@torch.no_grad()
def main(max_steps=100, time_budget=60, result_prefix='INTERACTION_SHARED_PRIVATE_REFINE_V1'):
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build()
    x=t.permute(1,2,0).reshape(-1,12).contiguous();total=float(x.square().sum())
    prior=torch.load(P/'INTERACTION_SHARED_WRITE_POLISH_V1_PROGRAM.pt',weights_only=True)
    assert ids==prior['token_ids'];labels=prior['groups'].long()
    q=prior['output_bases'].double()[:,:,:1];sparse=torch.zeros_like(x);count=984234
    history=[];consecutive=0;stop='step_limit'
    for step in range(max_steps+1):
        target=x-sparse;q=update(target,labels,q)
        codes=torch.einsum('ndr,nd->nr',q[labels],target);common=reconstruction(labels,codes,q)
        residual=x-common;values,order=residual.flatten().square().sort(descending=True)
        flat=torch.zeros_like(residual.flatten());flat[order[:count]]=residual.flatten()[order[:count]];sparse=flat.reshape_as(x)
        loss=float(values[count:].sum()/total)
        gain=None if not history else (history[-1]['loss']-loss)/max(history[-1]['loss'],1e-30)
        if history:assert loss<=history[-1]['loss']+1e-10
        consecutive=consecutive+1 if gain is not None and 0<=gain<=1e-7 else 0
        history.append(dict(step=step,loss=loss,relative_improvement=gain))
        if consecutive>=2:stop='relative_objective_tolerance';break
        if time.perf_counter()-start>=time_budget:stop='time_limit';break
    fit=common+sparse;replay=abs(float((x-fit).square().sum()/total)-loss);assert replay<1e-12
    result=dict(pred_a=True,pred_b=loss<=.01,pred_c=consecutive>=2,stop=stop,history=history,
        relative_error=loss**.5,nominal_bytes=4896936,objective_replay_error=replay,seconds=time.perf_counter()-start,
        scope='One fixed-group rank1 alternating PCA/sparse-support fit. Objective tolerance is not global or full coordinate stationarity; no serialized or native validation claim. Original sequential failure retained.')
    (P/(result_prefix+'_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2));print('last',history[-1])

if __name__=='__main__':main()
