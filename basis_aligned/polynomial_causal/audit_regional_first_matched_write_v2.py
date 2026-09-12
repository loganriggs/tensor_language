"""Matched norm and pairwise-geometry controls for frozen first-branch writes."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
@torch.no_grad()
def main():
    out=P/'REGIONAL_FIRST_MATCHED_WRITE_V2_RESULT.json';assert not out.exists();torch.set_num_threads(2);torch.manual_seed(61212)
    rows=json.loads((P/'REGIONAL_BEHAVIOR_CONTROLS_V2_ROWS.json').read_text())['rows'];validate(rows);a=torch.load(P/'REGIONAL_BEHAVIOR_CONTROLS_V2_ARTIFACT.pt',weights_only=True);delta=a['write_vertices'][:,1]-a['write_vertices'][:,0];pre=a['pre'];sd=torch.load(CK,mmap=True,weights_only=True,map_location='cpu')
    L,R,D=[sd['transformer.h.17.mlp.'+n+'.weight'].float() for n in ('Left','Right','Down')];bias=sd['transformer.h.17.mlp.Down_bias'].float();ids=torch.tensor([[x['uk_id'],x['us_id'],*x['control_ids']] for x in rows]);U=sd['lm_head.weight'][ids].float()
    def margin(change):
        z=pre+change.float();x=F.rms_norm(z,(1152,));h=z+((x@L.T)*(x@R.T))@D.T+bias;s=30*torch.tanh(torch.einsum('nd,nkd->nk',F.rms_norm(h,(1152,)),U)/30);return torch.stack([s[:,0]-s[:,1],s[:,2]-s[:,3]],-1).double()
    base=margin(torch.zeros_like(pre));actual=margin(delta);ref=a['arm_margins'];error=float((torch.stack([base,actual])-ref).norm()/ref.norm());reguk=[i for i,x in enumerate(rows) if x['family']==0 and x['cue']=='British'];regus=[i+1 for i in reguk]
    def summarize(m):
        effect=base-m;return dict(regional_transfer=float(((effect[reguk,0]-effect[regus,0])/2).mean()),control_meanabs={str(j):float(effect[[i for i,x in enumerate(rows) if x['family']==j],0].abs().mean()) for j in (1,2,3)})
    true=summarize(actual);controls=[];errors=[];gram=delta@delta.T
    for seed_index in range(16):
        permutation=torch.randperm(1152);sign=(2*torch.randint(0,2,(1152,))-1).double();changed=delta[:,permutation]*sign
        errors.append(float((changed@changed.T-gram).norm()/gram.norm()));controls.append(summarize(margin(changed)))
    q95=float(torch.quantile(torch.tensor([c['regional_transfer'] for c in controls],dtype=torch.float64),.95));pred_a=error<=1e-5 and max(errors)<=1e-12
    result=dict(pred_a=pred_a,pred_b=pred_a and true['regional_transfer']>q95,replay_relative_error=error,max_relative_gram_error=max(errors),actual=true,control_95th_percentile=q95,controls=controls,scope='Fixed signed-coordinate permutation controls, preserving row norms/Gram. Not Haar rotations, globalselectivity or four-property promotion.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
