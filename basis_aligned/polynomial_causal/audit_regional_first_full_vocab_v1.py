"""CPU full-readout selectivity audit from saved native first-value interventions."""
from pathlib import Path
import json,hashlib
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
@torch.no_grad()
def main():
    out=P/'REGIONAL_FIRST_FULL_VOCAB_V1_RESULT.json';assert not out.exists();torch.set_num_threads(2)
    sd=torch.load(CK,mmap=True,weights_only=True,map_location='cpu');l,r,d=[sd['transformer.h.17.mlp.'+n+'.weight'].float() for n in ('Left','Right','Down')];bias=sd['transformer.h.17.mlp.Down_bias'].float();U=sd['lm_head.weight'].float();panels={};allcells=[];errors=[]
    for suffix,rowname in [('V1','V2'),('OOD_V1','OOD_V1')]:
        rows=json.loads((P/f'REGIONAL_SOURCE_BLOCK_{rowname}_ROWS.json').read_text())['rows'];a=torch.load(P/f'REGIONAL_PRODUCER_VALUE_STREAM_{suffix}_ARTIFACT.pt',weights_only=True,map_location='cpu');pre=a['pre'];w=a['write_vertices']
        def scores(delta):
            z=pre+delta.float();x=F.rms_norm(z,(1152,));h=z+((x@l.T)*(x@r.T))@d.T+bias
            return (30*torch.tanh((F.rms_norm(h,(1152,))@U.T)/30)).double()
        base=scores(torch.zeros_like(pre));edit=scores(w[:,2]-w[:,0]);indices=torch.arange(len(rows));uk=torch.tensor([x['uk_id'] for x in rows]);us=torch.tensor([x['us_id'] for x in rows]);c0=torch.tensor([x['control_ids'][0] for x in rows]);c1=torch.tensor([x['control_ids'][1] for x in rows])
        computed=torch.stack([torch.stack([z[indices,uk]-z[indices,us],z[indices,c0]-z[indices,c1]],-1) for z in (base,edit)])
        reference=a['arm_margins'][[0,2]];error=float((computed-reference).norm()/reference.norm());errors.append(error)
        logp=base.log_softmax(-1);logq=edit.log_softmax(-1);prob=logp.exp();target_effect=(edit[indices,uk]-edit[indices,us])-(base[indices,uk]-base[indices,us])
        pairlogp=torch.logsumexp(logp[indices[:,None],torch.stack([uk,us],1)],-1);pairlogq=torch.logsumexp(logq[indices[:,None],torch.stack([uk,us],1)],-1)
        fullkl=(prob*(logp-logq)).sum(-1);fulltv=.5*(prob-logq.exp()).abs().sum(-1)
        bp=base.clone();eq=edit.clone();bp[indices,uk]=float('-inf');bp[indices,us]=float('-inf');eq[indices,uk]=float('-inf');eq[indices,us]=float('-inf');lp=bp.log_softmax(-1);lq=eq.log_softmax(-1);p=lp.exp();dl=lq-lp;dl[indices,uk]=0;dl[indices,us]=0
        sq=(p*dl.square()).sum(-1);kl=-(p*dl).sum(-1);tv=.5*(p-lq.exp()).abs().sum(-1);cells=[]
        for family in sorted(set(x['family'] for x in rows)):
            ids=[i for i,x in enumerate(rows) if x['family']==family];targetrms=target_effect[ids].square().mean().sqrt();offrms=sq[ids].mean().sqrt()
            cell=dict(family=family,target_logodds_effect_rms=float(targetrms),offpair_logprob_effect_rms=float(offrms),offpair_to_target_ratio=float(offrms/targetrms),offpair_kl_mean=float(kl[ids].mean()),offpair_tv_mean=float(tv[ids].mean()),full_kl_mean=float(fullkl[ids].mean()),full_tv_mean=float(fulltv[ids].mean()),baseline_target_pair_probability_mean=float(pairlogp[ids].exp().mean()),target_pair_logmass_change_rms=float((pairlogq[ids]-pairlogp[ids]).square().mean().sqrt()),prefix_offpair_to_target_ratios=(sq[ids].sqrt()/target_effect[ids].abs().clamp_min(1e-12)).tolist())
            cells.append(cell);allcells.append(cell)
        panels[suffix]=dict(replay_relative_error=error,cells=cells)
    pred_a=max(errors)<=1e-5;pred_b=pred_a and all(c['offpair_to_target_ratio']<=.25 for c in allcells);pred_c=pred_b and all(c['offpair_tv_mean']<=.005 for c in allcells)
    result=dict(pred_a=pred_a,pred_b=pred_b,pred_c=pred_c,panels=panels,scope='Strict off-pair vocabulary endpoint screen; off-pair is not equivalent to semantically unrelated. First-valueswap withnativebackground, not full-network circuitremoval. No arrays saved.',script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
