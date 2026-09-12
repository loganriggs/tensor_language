"""Descriptive off-pair token rankings after the frozen full-vocabulary screen."""
from pathlib import Path
import json
import torch,tiktoken
import torch.nn.functional as F
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
@torch.no_grad()
def main():
    out=P/'REGIONAL_FIRST_OFFPAIR_PROFILE_V1.json';assert not out.exists();torch.set_num_threads(2);enc=tiktoken.get_encoding('gpt2')
    sd=torch.load(CK,mmap=True,weights_only=True,map_location='cpu');L,R,D=[sd['transformer.h.17.mlp.'+n+'.weight'].float() for n in ('Left','Right','Down')];bias=sd['transformer.h.17.mlp.Down_bias'].float();U=sd['lm_head.weight'].float();panels={}
    for suffix,rowname in [('V1','V2'),('OOD_V1','OOD_V1')]:
        rows=json.loads((P/f'REGIONAL_SOURCE_BLOCK_{rowname}_ROWS.json').read_text())['rows'];a=torch.load(P/f'REGIONAL_PRODUCER_VALUE_STREAM_{suffix}_ARTIFACT.pt',weights_only=True,map_location='cpu');pre=a['pre'];w=a['write_vertices']
        def scores(delta):
            z=pre+delta.float();x=F.rms_norm(z,(1152,));h=z+((x@L.T)*(x@R.T))@D.T+bias;return (30*torch.tanh((F.rms_norm(h,(1152,))@U.T)/30)).double()
        b=scores(torch.zeros_like(pre));e=scores(w[:,2]-w[:,0]);logp=b.log_softmax(-1);logq=e.log_softmax(-1);prob=logp.exp();mask=torch.ones_like(prob)
        for i,row in enumerate(rows):mask[i,row['uk_id']]=0;mask[i,row['us_id']]=0
        weighted=prob*(logq-logp).square()*mask;raw=(e-b).abs()*mask;cells=[]
        for family in sorted(set(r['family'] for r in rows)):
            ids=[i for i,r in enumerate(rows) if r['family']==family];weighted_mean=weighted[ids].mean(0);raw_mean=raw[ids].mean(0)
            def describe(order):
                return [dict(token_id=int(t),token=enc.decode([int(t)]) if int(t)<enc.n_vocab else '[unused vocabulary row]',weighted_squared_logprob_change=float(weighted_mean[t]),mean_absolute_logit_change=float(raw_mean[t]),mean_baseline_probability=float(prob[ids,t].mean())) for t in order]
            cells.append(dict(family=family,largest_probability_weighted_changes=describe(weighted_mean.topk(12).indices),largest_absolute_logit_changes=describe(raw_mean.topk(12).indices)))
        panels[suffix]=cells
    result=dict(panels=panels,scope='Post-result descriptive rankings. Each row excludes its tested UK/US target pair only; a target for another row can appear. Probability-weighted ranking uses full softmax; this differs from the primary off-pair conditional-distribution metric. Raw-logit ranks can emphasize extremely unlikely tokens; no new semantic or selectivity threshold.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    for name,cells in panels.items():
        for c in cells:print(name,c['family'],'weighted',[x['token'] for x in c['largest_probability_weighted_changes']],'raw',[x['token'] for x in c['largest_absolute_logit_changes']])
if __name__=='__main__':main()
