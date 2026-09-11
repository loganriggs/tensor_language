"""Post-screen normalization accounting, preserving neighboring-selectivity misses.
A actual CE/margin replay<=1e-4 nats, CE difference identity<=1e-9.
B fixed-denominator versus actual CE-effect meanabsdifference<=.01 eachfamily/arm.
C difference RMS / actual-effect RMS<=.25 eachfamily/arm.
No fitting; full-vocabulary native tail, CPU only, 8-row batches.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from native_relation_split_v1 import evaluate


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;stem='NATIVE_RELATION_NEIGHBOR_V1'
    output=p/'NATIVE_RELATION_NEIGHBOR_NORM_V1.json';assert not output.exists()
    prior=json.loads((p/(stem+'_RESULT.json')).read_text());assert prior['pred_a']
    binding=json.loads((p/(stem+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    cache=torch.load(p/(stem+'_ENDPOINTS.pt'),weights_only=True,map_location='cpu')
    assert digest(p/(stem+'_ENDPOINTS.pt'))==prior['cache_sha256']
    assert cache['rows_sha256']==digest(p/(stem+'_ROWS.json'))
    rows=json.loads((p/(stem+'_ROWS.json')).read_text())['rows']
    program=torch.load(p/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].float();ports=cache['ports'];x=ports['input'].double()
    lead,rest=evaluate(program,x);whole=(lead+rest)@program['writers'].T
    exact=ports['native_output'].double()@program['readouts'].T@program['writers'].T
    h=ports['pre']+ports['native_output'];records=[];replay=[];identities=[]
    for start in range(0,64,8):
        hb=h[start:start+8]
        denominator=torch.sqrt(hb.square().mean(-1,keepdim=True)+torch.finfo(hb.dtype).eps)
        z0=30*torch.tanh(F.linear(F.rms_norm(hb,(1152,)),u)/30)
        targets=torch.tensor([rows[j//2][('base' if j%2==0 else 'donor')+'_answer_id'] for j in range(start,start+len(hb))])
        ce0=F.cross_entropy(z0.double(),targets,reduction='none')
        lz0=torch.logsumexp(z0.double(),-1)
        for name,write,arm in [('whole',whole,3),('exact_span',exact,4)]:
            modified=hb-write[start:start+len(hb)].float()
            actual=30*torch.tanh(F.linear(F.rms_norm(modified,(1152,)),u)/30)
            fixed=30*torch.tanh(F.linear(modified/denominator,u)/30)
            actual_ce=F.cross_entropy(actual.double(),targets,reduction='none')
            fixed_ce=F.cross_entropy(fixed.double(),targets,reduction='none')
            target_delta=(actual-z0).double().gather(1,targets[:,None]).squeeze(1)
            partition_delta=torch.logsumexp(actual.double(),-1)-lz0
            identities.append(float(((actual_ce-ce0)-(partition_delta-target_delta)).abs().max()))
            for k,j in enumerate(range(start,start+len(hb))):
                row=rows[j//2];old=prior['records'][j//2]['endpoints'][j%2]
                margin=float(actual[k,row['donor_answer_id']]-actual[k,row['donor_foil_id']])
                replay.extend([abs(float(actual_ce[k])-old['ce'][arm]),abs(margin-old['margin'][arm])])
                records.append(dict(family=row['family'],verb=row['verb'],side=old['side'],arm=name,
                    actual_effect=float(actual_ce[k]-ce0[k]),fixed_norm_effect=float(fixed_ce[k]-ce0[k]),
                    norm_correction=float(actual_ce[k]-fixed_ce[k]),target_logit_delta=float(target_delta[k]),
                    logpartition_delta=float(partition_delta[k])))
    cells=[]
    for family in ('past','progressive'):
        for arm in ('whole','exact_span'):
            local=[r for r in records if r['family']==family and r['arm']==arm]
            real=torch.tensor([r['actual_effect'] for r in local],dtype=torch.float64)
            fixed=torch.tensor([r['fixed_norm_effect'] for r in local],dtype=torch.float64)
            correction=real-fixed
            cells.append(dict(family=family,arm=arm,actual_meanabs=float(real.abs().mean()),fixed_meanabs=float(fixed.abs().mean()),
                norm_correction_meanabs=float(correction.abs().mean()),norm_relative_rms=float(correction.norm()/real.norm()),
                side_mean_actual={s:sum(r['actual_effect'] for r in local if r['side']==s)/16 for s in ('base','donor')}))
    result=dict(pred_a=max(replay)<=1e-4 and max(identities)<=1e-9,
        pred_b=all(c['norm_correction_meanabs']<=.01 for c in cells),pred_c=all(c['norm_relative_rms']<=.25 for c in cells),
        max_replay_nats=max(replay),max_identity_error=max(identities),cells=cells,records=records,
        source_result_sha256=digest(p/(stem+'_RESULT.json')),
        scope='Known fixed-denominator tail accounting. Original control preservation misses remain. No fit or repaired verdict.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
