"""Conditional MLP16-versus-background producer contributions, no fitting.
A prior residual-only margin replay<=1e-4 nats, p+g=r identity<=1e-9 relative.
B exact AND approximate p-only meanabs>=20%residual-only and positive eachtaskcell.
C approximate/exact p-only effectrelativeRMS<=.25 and sign>=.9 eachtaskcell.
D approximate p/g margincross<=10%residualmeanabs eachtaskfamily.
Native attention remains at base; component-only edge edit, not whole-module ablation.
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
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'NATIVE_SUFFIX_MLP16_PRODUCER_V1.json';assert not out.exists()
    prior=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text());assert prior['pred_b']
    binding=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    up=torch.load(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt',weights_only=True,map_location='cpu')
    assert digest(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt')==prior['cache_sha256']
    cache=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')
    rows=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    program=torch.load(p/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    split=torch.load(p/'NATIVE_RELATION_OUTPUT_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    writer=split['splits']['ambient']['private_writers']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].float();left,right,down=[state['transformer.h.17.mlp.'+name+'.weight'].float() for name in ('Left','Right','Down')]
    bias=state['transformer.h.17.mlp.Down_bias'].float()
    residual=up['ports']['residual'].double();producer=up['block17_lambdas'][0].double()*up['ports']['mlp16_output'].double();background=residual-producer
    identity=float((producer+background-residual).norm()/residual.norm())
    pre=cache['ports']['pre'];h=pre+cache['ports']['native_output'];records=[];replay=[]
    for i,row in enumerate(rows):
        b,d=2*i,2*i+1;ub=pre[b].double()
        mixed=torch.stack([ub,ub+residual[d]-residual[b],ub+producer[d]-producer[b],ub+background[d]-background[b]]).float()
        x=F.rms_norm(mixed,(1152,))
        lead,rest=evaluate(program,x.double());approx=lead+rest
        native=F.linear(F.linear(x,left)*F.linear(x,right),down)+bias
        exact=native.double()@program['readouts'].T
        reader=u[[row['donor_answer_id'],row['donor_foil_id']]]
        effects=[]
        for scalar in (approx,exact):
            states=h[b]+((scalar-scalar[:1])@writer.T).float()
            z=30*torch.tanh(F.linear(F.rms_norm(states,(1152,)),reader)/30)
            margin=(z[:,0]-z[:,1]).double();effects.append(margin-margin[0])
        replay.append(abs(float(effects[0][1])-prior['records'][i]['effects'][2]))
        records.append(dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],
            approximate_effects=effects[0].tolist(),exact_effects=effects[1].tolist(),
            approximate_cross=float(effects[0][1]-effects[0][2]-effects[0][3])))
    cells=[]
    for cell in prior['cells']:
        local=[r for r in records if r['family']==cell['family'] and r['direction']==cell['direction']]
        a=torch.tensor([r['approximate_effects'] for r in local],dtype=torch.float64)
        e=torch.tensor([r['exact_effects'] for r in local],dtype=torch.float64)
        cells.append(dict(family=cell['family'],direction=cell['direction'],approx_means=a.mean(0).tolist(),exact_means=e.mean(0).tolist(),
            approx_producer_fraction=float(a[:,2].abs().mean()/a[:,1].abs().mean()),exact_producer_fraction=float(e[:,2].abs().mean()/e[:,1].abs().mean()),
            producer_effect_relative_rms=float((a[:,2]-e[:,2]).norm()/e[:,2].norm()),
            producer_sign_agreement=float((a[:,2].sign()==e[:,2].sign()).double().mean())))
    compositions=[]
    for family in ('A1','A2'):
        local=[r for r in records if r['family']==family]
        cross=sum(abs(r['approximate_cross']) for r in local);total=sum(abs(r['approximate_effects'][1]) for r in local)
        compositions.append(dict(family=family,relative_cross=cross/total))
    tested=[c for c in cells if c['family'] in ('A1','A2')]
    result=dict(pred_a=max(replay)<=1e-4 and identity<=1e-9,
        pred_b=all(min(c['approx_producer_fraction'],c['exact_producer_fraction'])>=.2 and min(c['approx_means'][2],c['exact_means'][2])>0 for c in tested),
        pred_c=all(c['producer_effect_relative_rms']<=.25 and c['producer_sign_agreement']>=.9 for c in tested),
        pred_d=all(c['relative_cross']<=.1 for c in compositions),max_replay_nats=max(replay),partition_identity_error=identity,
        cells=cells,compositions=compositions,records=records,source_upstream_result_sha256=digest(p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json'),
        scope='Conditional native/approximate suffix-component effect from MLP16 output versus remaining incoming residual, fixed base attention17. No whole-module causal claim or closed extraction.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
