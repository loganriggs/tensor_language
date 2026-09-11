"""Fresh exact native-span versus approximate conditional effects, CPU only.
A approximate GPU margin replay<=1e-4 nats.
B whole/private effect relative RMS<=.25 and sign>=.9 each target/direction cell.
C whole/private paired write relative RMS<=.25 each target/direction cell.
Native input/background remain; not absolute extraction or a fitted repair.
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
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'NATIVE_RELATION_OUTPUT_FRESH_EXACT_V1.json';assert not out.exists()
    stem='NATIVE_RELATION_OUTPUT_FRESH_V1';prior=json.loads((p/(stem+'_RESULT.json')).read_text());assert prior['pred_a']
    binding=json.loads((p/(stem+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    cache=torch.load(p/(stem+'_ENDPOINTS.pt'),weights_only=True,map_location='cpu');ports=cache['ports']
    assert digest(p/(stem+'_ENDPOINTS.pt'))==prior['cache_sha256']
    assert cache['rows_sha256']==digest(p/(stem+'_ROWS.json'))
    rows=json.loads((p/(stem+'_ROWS.json')).read_text())['rows']
    program=torch.load(p/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    split=torch.load(p/'NATIVE_RELATION_OUTPUT_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    matrices=[program['writers'],split['splits']['ambient']['private_writers'],split['splits']['ambient']['shared_writers']]
    lead,rest=evaluate(program,ports['input'].double());approx=lead+rest
    exact=ports['native_output'].double()@program['readouts'].T
    h=ports['pre']+ports['native_output']
    weights=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=weights['lm_head.weight'].float();records=[];replay=[]
    for i,row in enumerate(rows):
        b,d=2*i,2*i+1;reader=u[[row['donor_answer_id'],row['donor_foil_id']]]
        def margin(state):
            z=30*torch.tanh(F.linear(F.rms_norm(state,(1152,)),reader)/30)
            return float(z[0]-z[1])
        baseline=margin(h[b]);arms=[]
        for j,v in enumerate(matrices):
            wd=v@(exact[d]-exact[b]);wa=v@(approx[d]-approx[b])
            actual_effect=margin(h[b]+wd.float())-baseline
            approx_effect=margin(h[b]+wa.float())-baseline
            replay.append(abs(approx_effect-prior['records'][i]['arms'][j]['swap_effect']))
            arms.append(dict(exact_effect=actual_effect,approx_effect=approx_effect,
                exact_write_energy=float(wd.square().sum()),error_write_energy=float((wa-wd).square().sum())))
        records.append(dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],arms=arms))
    cells=[]
    for c in prior['cells']:
        local=[r for r in records if r['family']==c['family'] and r['direction']==c['direction']]
        arms=[]
        for j in range(3):
            exact_effect=torch.tensor([r['arms'][j]['exact_effect'] for r in local],dtype=torch.float64)
            approx_effect=torch.tensor([r['arms'][j]['approx_effect'] for r in local],dtype=torch.float64)
            arms.append(dict(exact_mean=float(exact_effect.mean()),approx_mean=float(approx_effect.mean()),
                effect_relative_rms=float((exact_effect-approx_effect).norm()/exact_effect.norm()),
                sign_agreement=float((exact_effect.sign()==approx_effect.sign()).double().mean()),
                write_relative_rms=(sum(r['arms'][j]['error_write_energy'] for r in local)/sum(r['arms'][j]['exact_write_energy'] for r in local))**.5))
        cells.append(dict(family=c['family'],direction=c['direction'],arms=arms))
    tested=[a for c in cells if c['family'] in ('A1','A2') for a in c['arms'][:2]]
    result=dict(pred_a=max(replay)<=1e-4,pred_b=all(a['effect_relative_rms']<=.25 and a['sign_agreement']>=.9 for a in tested),
        pred_c=all(a['write_relative_rms']<=.25 for a in tested),max_gpu_margin_replay_nats=max(replay),cells=cells,records=records,
        source_result_sha256=digest(p/(stem+'_RESULT.json')),
        scope='Frozen conditional differences compared with exact native MLP readout-span differences. Full input and native background remain; no absolute extraction or data fitting.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
