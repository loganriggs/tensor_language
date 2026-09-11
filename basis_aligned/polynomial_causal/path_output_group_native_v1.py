"""Frozen centered leading8 Schmidt group native replication; full96 control.
A source replay<=1e-5 and group identity<=1e-9. Primary centered8:
B physical symrelativeRMS<=.25. C swaps relative<=.25/sign>=.9 eachfamily.
D removalCE meanabs disagreement<=.02/sign>=.9 eachfamily.
Live floors1e-4 swaps/.001CE; >=4live percell. Existing developmental cache.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from sparse_path_program_v1 import run
from sparse_path_stability_atlas_v1 import digest
from stable_path_native_cache_v1 import relative


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'PATH_OUTPUT_GROUP_NATIVE_V1.json';assert not out.exists()
    receipt=json.loads((p/'PATH_OUTPUT_SCHMIDT_V1.json').read_text());ap=p/'PATH_OUTPUT_SCHMIDT_V1.pt'
    assert receipt['pred_a'] and digest(ap)==receipt['artifact_sha256']
    groups=[g for g in torch.load(ap,weights_only=True,map_location='cpu')['groups'] if g['metric']=='centered']
    binding=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    upfile=p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt';assert digest(upfile)==json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text())['cache_sha256']
    up=torch.load(upfile,weights_only=True,map_location='cpu')['ports']
    cache=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']
    rows=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].float();o=state['transformer.h.17.attn.c_proj.weight'].float()
    pre=cache['pre'];h=pre+cache['native_output']
    source_error=float((pre-up['residual']-F.linear(up['head_values'],o)).norm()/pre.norm())
    denominator=pre.double().square().mean(-1)+torch.finfo(torch.float32).eps
    targets=torch.tensor([row[side+'_answer_id'] for row in rows for side in ('base','donor')])
    def ce(x):
        values=[]
        for i in range(0,len(x),32):
            logits=30*torch.tanh(F.linear(F.rms_norm(x[i:i+32],(1152,)),u)/30)
            values.append(F.cross_entropy(logits,targets[i:i+32],reduction='none').double())
        return torch.cat(values)
    base_ce=ce(h);reports=[];errors=[]
    for name,k in [('centered8',8),('full96',96)]:
        writes=[];zero=[];swap=[]
        for g in groups:
            a=g['program'];native=run(a,up['residual'].double(),up['head_values'].double(),denominator)
            mixed=native['amplitudes']@g['edge_to_group'][:,:k]
            w=(mixed@g['group_writer'][:,:k].T)/denominator[:,None]
            alt=dict(a);alt['physical_writer']=g['group_writer'][:,:k]@g['edge_to_group'][:,:k].T
            replay=run(alt,up['residual'].double(),up['head_values'].double(),denominator)['write']
            errors.append(relative(w,replay))
            if k==96:errors.append(relative(w,native['write']))
            writes.append(w);zero.append(ce(h-w.float())-base_ce)
            swapped=h[::2]+(w[1::2]-w[::2]).float();values=[]
            for i,row in enumerate(rows):
                readers=u[[row['donor_answer_id'],row['donor_foil_id']]]
                z=30*torch.tanh(F.linear(F.rms_norm(torch.stack([h[2*i],swapped[i]]),(1152,)),readers)/30)
                values.append(float((z[1,0]-z[1,1])-(z[0,0]-z[0,1])))
            swap.append(torch.tensor(values,dtype=torch.float64))
        families=[]
        for family in sorted({r['family'] for r in rows}):
            ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);endpoint=(2*ids[:,None]+torch.tensor([0,1])).flatten()
            a,b=swap[0][ids],swap[1][ids];live=torch.maximum(a.abs(),b.abs())>=1e-4
            c,d=zero[0][endpoint],zero[1][endpoint];zlive=torch.maximum(c.abs(),d.abs())>=.001
            families.append(dict(family=family,swap_relative_rms=relative(a,b),swap_live=int(live.sum()),
                swap_sign_agreement=float((a[live].sign()==b[live].sign()).double().mean()) if live.any() else None,
                swap_meanabs=[float(a.abs().mean()),float(b.abs().mean())],zero_ce_meanabs_disagreement=float((c-d).abs().mean()),zero_live=int(zlive.sum()),
                zero_sign_agreement=float((c[zlive].sign()==d[zlive].sign()).double().mean()) if zlive.any() else None,
                zero_ce_meanabs=[float(c.abs().mean()),float(d.abs().mean())]))
        physical=relative(*writes)
        reports.append(dict(name=name,physical_write_relative_rms=physical,pred_b=physical<=.25,
            pred_c=all(c['swap_relative_rms']<=.25 and c['swap_live']>=4 and c['swap_sign_agreement']>=.9 for c in families),
            pred_d=all(c['zero_ce_meanabs_disagreement']<=.02 and c['zero_live']>=4 and c['zero_sign_agreement']>=.9 for c in families),
            families=families,zero_ce_effects=[z.tolist() for z in zero],swap_margin_effects=[s.tolist() for s in swap]))
    result=dict(pred_a=source_error<=1e-5 and max(errors)<=1e-9,pred_b=reports[0]['pred_b'],pred_c=reports[0]['pred_c'],pred_d=reports[0]['pred_d'],
        source_error=source_error,group_identity_error=max(errors),reports=reports,source_sha256=digest(ap),script_sha256=digest(__file__),
        scope='Frozen output-group replication on existing developmental morphology cache. Full96 descriptive control. No text fitting, new rank selection, semantic or OOD claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('zero_ce_effects','swap_margin_effects')} for r in reports],indent=2))
    assert result['pred_a']


if __name__=='__main__':main()
