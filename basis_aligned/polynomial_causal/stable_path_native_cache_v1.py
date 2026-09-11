"""Developmental native-state replication of frozen18edge banks; no fitting.
A hashes/source pre replay<=1e-5 and finite tail. B physicalwrite symrelativeRMS<=.25.
C swaps symrelativeRMS<=.25/sign>=.9 eachfamily atfloor1e-4,>=4live.
D zeroCE meanabsdisagreement<=.02/sign>=.9 eachfamily atfloor.001,>=4live.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from sparse_path_program_v1 import run
from sparse_path_stability_atlas_v1 import digest


def relative(a,b):return float((a-b).norm()/((a.square().sum()+b.square().sum())/2).sqrt().clamp_min(1e-30))


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'STABLE_PATH_NATIVE_CACHE_V1.json';assert not out.exists()
    receipt=json.loads((p/'STABLE_PATH_BANK_V1.json').read_text());ap=p/'STABLE_PATH_BANK_V1.pt';assert digest(ap)==receipt['artifact_sha256']
    programs=torch.load(ap,weights_only=True,map_location='cpu')['programs']
    binding=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    upstream=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text());assert digest(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt')==upstream['cache_sha256']
    up=torch.load(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt',weights_only=True,map_location='cpu')['ports']
    cache=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports'];rows=json.loads((p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].float();o=state['transformer.h.17.attn.c_proj.weight'].float()
    pre=cache['pre'];h=pre+cache['native_output'];reconstructed=up['residual']+F.linear(up['head_values'],o)
    error=float((pre-reconstructed).norm()/pre.norm());assert error<=1e-5
    denominator=pre.double().square().mean(-1)+torch.finfo(torch.float32).eps
    writes=[run(a,up['residual'].double(),up['head_values'].double(),denominator)['write'] for a in programs]
    physical=relative(*writes)
    targets=torch.tensor([row[side+'_answer_id'] for row in rows for side in ('base','donor')]);zero=[];swap=[]
    def scores(x):return 30*torch.tanh(F.linear(F.rms_norm(x,(1152,)),u)/30)
    def ce(x):
        values=[]
        for i in range(0,len(x),32):values.append(F.cross_entropy(scores(x[i:i+32]),targets[i:i+32],reduction='none').double())
        return torch.cat(values)
    base_ce=ce(h)
    for w in writes:
        zero.append(ce(h-w.float())-base_ce)
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
        families.append(dict(family=family,pairs=len(ids),swap_relative_rms=relative(a,b),swap_live=int(live.sum()),
            swap_sign_agreement=float((a[live].sign()==b[live].sign()).double().mean()) if live.any() else None,
            swap_meanabs=[float(a.abs().mean()),float(b.abs().mean())],zero_ce_meanabs_disagreement=float((c-d).abs().mean()),zero_live=int(zlive.sum()),
            zero_sign_agreement=float((c[zlive].sign()==d[zlive].sign()).double().mean()) if zlive.any() else None,
            zero_ce_meanabs=[float(c.abs().mean()),float(d.abs().mean())]))
    result=dict(pred_a=error<=1e-5 and all(torch.isfinite(x).all().item() for x in writes+zero+swap),pred_b=physical<=.25,
        pred_c=all(c['swap_relative_rms']<=.25 and c['swap_live']>=4 and c['swap_sign_agreement']>=.9 for c in families),
        pred_d=all(c['zero_ce_meanabs_disagreement']<=.02 and c['zero_live']>=4 and c['zero_sign_agreement']>=.9 for c in families),
        source_pre_relative_error=error,physical_write_relative_rms=physical,families=families,zero_ce_effects=[x.tolist() for x in zero],swap_margin_effects=[x.tolist() for x in swap],
        source_bank_sha256=digest(ap),scope='Developmental existing morphology/neighbor cache, frozen weight-only18edge replicas. Native source/background retained; replication is not intended-task selectivity, absolute full-MLP extraction, fresh text or OOD validation.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('zero_ce_effects','swap_margin_effects')},indent=2));assert result['pred_a']

if __name__=='__main__':main()
