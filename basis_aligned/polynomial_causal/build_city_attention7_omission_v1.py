"""Preflight all/no/leave-one attention7 configurations; no causal selection."""
import hashlib,json,os
from pathlib import Path
import torch
from city_attention7_omission_v1 import execute
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2)
    program={'attention7':torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True),'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
    fixtures=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    contexts=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
    expected=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_WRITES.pt',weights_only=True)['writes']
    sets=[list(range(9)),[]]+[[h for h in range(9) if h!=i] for i in range(9)]
    writes=[];errors=[];outside=[]
    for f,c,ref in zip(fixtures,contexts,expected,strict=True):
        values=[execute(program,f['residual6'],f['token_ids'],f['city'],c['inputs']['destination'],heads=heads) for heads in sets]
        errors.append(float((values[0]-ref).norm()/ref.norm()));outside.extend(float(w[:,~c['inputs']['destination']].abs().max()) for w in values);writes.append(torch.stack(values))
    total=sum(v.numel() for pack in program.values() for v in pack.values() if v.is_floating_point())
    result={'pred_a':max(errors)<=1e-4,'pred_b':max(outside)==0 and all(bool(torch.isfinite(w).all()) for w in writes),'full_replay_error':max(errors),'head_sets':sets,'potential_compact_float_counts':[total-(9-len(h))*6*128*1152 for h in sets],'scope':'CPU local subset executor; parameter count assumes packing only retained head maps. No causal or fresh claim; no subset selected.', 'source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_ATTENTION7_OMISSION_V1_PREREGISTRATION.md','city_attention7_subset_v1.py','city_attention7_omission_v1.py','build_city_attention7_omission_v1.py']}}
    torch.save({'writes':writes,'head_sets':sets},P/'CITY_ATTENTION7_OMISSION_V1_WRITES.pt');(P/'CITY_ATTENTION7_OMISSION_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
