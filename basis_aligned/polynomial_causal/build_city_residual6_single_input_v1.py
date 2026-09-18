"""Freeze the five-reader fold and preflight the one-input city generator."""
import hashlib,json,os
from pathlib import Path
import torch
from city_residual6_single_input_v1 import execute
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    files=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in files if k.endswith('pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
    head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
    W=torch.cat([head[k] for k in ['q1','k1','q2','k2','current_value']])
    reader=torch.load(P/'CITY_MLP7_READERS_V1_PROGRAM.pt',weights_only=True)
    reader['folded_down']=(W.double()@weights['transformer.h.7.mlp.Down.weight'].double()).float()
    reader['folded_bias']=(W.double()@weights['transformer.h.7.mlp.Down_bias'].double()).float()
    head8={'sources':W,'output':head['output'],'mixture':head['mixture']}
    torch.save(reader,P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt');torch.save(head8,P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt')
    program={'attention7':torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True),'readers':reader,'head8':head8}
    prefix=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures'];reference=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
    records=[];writes=[];outside=[]
    for i,(f,r) in enumerate(zip(prefix,reference,strict=True)):
        delta,queries=execute(program,f['residual6'],f['token_ids'],f['city'],r['inputs']['destination'],return_queries=True)
        q=r['inputs']['rotated_queries'].double();native=r['expected_native_delta']
        records.append({'sequence':i,'query_error':float((queries-q).norm()/q.norm()),'write_error':float((delta-native).norm()/native.norm())})
        writes.append(delta);outside.append(float(delta[:,~r['inputs']['destination']].abs().max()))
    bad=prefix[0]['token_ids'].clone();bad[0,0]=-1;unknown=False
    try:execute(program,prefix[0]['residual6'],bad,prefix[0]['city'],reference[0]['inputs']['destination'])
    except ValueError:unknown=True
    result={'query_gate':max(r['query_error'] for r in records)<=1e-4,'write_gate':max(r['write_error'] for r in records)<=.10,
            'support_finite_unknown_gate':max(outside)==0 and all(bool(torch.isfinite(w).all()) for w in writes) and unknown,
            'max_query_error':max(r['query_error'] for r in records),'max_write_error':max(r['write_error'] for r in records),
            'floating_scalars':sum(v.numel() for pack in program.values() for v in pack.values() if v.is_floating_point()),
            'native_input_arrays':1,'native_input_scalars_T32':36864,'sequences':records,
            'scope':'Opened CPU preflight; one residual6 input, no supplied query or RMS. Approximate mixed8 normalization. Native MLP8/suffix external.',
            'source_shas':{str(P/n):hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_RESIDUAL6_SINGLE_INPUT_V1_PREREGISTRATION.md','city_attention7_full_v1.py','city_residual6_single_input_v1.py','build_city_residual6_single_input_v1.py']}}
    torch.save({'writes':writes},P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_WRITES.pt')
    (P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sequences','source_shas']},indent=2))
if __name__=='__main__':main()
