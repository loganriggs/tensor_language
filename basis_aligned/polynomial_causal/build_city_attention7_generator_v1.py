"""Build and check an earlier native-input boundary for the city readers."""
import hashlib,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from city_attention7_generator_v1 import execute
from city_mlp7_readers_v1 import execute as read
from city_mlp7_generated_norm_v1 import execute as city_write
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    start=time.perf_counter();binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'));state=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
    tables=torch.load(P/'CITY_FULL_PILE_V2_TABLES.pt',weights_only=True)
    program={k:tables[k] for k in ['token_ids','initial_table','first_table','lambdas8']}
    for k,n in [('q1','c_q'),('k1','c_k'),('q2','c_q2'),('k2','c_k2'),('value','c_v'),('output','c_proj')]:
        program[k]=state[f'transformer.h.7.attn.{n}.weight'].float()
    program['mixture']=state['transformer.h.7.attn.lamb'].float();program['lambdas7']=state['transformer.h.7.lambdas'].float()
    torch.save(program,P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt')
    reader=torch.load(P/'CITY_MLP7_READERS_V1_PROGRAM.pt',weights_only=True)
    head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
    W=torch.cat([head[k] for k in ['k1','k2','current_value']])
    prefix=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures']
    reference=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
    records=[];outputs=[]
    def rel(a,b):return float((a.double()-b.double()).norm()/b.double().norm())
    for i,(f,ref) in enumerate(zip(prefix,reference,strict=True)):
        city=f['city'];got=execute(program,f['residual6'],f['token_ids'],city)
        readings=read(reader,got['normalized_mlp7_city']);native=F.linear(f['mlp7'],W)[:,city]
        context={k:v for k,v in ref['inputs'].items() if k not in ['key_rms','normalized_mlp7_city','other_city_sources']}
        delta=city_write(reader,head,**context,normalized_mlp7_city=got['normalized_mlp7_city'],other_city_sources=got['other_city_sources'])
        records.append({'sequence':i,'attention7_error':rel(got['attention7_city'],f['attention7'][:,city]),'input_error':rel(got['normalized_mlp7_city'],f['normalized_mlp7'][:,city]),
                        'reader_error':rel(readings,native),'city_write_error':rel(delta,ref['expected_native_delta']),
                        'residual6_prefix_scalars':(city+1)*1152,'previous_two_city_vector_scalars':2304})
        outputs.append({'generated':got,'delta':delta})
    unknown=False
    try:execute(program,prefix[0]['residual6'][:,:1],torch.tensor([[-1]]),0)
    except ValueError:unknown=True
    result={'pred_a':max(r['attention7_error'] for r in records)<=1e-4,
            'pred_b':max(max(r['input_error'],r['reader_error']) for r in records)<=1e-4,
            'pred_c':max(r['city_write_error'] for r in records)<=1e-4 and all(bool(torch.isfinite(o['delta']).all()) for o in outputs) and not torch.cuda.is_initialized(),
            'unknown_token_rejected':unknown,'generator_floating_scalars':sum(v.numel() for v in program.values() if v.is_floating_point()),
            'reader_floating_scalars':sum(v.numel() for v in reader.values()),'sequences':records,'seconds':time.perf_counter()-start,
            'scope':'Opened CPU earlier-boundary generator. Residual6 prefix and head8 query fields remain native. Native mixed8 RMS retained here; no full-suffix, fresh or composition certification.',
            'source_shas':{str(P/n):hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_ATTENTION7_GENERATOR_V1_PREREGISTRATION.md','city_attention7_generator_v1.py','build_city_attention7_generator_v1.py']}}
    torch.save({'outputs':outputs},P/'CITY_ATTENTION7_GENERATOR_V1_ARTIFACT.pt')
    (P/'CITY_ATTENTION7_GENERATOR_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sequences','source_shas']},indent=2))
    print(json.dumps({key:max(r[key] for r in records) for key in ['attention7_error','input_error','reader_error','city_write_error']}))
if __name__=='__main__':main()
