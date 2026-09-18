"""Opened local normalizer approximation screen; no behavioral claim."""
import hashlib,json
from pathlib import Path
import torch
from city_mlp7_generated_norm_v1 import execute
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
    program=torch.load(P/'CITY_MLP7_READERS_V1_PROGRAM.pt',weights_only=True)
    head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
    fixtures=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
    records=[];saved=[]
    for i,f in enumerate(fixtures):
        context={k:v for k,v in f['inputs'].items() if k!='key_rms'}
        exact=execute(program,head,**context)
        native_rms=context.pop('input_rms')
        approx=execute(program,head,**context)
        rho=(context['other_city_sources'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
        reference=f['expected_native_delta']
        records.append({'sequence':i,'exact_mode_error':float((exact-reference).norm()/reference.norm()),
                        'approximate_mode_error':float((approx-reference).norm()/reference.norm()),'rms_ratio':float((rho/native_rms).item())})
        saved.append({'exact':exact,'approximate':approx,'inputs':context})
    result={'pred_a':max(r['exact_mode_error'] for r in records)<=1e-4,
            'pred_b':max(r['approximate_mode_error'] for r in records)<=.10,
            'pred_c':max(abs(r['rms_ratio']-1) for r in records)<=.10,
            'max_exact_mode_error':max(r['exact_mode_error'] for r in records),
            'max_approximate_mode_error':max(r['approximate_mode_error'] for r in records),
            'rms_ratio_range':[min(r['rms_ratio'] for r in records),max(r['rms_ratio'] for r in records)],'sequences':records,
            'scope':'Opened local CPU screen. Exact key RMS generation; other-source-only mixed8 RMS approximation. Native queries and upstream vectors remain external. No suffix, selectivity or fresh claim.',
            'source_shas':{str(P/name):hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['CITY_MLP7_GENERATED_NORM_V1_PREREGISTRATION.md','city_mlp7_generated_norm_v1.py','check_city_mlp7_generated_norm_v1.py']}}
    torch.save({'fixtures':saved},P/'CITY_MLP7_GENERATED_NORM_V1_CPU_ARTIFACT.pt')
    (P/'CITY_MLP7_GENERATED_NORM_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sequences','source_shas']},indent=2))
if __name__=='__main__':main()
