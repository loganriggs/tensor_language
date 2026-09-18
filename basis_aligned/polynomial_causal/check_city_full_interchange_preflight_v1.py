"""Opened factored-field replay for the new coupled city interchange."""
import hashlib,json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from city_residual6_fields_v1 import execute,from_projected,city_write
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter()
    program={'attention7':torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True),'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
    prefix=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures'];sources=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)['fixtures'];contexts=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures'];expected=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_WRITES.pt',weights_only=True)['writes']
    generated=[execute(program,f['residual6'],f['token_ids']) for f in prefix]
    native=[from_projected(F.linear(s['current8'].double(),program['head8']['sources'].double()),f['first_values'],program['head8']) for s,f in zip(sources,prefix,strict=True)]
    gen_errors=[];native_errors=[];swap_errors=[];outside=[];fixtures=[]
    for i,(f,c,ref) in enumerate(zip(prefix,contexts,expected,strict=True)):
        city=f['city'];mask=c['inputs']['destination'];head=program['head8']
        own=city_write(generated[i],generated[i],head,city,mask);own_native=city_write(native[i],native[i],head,city,mask)
        gen_errors.append(float((-own-ref).norm()/ref.norm()));native_errors.append(float((-own_native-c['expected_native_delta']).norm()/c['expected_native_delta'].norm()))
        swap=city_write(generated[i],generated[i^1],head,city,mask)-own
        swap_native=city_write(native[i],native[i^1],head,city,mask)-own_native
        swap_errors.append(float((swap-swap_native).norm()/swap_native.norm()));outside.append(float(swap[:,~mask].abs().max()))
        fixtures.append({'generated_swap':swap,'native_swap':swap_native})
    r={'pred_a':max(gen_errors)<=1e-10,'pred_b':max(native_errors)<=1e-4,'pred_c':max(outside)==0 and all(bool(torch.isfinite(f['generated_swap']).all()) for f in fixtures),'max_generator_replay_error':max(gen_errors),'max_native_removal_replay_error':max(native_errors),'local_swap_error_range':[min(swap_errors),max(swap_errors)],'sequences':40,'native_contexts_per_edit':2,'native_state_scalars_T32_pair':73728,'seconds':time.perf_counter()-start,'scope':'Opened CPU interchange factorization only; no installed behavioral/fresh or composition result.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_FULL_INTERCHANGE_V1_PREREGISTRATION.md','city_residual6_fields_v1.py','check_city_full_interchange_preflight_v1.py']}}
    torch.save({'fixtures':fixtures},P/'CITY_FULL_INTERCHANGE_V1_CPU_ARTIFACT.pt');(P/'CITY_FULL_INTERCHANGE_V1_CPU_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
