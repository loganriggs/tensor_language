"""Preflight the trimmed interchange boundary on frozen generated outputs."""
import hashlib,json,os,time
from pathlib import Path
import torch
from city_interchange_prefix_v1 import execute
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter()
    program={'attention7':torch.load(P/'CITY_FULL_INTERCHANGE_V1_ATTENTION7.pt',weights_only=True),'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
    original=torch.load(P/'CITY_FULL_INTERCHANGE_V1_ARTIFACT.pt',weights_only=True)['fixtures'];fixtures=[];errors=[];outside=[];prices=[]
    for f in original:
        x=dict(f['inputs']);city=x['city'];stop=int(torch.where(x['destination'])[0].max())+1
        for name in ['recipient_residual6','recipient_tokens']:x[name]=x[name][:,:stop].clone()
        for name in ['donor_residual6','donor_tokens']:x[name]=x[name][:,:city+1].clone()
        got=execute(program,**x);ref=f['candidate_delta'];errors.append(float((got-ref).norm()/ref.norm()))
        outside.append(float(got[:,~x['destination']].abs().max()));prices.append(x['recipient_residual6'].numel()+x['donor_residual6'].numel())
        fixtures.append({'inputs':x,'delta':got,'expected':ref})
    bad=dict(fixtures[0]['inputs']);bad['donor_tokens']=bad['donor_tokens'].clone();bad['donor_tokens'][0,0]=-1;unknown=False
    try:execute(program,**bad)
    except ValueError:unknown=True
    r={'pred_a':max(errors)<=1e-4,'pred_b':max(outside)==0 and unknown and all(bool(torch.isfinite(f['delta']).all()) for f in fixtures),
       'max_relative_error':max(errors),'max_outside':max(outside),'unknown_token_rejected':unknown,'fixtures':len(fixtures),'native_state_scalar_range':[min(prices),max(prices)],'logical_native_contexts':2,'seconds':time.perf_counter()-start,'scope':'Opened prefix implementation preflight; weights unchanged, two native contexts and suffix external.',
       'source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_INTERCHANGE_PREFIX_V1_PREREGISTRATION.md','city_interchange_prefix_v1.py','city_residual6_fields_v1.py','check_city_interchange_prefix_cpu_v1.py']}}
    torch.save({'fixtures':fixtures},P/'CITY_INTERCHANGE_PREFIX_V1_CPU_ARTIFACT.pt');(P/'CITY_INTERCHANGE_PREFIX_V1_CPU_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
