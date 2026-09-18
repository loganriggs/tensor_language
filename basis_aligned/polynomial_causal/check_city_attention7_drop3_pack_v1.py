"""CPU-only physical packing/replay; gates in frozen preregistration."""
import os,json,hashlib,time
from pathlib import Path
import torch
from city_attention7_drop3_v1 import execute
from city_attention7_omission_v1 import execute as reference
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 torch.set_num_threads(2);start=time.perf_counter()
 out=P/'CITY_ATTENTION7_DROP3_PACK_V1_RESULT.json';assert not out.exists()
 a=torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True)
 heads=[h for h in range(9) if h!=3];idx=torch.tensor([128*h+j for h in heads for j in range(128)])
 packed={k:v.clone() for k,v in a.items()}
 for k in ['q1','k1','q2','k2','value']:packed[k]=a[k][idx].clone()
 packed['output']=a['output'][:,idx].clone();packed['head_ids']=torch.tensor(heads)
 program={'attention7':packed,'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
 old={**program,'attention7':a}
 fixtures=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 contexts=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
 errors=[];outside=[];finite=True
 for f,c in zip(fixtures,contexts,strict=True):
  args=[f['residual6'],f['token_ids'],f['city'],c['inputs']['destination']]
  got=execute(program,*args);want=reference(old,*args,heads=heads)
  errors.append(float((got-want).norm()/want.norm()));outside.append(float(got[:,~args[-1]].abs().max()));finite &= bool(torch.isfinite(got).all())
 count=sum(v.numel() for pack in program.values() for v in pack.values() if v.is_floating_point())
 owned=all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in packed.values())
 paths=['CITY_ATTENTION7_DROP3_PACK_V1_PREREGISTRATION.md','city_attention7_packed_v1.py','city_attention7_drop3_v1.py','check_city_attention7_drop3_pack_v1.py']
 r={'pred_a':len(errors)==40 and max(errors)<=1e-4,'pred_b':count==22441606 and owned,'pred_c':finite and max(outside)==0 and not torch.cuda.is_initialized(),'max_relative_error':max(errors),'max_outside':max(outside),'static_float_scalars':count,'saved_float_scalars':23326342-count,'owned_tensor_storage':owned,'fixtures':len(errors),'full_model_forwards':0,'seconds':time.perf_counter()-start,'scope':'Opened implementation replay only; no fresh causal or composition promotion. Full first-value table retained.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in paths}}
 torch.save(packed,P/'CITY_ATTENTION7_DROP3_PACK_V1_ATTENTION.pt');out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
