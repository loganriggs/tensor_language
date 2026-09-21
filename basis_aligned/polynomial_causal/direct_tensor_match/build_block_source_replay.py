"""Freeze exact block rewrite and native replay controls on reused balanced panel."""
from pathlib import Path
import torch,json,hashlib
from source_interface import residual_write
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);old=torch.load(p/'MIDPOINT_SHARED_SOURCE_PROGRAMS_V1.pt',weights_only=True);block=torch.load(p/'MIDPOINT_SOURCE_BLOCK_PROGRAMS_V1.pt',weights_only=True);programs={k:old[k] for k in ['covariance_16','shared16','shared24']};programs.update({'block'+k:v for k,v in block.items()});torch.manual_seed(2243);z=torch.randn(4,7,1152,dtype=torch.float64);h=torch.randn_like(z);errors={}
for k in ['16','24']:
 ref=residual_write(z,h,old['shared'+k]);pred=residual_write(z,h,block[k]);err=float((pred-ref).norm()/ref.norm());assert err<1e-10
 cast={n:v.float() if v.is_floating_point() else v for n,v in block[k].items()};fp32=residual_write(z.float(),h.float(),cast);err32=float((fp32.double()-ref).norm()/ref.norm());assert err32<1e-4;errors[k]=dict(fp64=err,fp32=err32)
f=p/'MIDPOINT_BLOCK_SOURCE_REPLAY_PROGRAMS_V1.pt';torch.save(programs,f);plan=json.loads((p/'MIDPOINT_BALANCED_SOURCE_PLAN_V1.json').read_text());plan['program_sha256']=hashlib.sha256(f.read_bytes()).hexdigest();plan['cpu_replay']=errors;plan['scope']='Exact numerical block-product rewrite of shared16/24 on reused balanced-donor panel. Native target remains exact leading-feature removal. Verify inherited approximation errors/CE to1e-6, not a new semantic/behavioral gate or fresh evaluation.';(p/'MIDPOINT_BLOCK_SOURCE_REPLAY_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(errors)
