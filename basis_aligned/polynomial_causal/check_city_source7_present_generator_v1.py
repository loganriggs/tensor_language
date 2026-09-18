"""Opened reference-group and complete-generator replay before fresh capture."""
import json,os
from pathlib import Path
import torch
import torch.nn.functional as F
from city_source7_present_generator_v1 import execute,from_projected
from city_mlp7_readers_v1 import execute as readers
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2)
    program={'attention7':torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True),'readers':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt',weights_only=True),'head8':torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt',weights_only=True)}
    prefix=torch.load(P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt',weights_only=True)['fixtures'];source=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)['fixtures'];old=torch.load(P/'CITY_SOURCE7_EDIT_V1_WRITES.pt',weights_only=True)['writes'];contexts=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures'];full=torch.load(P/'CITY_RESIDUAL6_SINGLE_INPUT_V1_WRITES.pt',weights_only=True)['writes']
    errors=[];closure=[]
    for f,s,w,c,ref in zip(prefix,source,old,contexts,full,strict=True):
        city=f['city'];mask=c['inputs']['destination'];rho=(s['mixed8_city'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
        current=s['current8'];projected=F.linear(current.double(),program['head8']['sources'].double())
        contribution=torch.zeros_like(projected);contribution[:,city]=F.linear(s['sources'][2].double()/rho,program['head8']['sources'].double())
        native_full,native_group,_=from_projected(projected,contribution,f['first_values'][:,city,256:384],program['head8'],city,mask)
        generated,group,_=execute(program,f['residual6'],f['token_ids'],city,mask,return_debug=True)
        errors.append(float((native_group-w[1]).norm()/w[1].norm()));closure.append(float((generated-ref).norm()/ref.norm()))
    r={'pred_a':max(errors)<=1e-4,'pred_b':max(closure)<=1e-4,'native_group_replay_error':max(errors),'full_generator_replay_error':max(closure),'sequences':40,'scope':'Opened CPU source-group semantics check; no fresh result or group approximation gate.'}
    (P/'CITY_SOURCE7_PRESENT_GENERATOR_V1_CPU_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
