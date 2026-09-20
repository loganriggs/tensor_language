from pathlib import Path
import json,torch
from affine_state_readout import compile,evaluate
from final_readout_field_program import pack_fields,evaluate_fields
torch.manual_seed(2401);torch.set_num_threads(2);b=torch.randn(3,3,11,dtype=torch.float64);u=torch.randn(3,9,2,11,dtype=torch.float64);program=compile(b,u);errors=[]
for s,a in [(0,0),(1,0),(0,1),(1,1),(.3,.7)]:errors.append(float((evaluate(program,s,a)-evaluate_fields(pack_fields(b[:,0]+s*b[:,1]+a*b[:,2],u))).abs().max()))
wrong=(program[0],program[1].clone());wrong[1][:,[1,2,4]]=0;live=float((evaluate(program,1,1)-evaluate(wrong,1,1)).norm());assert max(errors)<1e-12 and live>.1
p=Path(__file__).with_name('AFFINE_STATE_READOUT_CPU_V1.json');assert not p.exists();p.write_text(json.dumps(dict(max_error=max(errors),omitted_gram_cross_error=live,values_per_input=60,scope='Exact readout on affine state span, not an assertion of affine native dynamics'),indent=2)+'\n');print(max(errors),live)
