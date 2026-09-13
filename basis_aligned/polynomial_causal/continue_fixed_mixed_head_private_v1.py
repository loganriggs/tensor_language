"""CPU convergence extension; original multi-start failure remains unchanged."""
from pathlib import Path
import time,json,torch
from head17_source_interface_v1 import CHECKPOINT
from head_private_matrix_v1 import reconstruct,cycle
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter()
 assert not (P/'FIXED_MIXED_HEAD_PRIVATE_CONTINUE_V1_RESULT.json').exists()
 old=json.loads((P/'FIXED_MIXED_HEAD_PRIVATE_V1_RESULT.json').read_text());f=torch.load(P/'FIXED_MIXED_HEAD_PRIVATE_V1_PROGRAM.pt',weights_only=True)
 private=(f['private_left']@f['private_right']).permute(1,0,2).reshape(1152,1152)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu');program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
 w=program['direction'].double()*float(sd['transformer.h.10.lambdas'][0]);l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']];o=sd['transformer.h.10.attn.c_proj.weight'].double()
 x=((d*(r@w)[None,:])@l+(d*(l@w)[None,:])@r)@o;norm=x.square().sum();previous=float((x-reconstruct(f)).square().sum()/norm);history=[];stable=0;reason='step_limit';monotone=True
 for step in range(800):
  if time.perf_counter()-tic>240:reason='time_limit';break
  shared,private,f=cycle(x,private,9,128,16);loss=float((x-shared-private).square().sum()/norm);gain=(previous-loss)/previous;monotone &= loss<=previous+1e-10;history.append(dict(loss=loss,relative_gain=gain));previous=loss;stable=stable+1 if 0<=gain<=1e-7 else 0
  if step%100==0:print(json.dumps(dict(step=step,loss=loss,gain=gain,seconds=time.perf_counter()-tic)),flush=True)
  if stable>=3:reason='objective_plateau';break
 torch.save(f,P/'FIXED_MIXED_HEAD_PRIVATE_CONTINUE_V1_PROGRAM.pt')
 replay=float((x-reconstruct(f)).square().sum()/norm)
 result={'pred_a':monotone and abs(replay-previous)<=1e-10,'pred_b':reason=='objective_plateau','pred_c':previous<=.95*old['baseline_rank208_squared_error'],
  'initial_squared_error':old['best_squared_error'],'final_squared_error':previous,'baseline_rank208_squared_error':old['baseline_rank208_squared_error'],'stop':reason,'history':history,'seconds':time.perf_counter()-tic,
  'scope':'Best original seed continued on CPU with same objective and exact conditional updates; objective plateau is not global optimality or all-start convergence. No native evaluation.'}
 (P/'FIXED_MIXED_HEAD_PRIVATE_CONTINUE_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))
if __name__=='__main__':main()
