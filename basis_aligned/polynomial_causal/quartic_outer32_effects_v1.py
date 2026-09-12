"""Complete the missing native intervention screen for the fixed outer32 extraction check.
A: old native-write error<=1e-8 replay and initial effect reports exactly match.
B: every family swap relative error<=.1, sign>=.9, live>=4.
C: every family removal CE mean absolute disagreement<=.02.
CPU only, no model body forwards, data fitting, or rank selection.
"""
from pathlib import Path
import json,time,torch
from sparse_path_stability_atlas_v1 import digest
from coupled_quartic_writer_v1 import features
from quartic_frozen_native_score_v1 import score
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);start=time.perf_counter()
 binding=json.loads((P/'QUARTIC_BANK_PRODUCER_SPAN_V1_BINDING.json').read_text())['files']
 assert all(digest(k)==v for k,v in binding.items())
 out=P/'QUARTIC_OUTER32_NATIVE_EFFECTS_V1.json';ap=P/'QUARTIC_OUTER32_V1_PROGRAM.pt'
 assert not out.exists() and not ap.exists()
 p=torch.load(P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt',weights_only=True)
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 u=state['lm_head.weight'].double();writer=p['output_writers']
 uw=u@writer;readers=u.T@uw-len(u)*u.mean(0)[:,None]*uw.mean(0)[None,:]
 l,r,d=[state['transformer.h.17.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')]
 matrices=[];modes=[];weights=[]
 for m in range(2):
  raw=l.T@((d.T@readers[:,m])[:,None]*r);matrix=(raw+raw.T)/2
  eig,vec=torch.linalg.eigh(matrix);ids=eig.abs().topk(32).indices
  modes.append(vec[:,ids]);weights.append(eig[ids]);matrices.append(matrix)
 l0,r0,d0=[state['transformer.h.16.mlp.'+name+'.weight'].double() for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0])
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
 ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']
 den=ports['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
 producer=((x@l0.T)*(x@r0.T))@d0.T*scale
 scalar=torch.stack([((producer@a).square()*b).sum(-1) for a,b in zip(modes,weights)],1)
 write=scalar@writer.T/den[:,None]
 ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].double()
 error=float((write-ref).norm()/ref.norm())
 prior=next(r for r in json.loads((P/'FULLSOURCE_QUARTIC_SPECTRAL_V1_RESULT.json').read_text())['reports'] if r['seed']==p['seed'])
 initial=features(p['input_readers'],p['inner_weights'],x)@p['mixing']@writer.T/den[:,None]
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
 result=score([ref,initial,write],ports['pre']+ports['native_output'],rows,u.float())
 prior_effects=json.loads((P/'QUARTIC_FULL_QUADRATIC_BANK_NATIVE_EFFECTS_V1.json').read_text())['reports'][0]
 regression=result['reports'][0]==prior_effects
 old_scalar=torch.stack([((producer@a[:,:16]).square()*b[:16]).sum(-1) for a,b in zip(modes,weights)],1)
 old_write=old_scalar@writer.T/den[:,None]
 replay=abs(float((old_write-ref).norm()/ref.norm())-prior['outer16_relative_error'])
 torch.save(dict(seed=p['seed'],output_readers=torch.stack(modes),outer_weights=torch.stack(weights),output_writers=writer,native_write=write,producer_scale=scale),ap)
 result['pred_a']=result['pred_a'] and regression and replay<=1e-8
 result.update(dict(old_write_error_replay=replay,initial_effect_regression=regression,artifact_sha256=digest(ap),
  fitted_floats=sum(a.numel()+b.numel() for a,b in zip(modes,weights))+writer.numel(),native_parent_floats=l0.numel()+r0.numel()+d0.numel(),
  wall_seconds=time.perf_counter()-start,scope='Fixed outer32 per output, with previous outer16 replay, exact bias-free MLP16 producer, two-output group. Reused developmental native background. No OOD, selective circuit or new weight fit.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
if __name__=='__main__':main()
