#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;3nativequarticmatrixactions,2four-slotcontractions.
"""pred_a identity/scalar/packed replay<=1e-8;
pred_b mean GPU action time below previous3.70sec CPU actions;
pred_c peak allocation<28GiB. Price only, no eigenfit or circuit claim.
"""
import os,sys,json,time,math,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quartic_weighted_trace_v1 import producer_core,native_action
from quartic_matrixfree_eigen_v2 import SymmetricCoordinates
from composed_quartic_contraction_v1 import contract
STEM='QUARTIC_EIGEN_OPERATOR_PRICE_V1'
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 control=json.loads((P/'QUARTIC_MATRIXFREE_EIGEN_V2_CONTROL.json').read_text());assert control['pred_a'] and control['pred_b'] and control['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,matrix_actions=3,four_slot_contractions=2,packed_dimension=664128)));return
 out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(300)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 target=torch.load(P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_PROGRAMS.pt',weights_only=True)['programs'][0]
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);writer=target['output_writers'][:,0].cuda()
 reader=u.T@(u@writer)-len(u)*mean*torch.dot(mean,writer);del u
 l,r,down,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 coefficient=reader@d1;scale=float(state['transformer.h.17.lambdas'][0]);h=producer_core(down,l1,r1,coefficient,scale)
 times=[];torch.cuda.synchronize();tic=time.perf_counter()
 identity=native_action(l,r,h,torch.eye(1152,device='cuda')/math.sqrt(1152));torch.cuda.synchronize();times.append(time.perf_counter()-tic)
 reference=target['native_target_traces'][0].cuda()/math.sqrt(1152);identity_error=float((identity-reference).norm()/reference.norm())
 torch.manual_seed(91891);basis=torch.linalg.qr(torch.randn(1152,3),mode='reduced')[0].cuda();a,b,z=basis.T
 q=(torch.outer(a,a)-torch.outer(b,b))/math.sqrt(2)
 torch.cuda.synchronize();tic=time.perf_counter();answer=native_action(l,r,h,q);torch.cuda.synchronize();times.append(time.perf_counter()-tic)
 slots=torch.stack([torch.stack([z,z,a,a]),torch.stack([z,z,b,b])]);values=contract(slots,l,r,down,l1,r1,coefficient[None],scale)[:,0]
 expected=float((values[0]-values[1])/math.sqrt(2));actual=float(z@answer@z)
 scalar_error=abs(expected-actual)/max(abs(expected),abs(actual),1e-30)
 coordinates=SymmetricCoordinates(1152,'cuda');vector=coordinates.pack(q).cpu().numpy()
 torch.cuda.synchronize();tic=time.perf_counter()
 restored=coordinates.unpack(torch.as_tensor(vector,dtype=torch.float64,device='cuda'))
 packed=coordinates.pack(native_action(l,r,h,restored)).cpu().numpy();torch.cuda.synchronize();roundtrip_seconds=time.perf_counter()-tic
 packed_error=float((torch.from_numpy(packed).cuda()-coordinates.pack(answer)).norm()/coordinates.pack(answer).norm())
 cpu=json.loads((P/'QUARTIC_WEIGHTED_TRACE_NATIVE_V1_PRICE.json').read_text())['operator_seconds']
 result={'pred_a':max(identity_error,scalar_error,packed_error)<=1e-8,'pred_b':sum(times)<sum(cpu),'pred_c':torch.cuda.max_memory_allocated()<28*(1<<30)}
 result.update(dict(identity_error=identity_error,scalar_contraction_error=scalar_error,packed_replay_error=packed_error,
   gpu_action_seconds=times,cpu_reference_seconds=cpu,packed_roundtrip_seconds=roundtrip_seconds,
   core_bytes=h.numel()*h.element_size(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-start,
   scope='First fixed output native quartic operator GPU price including packed roundtrip; no eigensolver convergence, residual candidate, fit or behavioral claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
