#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_trace pred_b_stability pred_c_rank
"""Native coefficient input-mode probe.
pred_a_trace finite & trace near1; pred_b_stability panels within.02;
pred_c_rank pooled32tail>.90. Two4096panels, no textforwards.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(panels=2,probes_per_panel=4096,projection_probes=1024,batch=128,ranks=[32,64,128,256,512],native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from quartic_input_probes import input_probes
 from quartic_cp import directional
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_INPUT_MODE_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher[0]/=scale;d=1152;grams=[];records=[];bases={}
  def signs(n,width,gen):return torch.randint(2,(n,width),device='cuda',generator=gen).float()*2-1
  def summarize(G,name):
   ev,V=torch.linalg.eigh(G);ev=ev.clamp_min(0);total=ev.sum();row=dict(panel=name,normalized_trace=float(total),finite_probe_rank_tail_errors={str(r):float((ev[:-r].sum()/total).sqrt()) for r in PLAN['ranks']});records.append(row);bases[name]=V[:,-128:].float().cpu();print(json.dumps(row),flush=True);return ev,V
  for panel in range(2):
   gen=torch.Generator(device='cuda');gen.manual_seed(1807+panel);values=[]
   for offset in range(0,4096,128):
    b,c,e=[signs(128,d,gen) for _ in range(3)];g=signs(128,1152,gen);values.append(input_probes(teacher,b,c,e,g))
   t=torch.cat(values).double();G=t.T@t/len(t);grams.append(G);summarize(G,str(panel))
  ev,V=summarize((grams[0]+grams[1])/2,'pooled');projections={f'input_mode_{r}':V[:,-r:].float() for r in [32,64,128]};cp=torch.load(P/'NATIVE_EXACT_CP_GREEDY_V1.pt',weights_only=True)['students'][8];projections['cp8_span']=torch.linalg.qr(torch.cat(cp['factors']).T.cuda())[0]
  learned=P/'NATIVE_LEARNED_SHARED_BANK_V1.pt'
  if learned.exists():
   result=json.load(open(P/'NATIVE_LEARNED_SHARED_BANK_V1.json'));best=max(result['records'],key=lambda r:r['explained_fraction_using_estimated_teacher_norm']);student=torch.load(learned,weights_only=True)['students'][(best['optimizer'],best['lr'],best['seed'])];vectors=torch.cat([student['U'].flatten(0,1),student['V'].flatten(0,1)]);projections['learned_bank_span']=torch.linalg.qr(vectors.T.cuda())[0]
  energy=[]
  for name,basis in projections.items():
   gen=torch.Generator(device='cuda');gen.manual_seed(1810);samples=[]
   for offset in range(0,1024,128):
    vectors=[signs(128,basis.shape[1],gen)@basis.T for _ in range(4)];y=directional(*teacher,vectors);samples.append(y.double().square().sum(1))
   values=torch.cat(samples);row=dict(subspace=name,dimension=basis.shape[1],normalized_projected_energy=float(values.mean()),standard_error=float(values.std()/len(values)**.5),estimated_projection_error=math.sqrt(max(0,1-float(values.mean()/ev.sum()))));energy.append(row);print(json.dumps(row),flush=True)
  gap=abs(records[0]['finite_probe_rank_tail_errors']['32']-records[1]['finite_probe_rank_tail_errors']['32']);pred=dict(pred_a_trace=all(math.isfinite(r['normalized_trace']) and abs(r['normalized_trace']-1)<.15 for r in records),pred_b_stability=gap<.02,pred_c_rank=records[-1]['finite_probe_rank_tail_errors']['32']>.9);guard_torch_save(dict(bases=bases,pooled_eigenvalues=ev.cpu(),projection_bases={k:v.cpu() for k,v in projections.items()}),str(P/'NATIVE_INPUT_MODE_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,panels=records,projections=energy,rank32_panel_gap=gap,seconds=time.perf_counter()-start,predictions=pred,scope='Unbiased first-input Gram estimate and finite-measurement rank tails. NOT certified full-tensor rank/error bounds. Particular projected-subspace energies estimated with independent input-slot probes; no Gaussianfunction/circuitclaim.'),indent=2)+'\n')
if __name__=='__main__':main()
