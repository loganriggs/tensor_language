#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_identity pred_b_precision pred_c_coverage
"""Projected weighted norm: pred_a_identity pred_b_precision pred_c_coverage."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(metrics=3,projection_rank=4,probes=4096,batch=128,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P))
 from quartic_cp import directional
 from projected_quartic_energy import projected_energy
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_WEIGHTED_NORM_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True);previous=json.load(open(P/'NATIVE_WEIGHTED_BANK_V1.json'));state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'],w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];rows=[]
  for name,transform in source['metric_transforms'].items():
   if transform is None:continue
   L=transform.cuda();ev,V=torch.linalg.eigh((L.double()+L.double().T)/2);basis=V[:,-4:].float();orth=float((basis.double().T@basis.double()-torch.eye(4,device='cuda')).norm());t=[*teacher[:-2],teacher[-2]@L,teacher[-1]@L];known=float(projected_energy(t,basis));gen=torch.Generator(device='cuda');gen.manual_seed(1858);raw=[];res=[]
   for _ in range(32):
    x=[torch.randint(2,(128,1152),device='cuda',generator=gen).float()*2-1 for _ in range(4)];y=directional(*t,x).double();z=directional(*t,[v@basis@basis.T for v in x]).double();raw.append(y.square().sum(1));res.append((y-z).square().sum(1))
   raw=torch.cat(raw);res=torch.cat(res);rawenergy=float(raw.mean());energy=known+float(res.mean());se=float(res.std()/len(res)**.5);rawse=float(raw.std()/len(raw)**.5);gain=max(r['weighted_gain'] for r in previous['records'] if r['metric']==name);row=dict(metric=name,exact_projected_energy=known,refined_energy_estimate=energy,refined_standard_error=se,raw_energy_estimate=rawenergy,raw_standard_error=rawse,old_energy_estimate=previous['norms'][name]['energy'],old_standard_error=previous['norms'][name]['standard_error'],projection_fraction_estimate=known/energy,selected_gain_fraction_estimate=gain/energy,orthogonality_error=orth,discrepancy_in_combined_se=abs(energy-rawenergy)/math.sqrt(se**2+rawse**2));rows.append(row);print(json.dumps(row),flush=True)
  second=next(r for r in rows if r['metric']=='second_floor01');predictions=dict(pred_a_identity=all(r['orthogonality_error']<1e-5 and r['discrepancy_in_combined_se']<3 for r in rows),pred_b_precision=second['refined_standard_error']<.5*second['raw_standard_error'],pred_c_coverage=second['refined_standard_error']/second['refined_energy_estimate']<.02);out.write_text(json.dumps(dict(plan=PLAN,records=rows,predictions=predictions,seconds=time.perf_counter()-start,scope='Orthogonalprojectedenergy exact inFP; residualnormestimated, withreportedSE. Sameprobe raw/refinedestimates correlated, combinedSE discrepancy is a descriptive tripwire, not calibrated confidence test. No studentchanges.'),indent=2)+'\n')
if __name__=='__main__':main()
