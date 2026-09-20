#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_precision pred_b_sampling pred_c_utilization
"""Exact span energy: pred_a_precision, pred_b_sampling, pred_c_utilization."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(spaces=3,dimension=32,unique_entries=52360,batch=256,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P))
 from projected_quartic_energy import projected_energy,symmetric_indices,check
 from implicit_quartic import entries
 validation=check();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_PROJECTED_ENERGY_V1.json';assert not out.exists();start=time.perf_counter()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
  bases=torch.load(P/'NATIVE_INPUT_MODE_V1.pt',weights_only=True)['projection_bases'];estimates={r['subspace']:r for r in json.load(open(P/'NATIVE_INPUT_MODE_V1.json'))['projections']};cp=json.load(open(P/'NATIVE_EXACT_CP_GREEDY_V1.json'))['records'][-1]['explained_fraction_using_estimated_teacher_norm'];bank=max(r['explained_fraction_using_estimated_teacher_norm'] for r in json.load(open(P/'NATIVE_LEARNED_SHARED_BANK_V1.json'))['records']);rows=[]
  for name in ['input_mode_32','cp8_span','learned_bank_span']:
   basis=torch.linalg.qr(bases[name].cuda().double())[0].float();orth=float((basis.double().T@basis.double()-torch.eye(32,device='cuda',dtype=torch.float64)).norm());energy=float(projected_energy(teacher,basis));idx,_=symmetric_indices(32,'cuda');idx=idx[:128];C,L,R,D,A,B=teacher;low=entries(C,L,R,D,A@basis,B@basis,idx).double();high=entries(C.double(),L.double(),R.double(),D.double(),A.double()@basis.double(),B.double()@basis.double(),idx);precision=float((low-high).norm()/high.norm());estimate=estimates[name];row=dict(subspace=name,normalized_projected_energy=energy,estimated_conditional_minimum_error=math.sqrt(max(0,1-energy)),sampled_energy=estimate['normalized_projected_energy'],sampled_standard_error=estimate['standard_error'],sample_discrepancy_in_se=abs(energy-estimate['normalized_projected_energy'])/estimate['standard_error'],precision_error=precision,orthogonality_error=orth)
   if name!='input_mode_32':
    gain=cp if name=='cp8_span' else bank;row.update(student_gain=gain,available_energy_utilization=gain/energy)
   rows.append(row);print(json.dumps(row),flush=True)
  predictions=dict(pred_a_precision=all(r['precision_error']<1e-4 and r['orthogonality_error']<1e-5 for r in rows),pred_b_sampling=all(r['sample_discrepancy_in_se']<6 for r in rows),pred_c_utilization=.6<rows[-1]['available_energy_utilization']<=1.001);out.write_text(json.dumps(dict(plan=PLAN,validation=validation,records=rows,predictions=predictions,seconds=time.perf_counter()-start,scope='Exhaustive projected coefficient energy in floating point. Conditional frozen-subspace ceiling; total teacher norm estimated. Not global capacity or circuit identification.'),indent=2)+'\n')
if __name__=='__main__':main()
