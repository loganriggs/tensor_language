#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_excess_alignment
"""Architecture-preserving source-channel permutation control.
pred_a native source replay and declared coefficient-Gram invariants<1e-10relative;
       GPU native overlap agrees with saved CPU statistic<1e-10absolute.
pred_b native overlap exceeds every one of16 null draws for each metric/null family.
Null: shared trained L/R and input geometry explain observed support alignment.
No model forward or fitted replacement. Pair support352 and edge64 fixed upstream.
Native z/h/circuit fidelity untouched. All contrasts descriptive, not causal adoption.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 plan=json.loads((P/'PAIR_SUPPORT_ARCHITECTURE_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 sys.path.insert(0,str(P));from pair_support_architecture import forms,permute,overlap
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 assert not (P/'PAIR_SUPPORT_ARCHITECTURE_V1.json').exists()
 raw=torch.load(P/'PAIR_SUPPORT_ARCHITECTURE_INPUTS_V1.pt',weights_only=True);data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
 L,R,C,S=[raw[k].cuda() for k in ('L','R','C','root')];Q=forms(L,R,C);truth=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]).cuda();replay=((Q-truth).flatten(1).norm(dim=1)/truth.flatten(1).norm(dim=1)).tolist();assert max(replay)<1e-10
 cpu={r['geometry']:r['native_statistic'] for r in json.loads((P/'PAIR_SUPPORT_NULL_V1.json').read_text())['records']};native={};rows=[];checks=[]
 for geometry in plan['metrics']:
  native[geometry]=overlap(Q,S if geometry=='calibration_shaped' else None,plan['pair_width'],plan['edge_width']);assert abs(native[geometry]['statistic']-cpu[geometry])<1e-10
 print('NATIVE',json.dumps(native),flush=True)
 for kind in plan['nulls']:
  for seed in plan['seeds']:
   coefficients=permute(C,kind,51000+seed);errors=[]
   for j in range(3):
    original=C[2*j:2*j+2]@C[2*j:2*j+2].T;new=coefficients[2*j:2*j+2]@coefficients[2*j:2*j+2].T;errors.append(float((new-original).norm()/original.norm()))
   full=float((coefficients@coefficients.T-C@C.T).norm()/(C@C.T).norm());assert max(errors)<1e-10
   if kind=='common':assert full<1e-10
   checks.append(dict(kind=kind,seed=seed,pair_gram_replay=errors,full_gram_difference=full));target=forms(L,R,coefficients)
   for geometry in plan['metrics']:
    values=overlap(target,S if geometry=='calibration_shaped' else None,plan['pair_width'],plan['edge_width']);row=dict(kind=kind,seed=seed,geometry=geometry,**values);rows.append(row);print(json.dumps(row),flush=True)
 summaries=[]
 for kind in plan['nulls']:
  for geometry in plan['metrics']:
   values=[r['statistic'] for r in rows if r['kind']==kind and r['geometry']==geometry];v=torch.tensor(values,dtype=torch.float64);actual=native[geometry]['statistic'];summaries.append(dict(kind=kind,geometry=geometry,native=actual,null_mean=float(v.mean()),null_std=float(v.std()),null_min=float(v.min()),null_max=float(v.max()),native_minus_null_mean=actual-float(v.mean()),exceeds_all=actual>float(v.max()),monte_carlo_upper_tail=(1+sum(x>=actual for x in values))/(1+len(values))))
 pred=dict(pred_a_instrument=True,pred_b_excess_alignment=all(s['exceeds_all'] for s in summaries))
 (P/'PAIR_SUPPORT_ARCHITECTURE_V1.json').write_text(json.dumps(dict(plan=plan,native=native,records=rows,invariant_checks=checks,native_form_replay=replay,summaries=summaries,predictions=pred,seconds=time.monotonic()-start),indent=2)+'\n');print(json.dumps(summaries),flush=True);print(pred,flush=True)
if __name__=='__main__':main()
