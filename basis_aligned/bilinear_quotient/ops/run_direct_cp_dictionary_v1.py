#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_monotone pred_c_native
"""Exact CP dictionary baseline. pred_a_finite, pred_b_monotone, pred_c_native beats random at8.
No native forwards;1024 candidates/dictionary; widths1/2/4/8.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(candidates=1024,widths=[1,2,4,8],native_diagonal=512,native_offdiagonal=512,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from quartic_cp import cp_gram,directional,cp_entries
 from cp_dictionary import select_atoms
 from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_CP_DICTIONARY_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher[0]/=scale;d=1152;gen=torch.Generator(device='cuda');gen.manual_seed(1651);idx=torch.randint(d,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in idx.split(512)]);x=torch.randn(256,d,device='cuda',generator=gen);C,L,R,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;fy=((h@L.T)*(h@R.T))@C.T;records=[];students={};all_receipts={}
  for family in ['native_channels','random']:
   torch.manual_seed(1707)
   if family=='native_channels':
    diag=torch.randperm(A.shape[0],device='cuda')[:512];left=torch.randint(A.shape[0],(512,),device='cuda');right=(left+torch.randint(1,A.shape[0],(512,),device='cuda'))%A.shape[0];i=torch.cat([diag,left]);j=torch.cat([diag,right]);F=[A[i],B[i],A[j],B[j]];pairs=torch.stack([i,j],1).cpu()
   else:F=[torch.randn(1024,d,device='cuda') for _ in range(4)];pairs=None
   F=[a/a.norm(dim=1,keepdim=True) for a in F];K=cp_gram(F,F);Q=torch.cat([directional(*teacher,[a[start:start+128] for a in F]) for start in range(0,1024,128)]).T;receipt=select_atoms(K,Q,8);all_receipts[family]=receipt
   for row in receipt:
    if row['width'] not in [1,2,4,8]:continue
    ix=row['selected'];f=[a[ix] for a in F];block=K[ix][:,ix].double();c=torch.linalg.solve(block,Q[:,ix].double().T).T.float();prediction=torch.cat([cp_entries(c,f,z) for z in idx.split(512)]);coefficient=float((prediction-target).norm()/target.norm());prod=torch.ones(len(x),len(ix),device='cuda')
    for a in f:prod*=x@a.T
    function=float((prod@c.T-fy).norm()/fy.norm());r=dict(family=family,width=len(ix),explained_fraction_using_estimated_teacher_norm=row['gain'],estimated_coefficient_error=math.sqrt(max(0,1-row['gain'])),evaluation_coefficient_error=coefficient,gaussian_function_error=function,reduced_parameters=5*d*len(ix),expanded_parameters=(4*d+50304)*len(ix),condition=row['condition']);records.append(r);students[(family,len(ix))]=dict(C=c.cpu(),factors=[a.cpu() for a in f],source_pairs=None if pairs is None else pairs[ix]);print(json.dumps(r),flush=True)
   out.write_text(json.dumps(dict(plan=PLAN,records=records,selections=all_receipts,seconds=time.perf_counter()-start),indent=2)+'\n')
  final={k:v[-1]['gain'] for k,v in all_receipts.items()};pred=dict(pred_a_finite=all(math.isfinite(r['gain']) for rows in all_receipts.values() for r in rows),pred_b_monotone=all(all(b['gain']>=a['gain']-1e-10 for a,b in zip(rows,rows[1:])) for rows in all_receipts.values()),pred_c_native=final['native_channels']>final['random']);guard_torch_save(dict(students=students,teacher_scale=scale),str(P/'NATIVE_CP_DICTIONARY_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,selections=all_receipts,seconds=time.perf_counter()-start,predictions=pred,scope='Limited candidate dictionary, exact symmetric training contractions, independent coefficient/Gaussian diagnostics. Norm estimate only scales error labels; no normalized model or circuit claim.'),indent=2)+'\n')
if __name__=='__main__':main()
