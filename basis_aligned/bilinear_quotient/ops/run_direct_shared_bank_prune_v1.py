#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_solver pred_b_retains pred_c_cp
"""Freeze learnedbank, exhaustive8of10root sparsity andexactwriterrefit.
pred_a_solver finite/residual<1e-10; pred_b_retains>=90%; pred_c_cp beatsCP8.
Eightstudents,45supports each, no activationforwards;46,080floats+support.
"""
import os,sys,json,time,math,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(students=8,root_candidates=10,retained_roots=8,supports=45,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from shared_quadratic_bank import bank_gram,native_bank_cross,bank_entries,roots
 from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_SHARED_BANK_PRUNE_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'NATIVE_LEARNED_SHARED_BANK_V1.pt',weights_only=True);source_rows=json.load(open(P/'NATIVE_LEARNED_SHARED_BANK_V1.json'))['records'];source_index={(r['optimizer'],r['lr'],r['seed']):r for r in source_rows};state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'],w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];gen=torch.Generator(device='cuda');gen.manual_seed(1651);idx=torch.randint(1152,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in idx.split(512)]).double();x=torch.randn(256,1152,device='cuda',generator=gen);C,L,R,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;fy=(((h@L.T)*(h@R.T))@C.T).double();i,j=roots(4,x.device);pairs=torch.stack([i,j],1);records=[];students={}
  for key,s in source['students'].items():
   U=s['U'].cuda();V=s['V'].cuda();G=bank_gram(U,V).double();G=(G+G.T)/2;cross=native_bank_cross(teacher,U,V).double();best=-float('inf')
   for support in itertools.combinations(range(10),8):
    ids=list(support);block=G[ids][:,ids];c=torch.linalg.solve(block,cross[:,ids].T).T;gain=float(2*(cross[:,ids]*c).sum()-((c@block)*c).sum())
    if gain>best:best=gain;chosen=ids;writer=c;normal=float((c@block-cross[:,ids]).norm()/cross[:,ids].norm())
   pred=torch.cat([bank_entries(U,V,z)[:,chosen].double()@writer.T for z in idx.split(512)]);q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);f=(q[:,i[chosen]]*q[:,j[chosen]]).double()@writer.T;old=source_index[key]['explained_fraction_using_estimated_teacher_norm'];row=dict(optimizer=key[0],lr=key[1],seed=key[2],root_pairs=pairs[chosen].cpu().tolist(),explained_fraction_using_estimated_teacher_norm=best,full_bank_gain=old,gain_retention=best/old,evaluation_coefficient_error=float((pred-target).norm()/target.norm()),gaussian_function_error=float((f-fy).norm()/fy.norm()),normal_residual=normal,reduced_float_parameters=46080,root_support_integer_count=16);records.append(row);students[key]=dict(U=U.cpu(),V=V.cpu(),root_pairs=pairs[chosen].cpu(),C=writer.cpu());print(json.dumps(row),flush=True)
  cp=json.load(open(P/'NATIVE_EXACT_CP_GREEDY_V1.json'))['records'][-1];pred=dict(pred_a_solver=all(math.isfinite(r['explained_fraction_using_estimated_teacher_norm']) and r['normal_residual']<1e-10 for r in records),pred_b_retains=max(r['gain_retention'] for r in records)>=.9,pred_c_cp=any(r['explained_fraction_using_estimated_teacher_norm']>cp['explained_fraction_using_estimated_teacher_norm'] and r['evaluation_coefficient_error']<cp['evaluation_coefficient_error'] for r in records));guard_torch_save(dict(students=students,teacher_scale=source['teacher_scale']),str(P/'NATIVE_SHARED_BANK_PRUNE_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,seconds=time.perf_counter()-start,predictions=pred,scope='Exact supportsearch/writerrefit with frozenlearnedquadratics.46,080floats matchCP8 plus16supportintegers; no individualfeature/causalclaim.'),indent=2)+'\n')
if __name__=='__main__':main()
