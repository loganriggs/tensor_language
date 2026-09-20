#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_numerics pred_b_objective pred_c_evaluation
"""Exact native quadratic-product Gram, eight fixed roots.
pred_a_numerics fp32/fp64<1e-4 & solve<1e-10; pred_b_objective monotone;
pred_c_evaluation heldout error improves>=1e-4. No native forward data.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(student_roots=8,teacher_roots=4608,teacher_block=16,evaluation_queries=8192,function_queries=256,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from quadratic_product_gram import product_gram
 from quartic_root_features import root_entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_EXACT_ROOT_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'NATIVE_HIERARCHICAL_ROOT_V1.pt',weights_only=True);s=source['students'][8];state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'];L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A=source['A'].cuda();B=source['B'].cuda();WL=L@D;WR=R@D
  def matrices(weights):
   raw=A.T@(weights[:,:,None]*B[None,:,:]);Q=(raw+raw.transpose(-1,-2))/2;norm=Q.flatten(1).norm(dim=1);return Q/norm[:,None,None],norm
  S,sn=matrices(s['left'].cuda());T,tn=matrices(s['right'].cuda());student_scale=(sn*tn).double();G=product_gram(S,T,S,T).double();G=(G+G.T)/2;all_cross=[];precision=None;torch.cuda.synchronize();scan_start=time.perf_counter()
  for offset in range(0,4608,16):
   Q,qn=matrices(WL[offset:offset+16]);V,vn=matrices(WR[offset:offset+16]);gram=product_gram(Q,V,S,T)
   if offset==0:
    reference=product_gram(Q.double(),V.double(),S.double(),T.double());precision=float((gram.double()-reference).norm()/reference.norm());assert precision<1e-4
   all_cross.append(gram.double()*(qn.double()*vn.double())[:,None])
   if offset%1024==0:print('teacher roots contracted',offset+16,flush=True)
  cross=C.double()@torch.cat(all_cross);torch.cuda.synchronize();scan_seconds=time.perf_counter()-scan_start;c=torch.linalg.solve(G,cross.T).T;residual=float((c@G-cross).norm()/cross.norm());old=s['writer'].cuda().double()*student_scale;gain=lambda v:float(2*(cross*v).sum()-((v@G)*v).sum());oldgain=gain(old);newgain=gain(c);raw=c/student_scale;gen=torch.Generator(device='cuda');gen.manual_seed(1651);ix=torch.randint(1152,(8192,4),device='cuda',generator=gen);features=[];targets=[];ids=s['root_indices'].cuda()
  for batch in ix.split(512):
   roots=root_entries(L,R,D,A,B,batch);features.append(roots[:,ids]);targets.append(roots@C.T)
  F=torch.cat(features).double();Y=torch.cat(targets).double();x=torch.randn(256,1152,device='cuda',generator=gen);m=(x@A.T)*(x@B.T);h=m@D.T;fy=((h@L.T)*(h@R.T))@C.T;fx=(m@s['left'].cuda().T)*(m@s['right'].cuda().T);oldraw=s['writer'].cuda().double();records=[]
  for name,v,g in [('sampled_writer',oldraw,oldgain),('exact_writer',raw,newgain)]:records.append(dict(method=name,explained_fraction_using_estimated_teacher_norm=g,estimated_coefficient_error=math.sqrt(max(0,1-g)),evaluation_coefficient_error=float((F@v.T-Y).norm()/Y.norm()),gaussian_function_error=float((fx.double()@v.T-fy.double()).norm()/fy.double().norm())))
  result=dict(plan=PLAN,records=records,scan_seconds=scan_seconds,seconds=time.perf_counter()-start,kernel_precision=precision,normal_residual=residual,condition=float(torch.linalg.cond(G)),predictions=dict(pred_a_numerics=precision<1e-4 and residual<1e-10,pred_b_objective=newgain>=oldgain-1e-10,pred_c_evaluation=records[1]['evaluation_coefficient_error']<records[0]['evaluation_coefficient_error']-1e-4),scope='Full teacher self/cross weight contractions for fixed eight-root student, fp32 kernels/fp64 solve. Teacher norm only estimated; heldout metrics finite. No feature discovery or circuitclaim.');guard_torch_save(dict(writer=raw.cpu(),G=G.cpu(),cross=cross.cpu(),student_scale=student_scale.cpu(),source='NATIVE_HIERARCHICAL_ROOT_V1.pt'),str(P/'NATIVE_EXACT_ROOT_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
