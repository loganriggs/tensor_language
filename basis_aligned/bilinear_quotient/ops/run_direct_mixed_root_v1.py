#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_retention pred_c_cp
"""Native validation of eight mixed roots: pred_a_replay pred_b_retention pred_c_cp."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(students=2,roots=8,mixing_scalars=16,coefficient_queries=8192,gaussian_rows=256,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from shared_quadratic_bank import bank_gram,native_bank_cross,bank_entries,roots
 from root_basis_search import symmetric_square
 from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_MIXED_ROOT_V1.json';assert not out.exists();start=time.perf_counter();source=torch.load(P/'ROOT_BASIS_REFINE_V1.pt',weights_only=True)['students'];old={(r['optimizer'],r['lr'],r['seed']):r for r in json.load(open(P/'NATIVE_LEARNED_SHARED_BANK_V1.json'))['records']};state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];gen=torch.Generator(device='cuda');gen.manual_seed(1651);idx=torch.randint(1152,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in idx.split(512)]).double();x=torch.randn(256,1152,device='cuda',generator=gen);C,L,R,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;fy=(((h@L.T)*(h@R.T))@C.T).double();i,j=roots(4,x.device);rows=[];exports={}
  for key,s in source.items():
   U=s['U'].cuda();V=s['V'].cuda();S=s['mixing'].cuda();support=s['support'].cuda();T=symmetric_square(S)[support];G=bank_gram(U,V);G=(G+G.T)/2;cross=native_bank_cross(teacher,U.float(),V.float()).double()@T.T;H=T@G@T.T;writer=torch.linalg.solve(H,cross.T).T;normal=float((writer@H-cross).norm()/cross.norm());features=torch.cat([bank_entries(U,V,z)@T.T for z in idx.split(512)]);q=((x.double()@U.flatten(0,1).T)*(x.double()@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);mixed=q@S.T;direct=mixed[:,i[support]]*mixed[:,j[support]];transport=(q[:,i]*q[:,j])@T.T;replay=float((direct-transport).norm()/direct.norm())
   for name,writer_here in [('student_refit',s['C'].cuda()),('native_refit',writer)]:
    gain=float(2*(cross*writer_here).sum()-((writer_here@H)*writer_here).sum());pred=features@writer_here.T;f=direct@writer_here.T;row=dict(source_seed=key[2],writer=name,explained_fraction_using_estimated_teacher_norm=gain,gain_retention=gain/old[key]['explained_fraction_using_estimated_teacher_norm'],evaluation_coefficient_error=float((pred-target).norm()/target.norm()),gaussian_function_error=float((f-fy).norm()/fy.norm()),feature_replay=replay,normal_residual=normal,reduced_float_parameters=46096,support_integers=16);rows.append(row);print(json.dumps(row),flush=True)
   exports[key]=dict(U=U.cpu(),V=V.cpu(),mixing=S.cpu(),support=support.cpu(),C=writer.cpu())
  native=[r for r in rows if r['writer']=='native_refit'];cp=json.load(open(P/'NATIVE_EXACT_CP_GREEDY_V1.json'))['records'][-1];predictions=dict(pred_a_replay=all(math.isfinite(r['explained_fraction_using_estimated_teacher_norm']) and r['feature_replay']<1e-10 and r['normal_residual']<1e-10 for r in native),pred_b_retention=all(r['gain_retention']>=.95 for r in native),pred_c_cp=all(r['explained_fraction_using_estimated_teacher_norm']>cp['explained_fraction_using_estimated_teacher_norm'] and r['evaluation_coefficient_error']<cp['evaluation_coefficient_error'] for r in native));guard_torch_save(dict(students=exports,teacher_scale=scale),str(P/'NATIVE_MIXED_ROOT_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=rows,predictions=predictions,seconds=time.perf_counter()-start,scope='Eight mixedroot native cross/self validation; estimatedteacher norm.46,096floating scalars+16supportintegers. No identifiedcircuit.'),indent=2)+'\n')
if __name__=='__main__':main()
