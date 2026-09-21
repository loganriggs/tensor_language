#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_parent pred_c_native
"""Outputsharedrootforms primaryrank8; ranks4/16/32secondary.
Projectedroot/FP32exportreplay<1e-5, pricesmatchformula. Parenterror<=5%both;
nativetargeterror<=1.1parentboth. No semantic,OODorfullmodeladoption.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,ranks=[4,8,16,32],primary=8,features=32)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from shared_root_block_compiler import compile_root,evaluate,price
 from empirical_quartic_dictionary import evaluate as parent_evaluate
 start=time.monotonic();out=P/'SHARED_ROOT_BLOCK_NATIVE_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 parent={k:v.cuda().double() for k,v in torch.load(P/'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt',weights_only=True).items()};xs=[p['rows'].cuda().double() for p in panels];ys=[y.cuda().double() for y in data['targets']];reduced=ru.double()/scale;parent_values=[parent_evaluate(parent,x)@reduced.T for x in xs];parent_errors=[float((p-y).norm()/y.norm()) for p,y in zip(parent_values,ys)];records=[]
 for rank in [4,8,16,32]:
  program=compile_root(parent['U'],parent['V'],parent['C'],ru.double(),xs[0],rank);basis=ru.double()@program['writer'];values=[evaluate(program,x)@reduced.T for x in xs];projected=[y@basis@basis.T for y in parent_values];replay=max(float((a-b).norm()/b.norm()) for a,b in zip(values,projected));parent_error=[float((a-b).norm()/b.norm()) for a,b in zip(values,parent_values)];native_error=[float((a-b).norm()/b.norm()) for a,b in zip(values,ys)];archive={k:v.cpu().float() for k,v in program.items()};physical={k:v.cuda() for k,v in archive.items()};fresh=evaluate(physical,xs[0][:128].float()).double()@reduced.T;export=float((fresh-values[0][:128]).norm()/values[0][:128].norm());cost=price(program);assert cost['products']==128+32*rank and cost['stored_coefficients']==294912+2208*rank and cost['additions']==293600+2175*rank
  path=P/f'SHARED_ROOT_BLOCK_RANK_{rank}_V1.pt';torch.save(archive,path);row=dict(rank=rank,parent_errors=parent_error,native_errors=native_error,projected_replay=replay,export_replay=export,price=cost,program_sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(row);print(json.dumps(row),flush=True)
 primary=next(r for r in records if r['rank']==8);pred=dict(pred_a_integrity=all(max(r['projected_replay'],r['export_replay'])<1e-5 for r in records),pred_b_parent=max(primary['parent_errors'])<=.05,pred_c_native=all(a<=1.1*b for a,b in zip(primary['native_errors'],parent_errors)))
 result=dict(predictions=pred,records=records,parent_native_errors=parent_errors,parent_price=dict(products=656,stored_coefficients=903168,additions=901856),seconds=time.monotonic()-start,scope='Calibration-prediction output subspace, fixedjointlambda1parent, signedrootquadraticforms, sharedquadraticleaves. Twoopenedpanels; native targetpurequarticnotfullmodel. No derivative/OOD/semanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
