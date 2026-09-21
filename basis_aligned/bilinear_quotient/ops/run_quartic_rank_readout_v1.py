#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_values pred_c_derivatives
"""Exact joint rankconstrained rootblocks, primaryrank8,lambda1fixed.
Integrityspectralcertificate<1e-6,replay<1e-5,matchedprices. Primaryvalues
andderivatives<=1.1parentonbothpanels. No semantic orfullmodeladoption.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,panels=2,anchors=16,ranks=[4,8,16,32],features=528)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from quartic_reader_span import jacobian
 from quartic_bank_derivative import feature_jacobian
 from empirical_quartic_dictionary import features,evaluate
 from quartic_joint_readout import solve_rank
 from shared_root_block_compiler import compile_basis,expand_root_writer,evaluate as block_evaluate,price
 start=time.monotonic();out=P/'QUARTIC_RANK_READOUT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 old=torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True);U,V=old['U'].cuda().double(),old['V'].cuda().double();L=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms']['centered_floor01'].cuda().double();xs=[p['rows'].cuda().double() for p in panels];ys=[y.cuda().double() for y in data['targets']];phi=[features(x,U,V) for x in xs];stats=[]
 for x in xs:
  dg=torch.zeros(528,528,device='cuda',dtype=torch.float64);dc=torch.zeros(1152,528,device='cuda',dtype=torch.float64);energy=torch.zeros((),device='cuda',dtype=torch.float64)
  for index in torch.linspace(0,len(x)-1,16).long():
   z=x[index];j=jacobian(teacher,z.float()).double()@L;p=feature_jacobian(U,V,z)@L;dg.add_(p@p.T);dc.add_(j@p.T);energy.add_(j.square().sum())
  stats.append((dg,dc,energy))
 records=[];parent=torch.load(P/'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt',weights_only=True);parent_writer=ru.double()@parent['C'].cuda().double()/scale;parent_values=[p@parent_writer.T for p in phi];scales=phi[0].square().mean(0).sqrt();ridge=len(phi[0])*1e-6/ys[0].square().sum()
 def metrics(writer):
  values=[float((p@writer.T-y).norm()/y.norm()) for p,y in zip(phi,ys)];derivatives=[]
  for dg,dc,energy in stats:
   squared=(((writer@dg)*writer).sum()-2*(writer*dc).sum()+energy)/energy;assert float(squared)>-1e-10;derivatives.append(float(squared.clamp_min(0).sqrt()))
  objective=values[0]**2+derivatives[0]**2+float(ridge*(writer*scales).square().sum())
  return values,derivatives,objective
 for rank in [4,8,16,32]:
  writer,basis,info=solve_rank(phi[0],ys[0],*stats[0],1.,rank);physical=torch.linalg.solve(ru.double(),writer*scale);program=compile_basis(U,V,physical,ru.double(),basis);expanded=expand_root_writer(program);coef=float((expanded-physical).norm()/physical.norm());value,derivative,objective=metrics(writer)
  archive={k:v.cpu().float() for k,v in program.items()};reloaded={k:v.cuda() for k,v in archive.items()};actual=block_evaluate(reloaded,xs[0][:128].float()).double()@ru.double().T/scale;expected=phi[0][:128]@writer.T;replay=float((actual-expected).norm()/expected.norm());cost=price(program);assert cost['products']==128+32*rank and cost['stored_coefficients']==294912+2208*rank
  oldblock={k:v.cuda().double() for k,v in torch.load(P/f'SHARED_ROOT_BLOCK_RANK_{rank}_V1.pt',weights_only=True).items()};oldwriter=ru.double()@expand_root_writer(oldblock)/scale;oldvalues,oldderivatives,oldobjective=metrics(oldwriter)
  path=P/f'QUARTIC_RANK_BLOCK_{rank}_V1.pt';torch.save(archive,path);row=dict(rank=rank,value_errors=value,derivative_errors=derivative,joint_objective=objective,prediction_pca_joint_objective=oldobjective,prediction_pca_derivative_errors=oldderivatives,parent_approximation_errors=[float((p@writer.T-y).norm()/y.norm()) for p,y in zip(phi,parent_values)],spectral_certificate=info['spectral_objective_relative_error'],coefficient_replay=coef,export_replay=replay,price=cost,program_sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(row);print(json.dumps(row),flush=True)
 primary=next(r for r in records if r['rank']==8);pred=dict(pred_a_integrity=all(r['spectral_certificate']<1e-6 and max(r['coefficient_replay'],r['export_replay'])<1e-5 and r['joint_objective']<=r['prediction_pca_joint_objective']+1e-6 for r in records),pred_b_values=all(a<=1.1*b for a,b in zip(primary['value_errors'],[.07744491681,.13609101747])),pred_c_derivatives=all(a<=1.1*b for a,b in zip(primary['derivative_errors'],[.20696935402,.27733554458])))
 result=dict(predictions=pred,records=records,seconds=time.monotonic()-start,scope='Globally optimal fixed-dictionary output-rank writer for calibration jointquadraticobjective includingridge. Doesnotgloballyoptimizefeatures orgraphs. Openedpanels, purequartictarget, nosemanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
