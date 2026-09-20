#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_identity pred_b_sign pred_c_gap
"""Frozen gain under matched panel moments; SKIP_MOMENT_MATCH_PLAN_V1.md.
pred_a_identity: calibration gain replay<1e-6, shifted-centering oracle<1e-10.
pred_b_sign: both matched Gaussians predict positive mode1gain despite textloss.
pred_c_gap: both mode1 gain discrepancies>10% of actual baseline mode1MSE.
Null: first-two-moment shift explains sign reversal. No refitting, no extra model.
Price: fixed21948coeff10product correction, analytic diagnostic only.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(contexts=[64,256],rank=2,native_forwards=0,fit=False)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P))
 from frozen_program_evaluation import load_teacher,quartic
 from gaussian_quadratic_skip import cross_covariance
 from gaussian_quartic_mean import native_mean,quadratic_moments
 from gaussian_frozen_skip_gain import mode_gain
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'SKIP_MOMENT_MATCH_V1.json';assert not out.exists();start=time.perf_counter()
 source=torch.load(P/'QUADRATIC_SKIP_PROGRAM_V1.pt',weights_only=True);s={k:t.cuda().double() for k,t in torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4].items()};new={k:t.cuda().double() for k,t in source['programs'][2].items()};W=new['skip_writer']@new['skip_reader'];c=source['primitive_mean'].cuda().double();A,B=s['A'],s['B'];U=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'].cuda().double();teacher=[t.double() for t in load_teacher('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',source['teacher_scale'])];base=[s['output_writer'],s['root_left'],s['root_right'],torch.eye(6,device='cuda',dtype=torch.float64),A,B]
 # Fixed-centering identity: q=0 removes residual-mean terms exactly.
 archived=source['cross_covariance'].cuda().double();G0=source['primitive_covariance'].cuda().double();cal=mode_gain(archived,G0,W,torch.zeros(1152,device='cuda',dtype=torch.float64),torch.zeros_like(c),U);previous=next(r for r in json.loads((P/'SKIP_MODE_OBJECTIVE_AUDIT_V1.json').read_text())['records'] if r['family']=='QUADRATIC' and r['rank']==2);reference=cal.new_tensor(previous['exact_gaussian_mode_mse_reduction']);replay=float((cal-reference).norm()/reference.norm());rows=[]
 for panel in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']:
  x=panel['rows'].cuda().double();y=panel['targets'].cuda().double();mu=x.mean(0);center=x-mu;M=center.T@center/len(x);mp,G=quadratic_moments(A,B,mu,M);K=cross_covariance(teacher,mu,M,A,B)-cross_covariance(base,mu,M,A,B);b=native_mean(teacher,mu,M)-native_mean(base,mu,M)-s['constant'];gain=mode_gain(K,G,W,b,mp-c,U)
  old=(y-quartic(s,x))@U;changed=(y-quartic(new,x))@U;baseline=old.square().mean(0);actual=baseline-changed.square().mean(0);meanonly=mode_gain(K,G,W,torch.zeros_like(b),torch.zeros_like(c),U)
  row=dict(context=panel['context'],matched_gaussian_gain=gain.cpu().tolist(),actual_gain=actual.cpu().tolist(),actual_baseline_mse=baseline.cpu().tolist(),relative_gain_gap=((gain-actual).abs()/baseline).cpu().tolist(),centering_terms=(gain-meanonly).cpu().tolist());rows.append(row);print(row,flush=True)
 oracle=json.loads((P/'FROZEN_SKIP_GAIN_ORACLE_V1.json').read_text());pred=dict(pred_a_identity=replay<1e-6 and oracle['shifted_centering_relative_error']<1e-10,pred_b_sign=all(r['matched_gaussian_gain'][1]>0 and r['actual_gain'][1]<0 for r in rows),pred_c_gap=all(r['relative_gain_gap'][1]>.1 for r in rows))
 result=dict(plan=PLAN,records=rows,predictions=pred,calibration_replay=replay,oracle=oracle,seconds=time.perf_counter()-start,scope='Frozen rank2 correction, actual-panel matched Gaussian mean/covariance, calibration centering retained. No fit or new OOD evaluation.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
