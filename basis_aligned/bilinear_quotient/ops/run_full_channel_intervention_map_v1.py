#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_alignment pred_b_retained pred_c_global
"""Weight-only audit of declared teacher/student hidden-product interventions.
Aligned reader replay<1e-6; >=90% retained writers error<10%; all-channel
isotropic hidden-intervention relative error<20%. No text forwards.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=0,retained=3686,omitted=922)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 p={k:v.cuda().double() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()};record=next(r for r in json.loads((P/'FULL_CHANNEL_DELETION_V1.json').read_text())['records'] if r['geometry']=='activation_covariance' and r['policy']=='conditional' and r['width']==3686);ids=torch.tensor(record['selected_channels'],device='cuda');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.17.mlp.Down.weight');U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 an=L[ids].norm(dim=1)/p['a'].norm(dim=1);bn=R[ids].norm(dim=1)/p['b'].norm(dim=1);alignment=max(float((p['a']*an[:,None]-L[ids]).norm()/L[ids].norm()),float((p['b']*bn[:,None]-R[ids]).norm()/R[ids].norm()));teacher=RU@D;student=RU@(p['writer']/(an*bn));truth=teacher[:,ids];err=(student-truth).norm(dim=0)/truth.norm(dim=0);cos=(student*truth).sum(0)/(student.norm(dim=0)*truth.norm(dim=0));omitted=torch.ones(4608,device='cuda',dtype=torch.bool);omitted[ids]=False;global_error=float(((student-truth).square().sum()+teacher[:,omitted].square().sum()).sqrt()/teacher.norm());retained_error=float((student-truth).norm()/truth.norm());fraction=float((err<.1).double().mean());q=torch.tensor([0.,.5,.9,.95,.99,1.],device='cuda',dtype=torch.float64)
 # Exact controlled change: arbitrary source activation change delta, mapped through scale.
 gen=torch.Generator(device='cuda').manual_seed(939);delta=torch.randn(3686,generator=gen,device='cuda',dtype=torch.float64);direct=RU@(p['writer']@(delta/(an*bn))-D[:,ids]@delta);predicted=(student-truth)@delta;control=float((direct-predicted).norm()/direct.norm());assert control<1e-10
 out=dict(reader_alignment_error=alignment,intervention_algebra_replay=control,quantile_levels=q.cpu().tolist(),retained_relative_error_quantiles=torch.quantile(err,q).cpu().tolist(),retained_cosine_quantiles=torch.quantile(cos,q).cpu().tolist(),retained_fraction_below_10pct=fraction,retained_aggregate_intervention_error=retained_error,all_channel_intervention_error=global_error,omitted_channels=922,omitted_writer_energy_fraction=float(teacher[:,omitted].square().sum()/teacher.square().sum()),predictions=dict(pred_a_alignment=alignment<1e-6,pred_b_retained=fraction>=.9,pred_c_global=global_error<.2),seconds=time.monotonic()-start,scope='Pre-final-normalization logit effects under independent unit-variance perturbations to native hidden products, with retained-coordinate scaling alignment and omitted perturbations mapped to no student node. Affine branch held fixed because input held fixed. This is a specific proposed intervention map, not a unique semantic alignment, nor a proof that all causal abstractions fail.')
 (P/'FULL_CHANNEL_INTERVENTION_MAP_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
