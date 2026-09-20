#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_frame pred_b_scalar pred_c_intervention
"""Export executable scalar interventions; SCALAR_EXTRACTION_PLAN_V1.md.
pred_a_frame: writer frame reprojection<1e-5.
pred_b_scalar: scalar/native extracted replay<1e-5.
pred_c_intervention: archived intervention/target energies replay<1e-5.
Null: hidden frame/rounding error breaks extraction. No new accuracy claim.
Price:13916coeff10products shared scalar library plus4608writer coefficients.
"""
import os,json,sys
from pathlib import Path
import run_direct_native_mode_intervention_v1 as mode

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=16,context=256,features=4,products=10,coefficients=18524)));return
 import torch
 from disk_guard import guard_torch_save
 P=mode.P;out=P/'SCALAR_EXTRACTION_NATIVE_V1.json';assert not out.exists();torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 scalar=torch.load(P/'EXTRACTED_SCALAR_MODES_V1.pt',weights_only=True);view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].cuda().double();scale=float(torch.load(P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt',weights_only=True)['teacher_scale']);state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True);_,ru=torch.linalg.qr(state['lm_head.weight'].cuda().float());del state;ru=ru.double();writer=torch.linalg.solve_triangular(ru,scale*U,upper=True).float();frame=float((ru@writer.double()-scale*U).norm()/(scale*U).norm());scalar['residual_writer']=writer.cpu();scalar['teacher_scale']=scale;guard_torch_save(scalar,str(P/'EXTRACTED_SCALAR_INTERVENTIONS_V1.pt'))
 mode.PANEL_PATH=P/'BLEND_CONFIRMATION_CODE_V1.pt';mode.PROGRAM_FILE='SKIP_METRIC_BLEND_PROGRAM_V1.pt';mode.PROGRAM_KEY=.5;mode.EXTRACTED_FILE=P/'EXTRACTED_SCALAR_INTERVENTIONS_V1.pt';mode.OUTPUT_STEM='SCALAR_EXTRACTION_MODE_CODE_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(16)),context=256,native_forwards=16,program_file=mode.PROGRAM_FILE,program_key=.5,extracted_file=mode.EXTRACTED_FILE.name)
 mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());old=json.loads((P/'BLEND_CONFIRMATION_MODE_CODE_PRIMARY_V1.json').read_text());assert r['token_hash']==old['token_hash'];checks=[]
 for a,b in zip(r['records'],old['records']):
  assert (a['document'],a['mode'])==(b['document'],b['mode'])
  for k in ['native_effect_energy','predicted_effect_energy','effect_dot']:checks.append(abs(a[k]-b[k])/max(abs(b[k]),1e-30))
 pred=dict(pred_a_frame=frame<1e-5,pred_b_scalar=r['replay_max']<1e-5,pred_c_intervention=max(checks)<1e-5)
 result=dict(predictions=pred,writer_reprojection=frame,scalar_and_native_replay=r['replay_max'],archived_intervention_replay=max(checks),shared_scalar_coefficients=13916,residual_writer_coefficients=4608,products=10,scope='Executable exact extraction of fixed operational features with native residual background. Prior feature1/KL failures remain unchanged; no semantic or standalone-model claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
