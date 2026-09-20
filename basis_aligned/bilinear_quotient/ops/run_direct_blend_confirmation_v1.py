#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_feature pred_c_preservation
"""Fresh frozen confirmation; METRIC_BLEND_CONFIRMATION_PLAN_V1.md.
pred_a_instrument: replay<1e-5, program/panel hashes match, source disjointness.
pred_b_feature: primarymode1 error<.4 and cosine>.9 BOTH domains.
pred_c_preservation: primary branch CE<.02 and KL<.02 BOTH domains.
Null: reused diagnostic success does not confirm on new files/context256.
Price:primary/control21948coeff10products, base19632/10. No fitting/selection.
"""
import os,json,hashlib
import run_direct_native_quartic_branch_v1 as branch
import run_direct_native_mode_intervention_v1 as mode

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(context=256,fineweb_documents=32,code_documents=16,native_forwards=144,primary_alpha=.5)));return
 import torch
 P=branch.P;out=P/'BLEND_CONFIRMATION_V1.json';assert not out.exists();meta=json.loads((P/'BLEND_CONFIRMATION_PANELS_V1.json').read_text());assert hashlib.sha256((P/'SKIP_METRIC_BLEND_PROGRAM_V1.pt').read_bytes()).hexdigest()==meta['program_sha256'];assert not ({r['path'] for r in meta['code_sources']} & set(meta['excluded_code_sources']));assert set(meta['fineweb_documents']).isdisjoint(range(96));records={};checks=[]
 for domain,n in [('fineweb',32),('code',16)]:
  panel=P/f'BLEND_CONFIRMATION_{domain.upper()}_V1.pt';tokens=torch.load(panel,weights_only=True);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==meta['panels'][domain]['token_sha256']
  branch.PANEL_PATH=panel;branch.PROGRAM_SPECS=dict(primary=('SKIP_METRIC_BLEND_PROGRAM_V1.pt',.5),gaussian=('SKIP_METRIC_BLEND_PROGRAM_V1.pt',0.),base=('FUSED_ROOT_PROGRAM_V1.pt',4));branch.PRIMARY='primary';branch.OUTPUT_STEM=f'BLEND_CONFIRMATION_BRANCH_{domain.upper()}_V1';branch.PLAN=dict(branch.PLAN,documents=list(range(n)),context=256,arms=['exact','ablation','primary','gaussian','base'],native_forwards=n,program_specs=branch.PROGRAM_SPECS)
  branch.main();r=json.loads((P/(branch.OUTPUT_STEM+'.json')).read_text());checks.extend([r['checks_max'],r['solve_error']]);records[domain]=dict(branch=r['summary'],modes={})
  for label,alpha in [('primary',.5),('gaussian',0.)]:
   mode.PANEL_PATH=panel;mode.PROGRAM_FILE='SKIP_METRIC_BLEND_PROGRAM_V1.pt';mode.PROGRAM_KEY=alpha;mode.OUTPUT_STEM=f'BLEND_CONFIRMATION_MODE_{domain.upper()}_{label.upper()}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(n)),context=256,native_forwards=n,program_file=mode.PROGRAM_FILE,program_key=alpha)
   mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());checks.append(r['replay_max']);records[domain]['modes'][label]=r['summary']
 pred=dict(pred_a_instrument=max(checks)<1e-5,pred_b_feature=all(r['modes']['primary']['1']['effect_relative_error']<.4 and r['modes']['primary']['1']['effect_cosine']>.9 for r in records.values()),pred_c_preservation=all(r['branch']['primary']['ce_added']<.02 and r['branch']['primary']['kl']<.02 for r in records.values()))
 result=dict(context=256,primary_alpha=.5,records=records,predictions=pred,replay_max=max(checks),panel_manifest='BLEND_CONFIRMATION_PANELS_V1.json',program_hash=meta['program_sha256'],scope='Frozen primary and controls on new documents/code files at longer context. Limited local-code domain shift, no semantic selectivity or whole-model speedup claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,primary={d:dict(branch=r['branch']['primary'],mode1=r['modes']['primary']['1']) for d,r in records.items()}),indent=2))
if __name__=='__main__':main()
