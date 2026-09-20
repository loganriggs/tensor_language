#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_fixed pred_c_dense
"""Native decoder comparison; PRIMITIVE_DECODER_PLAN_V1.md.
pred_a_replay: native targets and token hashes replay<1e-5.
pred_b_fixed: code mode1 error<.4, cosine>.9, MSEgain>=10% over frozen.
pred_c_dense: code mode1 MSEgain>=10% over fixed.
Null: cached polynomial gains fail native feature-effect transfer.
Price:96 native forwards; fixed13916coeff10products, dense13936coeff27products,
plus4608fixed writercoefficients. Native background retained.
"""
import os,json
import run_direct_native_mode_intervention_v1 as mode

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(native_forwards=96,context=256,arms=2)));return
 import torch
 from disk_guard import guard_torch_save
 P=mode.P;output=P/'PRIMITIVE_DECODER_NATIVE_V1.json';assert not output.exists()
 saved=torch.load(P/'PRIMITIVE_DECODER_PROGRAMS_V1.pt',weights_only=True);view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].double();mu=view['output_mean'].double();adapters={};adapter_checks=[]
 import sys
 sys.path.insert(0,str(P))
 from extract_scalar_modes import evaluate
 from frozen_program_evaluation import quartic
 x=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'][0]['rows'][:64].double()
 for key in ['fixed_0.0001','dense_0.0001']:
  s={k:v.double() for k,v in saved['programs'][key].items()};a={k:s[k] for k in ['A','B','root_left','root_right']};a.update(output_writer=U@s['quartic_readout'],skip_writer=U@s['quadratic_readout'],constant=mu+U@s['constant']);adapters[key]=a
  ref=evaluate(s,x);adapter_checks.append(float(((quartic(a,x)-mu)@U-ref).norm()/ref.norm()))
 guard_torch_save(dict(programs=adapters),str(P/'PRIMITIVE_DECODER_NATIVE_ADAPTERS_V1.pt'))
 results={};checks=list(adapter_checks)
 for domain,n in [('FINEWEB',32),('CODE',16)]:
  results[domain.lower()]={};old=json.loads((P/f'BLEND_CONFIRMATION_MODE_{domain}_PRIMARY_V1.json').read_text())
  for family in ['fixed','dense']:
   key=family+'_0.0001';mode.PANEL_PATH=P/f'BLEND_CONFIRMATION_{domain}_V1.pt';mode.PROGRAM_FILE='PRIMITIVE_DECODER_NATIVE_ADAPTERS_V1.pt';mode.PROGRAM_KEY=key;mode.EXTRACTED_FILE=None;mode.OUTPUT_STEM=f'PRIMITIVE_DECODER_NATIVE_{domain}_{family.upper()}_V1';mode.PLAN=dict(mode.PLAN,documents=list(range(n)),context=256,native_forwards=n,program_file=mode.PROGRAM_FILE,program_key=key)
   mode.main();r=json.loads((P/(mode.OUTPUT_STEM+'.json')).read_text());assert r['token_hash']==old['token_hash'];checks.append(r['replay_max'])
   for a,b in zip(r['records'],old['records']):
    assert (a['document'],a['mode'])==(b['document'],b['mode']);checks.append(abs(a['native_effect_energy']-b['native_effect_energy'])/max(b['native_effect_energy'],1e-30))
   results[domain.lower()][family]=r['summary']
  results[domain.lower()]['frozen']=old['summary']
 c=results['code'];fixed=c['fixed']['1'];dense=c['dense']['1'];frozen=c['frozen']['1']
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_fixed=fixed['effect_cosine']>.9 and fixed['effect_relative_error']<.4 and fixed['effect_relative_error']**2<.9*frozen['effect_relative_error']**2,pred_c_dense=dense['effect_relative_error']**2<.9*fixed['effect_relative_error']**2)
 result=dict(results=results,predictions=pred,replay_max=max(checks),scope='Fixed original-calibration readout fits tested on reused native confirmation panels. No fresh confirmation or semantic selectivity claim. Scalar costs13916/10 and13936/27; vector adapters are evaluation scaffolding.')
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
