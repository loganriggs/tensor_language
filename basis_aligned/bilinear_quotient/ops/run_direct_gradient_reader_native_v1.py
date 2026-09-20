#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_major pred_c_joint
"""Gradient/random reader additions, GRADIENT_READER_NATIVE_PLAN_V1.md.
pred_a_instrument: native/unchangedfeature/compact replay<1e-5.
pred_b_major: gradient feature1error<.4cos>.9 for removal/swap bothdomains.
pred_c_joint: gradient joint error<=random bothinterventions/bothdomains.
pred_d_gain: code feature1 MSE<=.9random bothinterventions.
Null: gradient proposals do not improve native effects versus matched random.
Price192forwards;18572scalar+4608writercoefficients14products each.
"""
import os,sys,json,hashlib
import run_direct_selective_confirmation_v1 as runner

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=192,products=14,scalar_coefficients=18572)));return
 import torch
 from disk_guard import guard_torch_save
 P=runner.mode.P;sys.path.insert(0,str(P))
 from gradient_reader_additions import expand,evaluate
 from extract_scalar_modes import evaluate as scalar_evaluate
 source=torch.load(P/'GRADIENT_READER_ADDITIONS_V1.pt',weights_only=True);old=torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True);view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].double();mu=view['output_mean'].double();x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][:64].double();adapters={};specs={};checks=[]
 for family in ['random','gradient']:
  key=family+'_4';s={k:v.double() for k,v in source['programs'][key].items()};expanded=expand(s);a={k:expanded[k] for k in ['A','B','root_left','root_right']};a.update(output_writer=U@expanded['quartic_readout'],skip_writer=U@expanded['quadratic_readout'],constant=mu+U@expanded['constant']);adapters[family]=a
  ref=evaluate(s,x);checks.append(float((scalar_evaluate(expanded,x)-ref).norm()/ref.norm()))
  artifact=f'GRADIENT_READER_NATIVE_{family.upper()}_V1.pt';guard_torch_save(dict(program=expanded,residual_writer=old['residual_writer'],teacher_scale=old['teacher_scale'],scope='Expanded evaluation adapter only. Priced executable is compact gradient_reader_additions DAG.'),str(P/artifact));specs[family]=('GRADIENT_READER_NATIVE_ADAPTERS_V1.pt',family,artifact)
 assert max(checks)<1e-5
 guard_torch_save(dict(programs=adapters),str(P/'GRADIENT_READER_NATIVE_ADAPTERS_V1.pt'))
 runner.SPECS=specs;runner.OUTPUT_PREFIX='GRADIENT_READER_NATIVE';runner.RESULT_SCOPE='Matched gradient4 versus random4, reused selective-confirmation panels. Compact candidates18572scalarcoeff14products. Data-informed calibration fitting; no fresh/semantic claim.';runner.main()
 path=P/'GRADIENT_READER_NATIVE_V1.json';r=json.loads(path.read_text());r['compact_expanded_replay']=max(checks);r['source_sha256']=hashlib.sha256((P/'GRADIENT_READER_ADDITIONS_V1.pt').read_bytes()).hexdigest();r['selective_baseline']=json.loads((P/'SELECTIVE_CONFIRMATION_V1.json').read_text())['results'];path.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(dict(pred_a_instrument=r['predictions']['pred_a_instrument'],pred_b_major=r['predictions']['pred_b_major'],pred_c_joint=r['predictions']['pred_c_joint'],pred_d_gain=r['predictions']['pred_d_gain'])))
if __name__=='__main__':main()
