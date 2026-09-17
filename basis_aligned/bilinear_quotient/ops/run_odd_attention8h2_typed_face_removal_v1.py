#!/usr/bin/env python3
# BQGATE:960bodyforwards;48prefixes;300seconds;no fitting.
"""pred_a norm/closure; pred_b capability and midpoint attenuation.
pred_c sixteen equal-norm same-head nulls; pred_d four unrelated readouts.
pred_e fresh target replay/half-full response; bars in frozen preregistration.
"""
import hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from odd_attention8h2_typed_face_removal_runtime_v1 import measure_lattice
from run_even_value_factorial_native_v1 import setup
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
STEM='ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1'
spec=importlib.util.spec_from_file_location('compiled_face',D/'native.py')
compiled=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiled)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows']
    control=json.loads((P/(STEM+'_CPU_CONTROL.json')).read_text())
    assert control['pred_a'] and control['prior_source_overlap']==0 and len(rows)==48
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert len({r['context_id'] for r in rows})==4
        p=torch.load(D/'program.pt',map_location='cpu',weights_only=True)
        assert all(r['ids'][r['city_position']] in p['token_ids'].tolist() for r in rows)
        print('960bodyforwards;48prefixes;4contexts;16normmatchednulls;frozen program')
        return
    output=P/(STEM+'_RESULT.json');assert not output.exists()
    start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda')
    p={k:v.to('cuda') for k,v in torch.load(D/'program.pt',weights_only=True).items()}
    measured=measure_lattice(model,graph,[],rows,compiled,p);v=measured['values']
    assert measured['body_forwards']==960 and bool(torch.isfinite(v).all())
    effects=v-v[0:1];mid=effects[7];nulls=effects[8:24]
    native_pair=v[0,::2,0]-v[0,1::2,0]
    midpoint_pair=v[7,::2,0]-v[7,1::2,0]
    capable=native_pair>=.1
    atten=(native_pair[capable]-midpoint_pair[capable])/native_pair[capable]
    target_rms=float(mid[:,0].square().mean().sqrt())
    null_rms=nulls[:,:,0].square().mean(1).sqrt()
    sorted_nulls=null_rms.sort().values
    median=float((sorted_nulls[7]+sorted_nulls[8])/2)
    control_ratios=(mid[:,1:].square().mean(0).sqrt()/max(target_rms,1e-30)).tolist()
    replay=((effects[6]-effects[5]).norm(dim=0)/effects[5].norm(dim=0).clamp_min(1e-8)).tolist()
    half_error=float((mid[:,0]-.5*effects[6,:,0]).norm()/(.5*effects[6,:,0]).norm().clamp_min(1e-8))
    pred_a=max(measured['compiled_errors'])<=1e-5 and max(measured['table_errors'])<=1e-5 and max(measured['null_norm_errors'])<=1e-5 and max(measured['outside'])==0
    attenuation_fraction=float((atten>0).double().mean()) if len(atten) else 0.
    attenuation_mean=float(atten.mean()) if len(atten) else 0.
    pred_b=int(capable.sum())>=18 and attenuation_fraction>=.75 and attenuation_mean>=.02
    pred_c=target_rms>=2*median and int((target_rms>null_rms).sum())>=15
    pred_d=max(control_ratios)<=.5
    pred_e=replay[0]<=.01 and half_error<=.1
    result={'pred_a':pred_a,'pred_b':pred_b,'pred_c':pred_c,'pred_d':pred_d,'pred_e':pred_e,
      'terminal':'fresh_midpoint_screen_pass' if all((pred_a,pred_b,pred_c,pred_d,pred_e)) else 'fresh_midpoint_screen_failed_gate',
      'capable_cells':int(capable.sum()),'total_cells':24,'attenuation_positive_fraction':attenuation_fraction,'attenuation_mean':attenuation_mean,
      'target_rms_logits':target_rms,'null_target_rms_logits':null_rms.tolist(),'target_over_median_null':target_rms/max(median,1e-30),
      'nulls_beaten':int((target_rms>null_rms).sum()),'control_over_target_rms':control_ratios,
      'full_replay_errors':replay,'half_full_target_error':half_error,
      'max_write_error':max(measured['compiled_errors']),'max_table_error':max(measured['table_errors']),
      'max_null_norm_error':max(measured['null_norm_errors']),'body_forwards':960,'seconds':time.perf_counter()-start,
      'contexts':4,'token_sequences':8,'rows':48,'panel_status':'fresh at freeze; source-disjoint from two earlier panels',
      'program_sha256':digest(D/'program.pt'),'source_shas':binding,
      'scope':'Paired midpoint, same-head equal-norm nulls, four controls; conditional native suffix. Earlier strict replay failure remains. No independent composition or broad-model adoption.'}
    torch.save({'values':v,'native_pair':native_pair,'capable':capable,'attenuation':atten},P/(STEM+'_ARTIFACT.pt'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)

if __name__=='__main__':main()
