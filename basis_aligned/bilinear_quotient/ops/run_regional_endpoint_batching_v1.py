#!/usr/bin/env python3
# BQGATE:160bodyforwards;8prefixes;300seconds;no fitting.
"""pred_a exact CPU packing; pred_b raw <=1e-5 abs/1e-6 relative.
pred_c effect replay<=1e-3 and160forwards. No scientific threshold change.
"""
import hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_typed_face_runtime import measure_lattice
from regional_endpoint_batching_v1 import group_rows,expand
from run_even_value_factorial_native_v1 import setup
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
STEM='REGIONAL_ENDPOINT_BATCHING_V1'
spec=importlib.util.spec_from_file_location('compiled_face',D/'native.py')
compiled=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiled)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_ROWS.json').read_text())['rows']
    old=torch.load(P/'ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    groups,mapping=group_rows(rows);assert len(groups)==8
    packed=torch.zeros(24,8,10,dtype=old.dtype)
    for i,(g,e) in enumerate(mapping):
        packed[:,g,e]=old[:,i,0];packed[:,g,6:]=old[:,i,1:]
    cpu_error=float((expand(packed,mapping,6)-old).abs().max());assert cpu_error==0
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print('160bodyforwards;8prefixes;CPU packing exact;original48rows restored');return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    started=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda')
    program={k:v.to('cuda') for k,v in torch.load(D/'program.pt',weights_only=True).items()}
    measured=measure_lattice(model,graph,[],groups,compiled,program)
    new=expand(measured['values'],mapping,6)
    delta=new-old;raw_abs=float(delta.abs().max());raw_rel=float(delta.norm()/old.norm())
    effects=new-new[0:1];old_effects=old-old[0:1]
    arms=[5,6,7,*range(8,24)]
    effect_errors=((effects[arms]-old_effects[arms]).norm(dim=1)/old_effects[arms].norm(dim=1).clamp_min(1e-8))
    result={'pred_a':cpu_error==0,'pred_b':raw_abs<=1e-5 and raw_rel<=1e-6,'pred_c':float(effect_errors.max())<=1e-3 and measured['body_forwards']==160,
      'max_abs_margin_error':raw_abs,'relative_margin_error':raw_rel,'max_relative_effect_error':float(effect_errors.max()),
      'body_forwards':measured['body_forwards'],'original_forwards':960,'seconds':time.perf_counter()-started,
      'original_seconds':16.292272353777662,'groups':8,'restored_rows':48,'source_shas':binding,'panel_status':'replay',
      'scope':'Execution batching equivalence only. No change to scientific claims, gates, interventions or input boundaries.'}
    torch.save({'values':new,'packed_values':measured['values'],'representatives':measured['representatives']},P/(STEM+'_ARTIFACT.pt'))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
