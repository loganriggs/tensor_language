#!/usr/bin/env python3
# BQGATE:288bodyforwards;96prefixes;300seconds;no fitting.
"""pred_a compiled write/token table <=1e-5; pred_b recursive effect <=1e-4.
pred_c parent absolute replay <=1e-4. Null: native rounding/port mismatch.
See frozen preregistration for literal price, scope and missing causal gates.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from odd_attention8h2_typed_face_native_runtime_v1 import measure_lattice
from run_even_value_factorial_native_v1 import setup
STEM='ODD_ATTENTION8H2_TYPED_FACE_NATIVE_V1'
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
spec=importlib.util.spec_from_file_location('compiled_face',D/'native.py')
compiled=importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiled)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1_ROWS.json').read_text())['rows']
    token_ids=sorted({r['ids'][r['city_position']] for r in rows})
    assert len(rows)==96 and len(token_ids)==8
    assert all(rows[i]['city_position']==rows[i^1]['city_position'] for i in range(96))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        control=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_V1_CPU_CONTROL.json').read_text())
        assert control['pred_a']
        try:
            compiled.inherited({'token_ids':torch.tensor(token_ids)},-1)
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown token accepted')
        print('288bodyforwards;96prefixes;8tokens;frozen rows and algebra control pass')
        return
    output=P/(STEM+'_RESULT.json')
    assert not output.exists() and not (D/'program.pt').exists()
    started=time.perf_counter();signal.alarm(300)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval()
    attn=model.transformer.h[8].attn
    p={name:getattr(attn,attr).weight[256:384].detach().clone() for name,attr in
       [('q1','c_q'),('q2','c_q2'),('k1','c_k'),('k2','c_k2'),('current_value','c_v')]}
    p['output']=attn.c_proj.weight[:,256:384].detach().clone()
    p['mixture']=attn.lamb.detach().clone()
    p['token_ids']=torch.tensor(token_ids,device='cuda')
    initial=F.rms_norm(model.transformer.wte(p['token_ids'][None]),(1152,))
    b0=model.transformer.h[0]
    first_input=F.rms_norm((b0.lambdas[0]+b0.lambdas[1])*initial,(1152,))
    p['first_table']=b0.attn.c_v(first_input)[0,:,256:384].detach().clone()
    graph,_,_,_=setup('cuda')
    measured=measure_lattice(model,graph,[],rows,compiled,p)
    values=measured['values']
    reference=values[5]-values[0];candidate=values[6]-values[0]
    effect_errors=((candidate-reference).norm(dim=0)/reference.norm(dim=0).clamp_min(1e-8)).tolist()
    parent=torch.load(P/'ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values'][:,48:]
    parent_error=float((values[[0,5]]-parent[[0,5]]).abs().max())
    pred_a=max(measured['compiled_errors'])<=1e-5 and max(measured['table_errors'])<=1e-5 and max(measured['outside'])==0
    pred_b=max(effect_errors)<=1e-4
    pred_c=parent_error<=1e-4
    cpu={k:v.detach().cpu().clone() for k,v in p.items()}
    torch.save(cpu,D/'program.pt')
    torch.save({'values':values,'example_current_inputs':'not saved; native states remain external'},P/(STEM+'_ARTIFACT.pt'))
    result={'pred_a':pred_a,'pred_b':pred_b,'pred_c':pred_c,
      'terminal':'native_closure_pass' if pred_a and pred_b and pred_c else 'native_closure_failed_gate',
      'max_write_relative_error':max(measured['compiled_errors']),
      'max_token_table_relative_error':max(measured['table_errors']),
      'recursive_effect_errors_by_readout':effect_errors,'parent_max_abs_margin_error':parent_error,
      'body_forwards':measured['body_forwards'],'seconds':time.perf_counter()-started,
      'static_scalars':sum(v.numel() for v in cpu.values()),'program_bytes':(D/'program.pt').stat().st_size,
      'program_sha256':digest(D/'program.pt'),'source_shas':binding,
      'declared_native_state_inputs':2,'city_token_vocabulary':token_ids,
      'panel_status':'opened replay','scope':'Head8.2 write only; head9 odd-value graph, reentry and full suffix external. No new OOD, removal or composition claim.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2))
    signal.alarm(0)

if __name__=='__main__':
    main()
