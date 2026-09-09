"""Integer token-incidence certificate and saved additive-response bound."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal';OPS=ROOT/'basis_aligned/bilinear_quotient/ops'
sys.path.insert(0,str(OPS))
import circuit_candidate_temporal_iswas_dual_command_v2 as C
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def contrast(tokens):
    out=Counter()
    for cell,sign in (('00',1),('11',1),('01',-1),('10',-1)):out[tokens[cell]]+=sign
    return {str(k):v for k,v in sorted(out.items()) if v}

def main():
    signal.alarm(30);started=time.perf_counter();out=BASE/'BILIN18_SHARED_VALUE_PRODUCER_INCIDENCE_V1_RESULT.json';assert not out.exists()
    builder=Path(C.__file__);assert digest(builder)=='72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0'
    controls={'independent_token_positive':not contrast({'00':0,'01':0,'10':1,'11':1}),
              'constant_token_positive':not contrast(dict.fromkeys(('00','01','10','11'),3)),
              'xor_negative':contrast({'00':0,'01':1,'10':1,'11':0})=={'0':2,'1':-2}}
    assert all(controls.values());rows=C.build_rows();assert len(rows)==32
    positions=0;nonzero=[]
    for row in rows:
        cells=row['endpoints'];lengths={len(e['ids']) for e in cells.values()};assert len(lengths)==1
        for pos in range(next(iter(lengths))):
            positions+=1;delta=contrast({ab:e['ids'][pos] for ab,e in cells.items()})
            if delta:nonzero.append({'row_id':row['row_id'],'position':pos,'contrast':delta})
    source=BASE/'BILIN18_SHARED_RESPONSE_COMMAND_MODES_V1_RESULT.json';modes=json.loads(source.read_text())
    assert modes['predictions']['pred_a_instrument'];bounds={}
    for key,summary in modes['summaries'].items():
        if not key.endswith('/ALL'):continue
        bounds[key]={label:p['mode_rms']['11']/max(p['effect_rms'],1e-6) for label,p in summary['panels'].items()}
    result={'scope':'All opened token incidences; exact producer property for every token-only map, conditional on the inspected native architecture.',
        'predictions':{'pred_a_instrument':all(controls.values()),'pred_b_zero_producer_mixed_mode':not nonzero},
        'worlds':len(rows),'aligned_source_positions':positions,'nonzero_incidence_positions':nonzero,'controls':controls,
        'additive_command_response_relative_lower_bounds':bounds,
        'consequence':'Any observed mixed shared-removal response is generated downstream of the token-only producer.' if not nonzero else 'Producer separability not established.',
        'source_sha256':digest(source),'builder_sha256':digest(builder),'runner_sha256':digest(Path(__file__)),
        'prereg_sha256':digest(BASE/'BILIN18_SHARED_VALUE_PRODUCER_INCIDENCE_V1_PREREGISTRATION.md'),
        'native_parameters_retained':545902902,'wall_seconds':time.perf_counter()-started}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
