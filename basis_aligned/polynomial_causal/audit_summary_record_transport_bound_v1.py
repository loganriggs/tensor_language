"""Fixed record reorderings; FP64 evaluation of an analytic transport bound."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import query_initializer_factorization_reference as Q
import summary_record_transport_reference as S
import field_intervention_metrics as M

BASE=Path(__file__).resolve().parent;ROWS=BASE/'QUERY_INITIALIZER_FACTORIZATION_V1_ROWS.pt';OUT=BASE/'SUMMARY_RECORD_TRANSPORT_BOUND_V1.json'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    checks=S.controls();assert checks['passed'];data=torch.load(ROWS,map_location='cpu',weights_only=True)['populations']
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));audits=[]
    permutations={'swap_first_two':torch.arange(24),'cyclic_shift_one':torch.arange(24).roll(-1)};permutations['swap_first_two'][[0,1]]=torch.tensor([1,0])
    with torch.inference_mode():
        a,offset,c=S.operator(program);targets={name:S.permuted_operator(a,c,p) for name,p in permutations.items()}
        for pop,block in data.items():
            for world in range(8):
                binding=block['native_tokens'][world*96,:48];x=S.coordinates(binding,c)
                predicted=(offset+a@x).reshape(4,128);audits.append(M.correspondence(predicted,block['document_summaries'][world]))
                for name,p in permutations.items():
                    changed=binding.reshape(24,2)[p].flatten()
                    oracle=Q.document_summary(program,changed[None].expand(4,-1),torch.arange(25,29))
                    audits.append(M.correspondence((offset+targets[name]@x).reshape(4,128),oracle))
        u,s,vh=torch.linalg.svd(a,full_matrices=False)
        bounds={name:S.transport_bound(a,b,384,1e6,vh) for name,b in targets.items()}
    result={'scope':'norm-bounded affine summary transport on affinehull of allbijective key/value assignments; not cohortKL, nonlineartransport or fullmodel impossibility',
        'controls':checks,'mechanical_passed':all(t['passed'] for t in audits),'operator_native_max_abs':max(t['max_abs'] for t in audits),
        'operator_native_max_relative_rms':max(t['relative_rms'] for t in audits),'operator_shape':list(a.shape),'bounds':bounds,
        'singular_values':s.tolist(),'spectrum_note':'no small singularvalue declared exactzero; fixed384 projection motivates bound but inequality holds for any projection rows',
        'numeric_status':'FP64 evaluation of analytic inequality, not interval-arithmetic certification','independent_constants':program.independent_constant_count(),
        'operator_values':a.numel(),'offset_values':offset.numel(),'native_coefficients_removed':0,
        'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert result['mechanical_passed'];print(json.dumps({k:v for k,v in result.items() if k!='singular_values'},indent=2))


if __name__=='__main__':main()
