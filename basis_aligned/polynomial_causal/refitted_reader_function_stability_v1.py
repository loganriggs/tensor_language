"""Whole-function stability after exact Down refits of converged oblique bases."""
import hashlib
import json
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P,CK
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);started=time.perf_counter()
    report=json.loads((P/'READER_CONDITIONAL_WRITER_V1_RESULT.json').read_text())
    rows=[a for a in report['arms'] if a['family']=='ordinary_covariance']
    assert len(rows)==2 and all(a['parent_converged'] for a in rows)
    m=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram']
    root=torch.linalg.cholesky((m+m.T)/2).T
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    native=(l,r,root@d);functions=[];sources=[]
    for row in rows:
        for receipt in (row['source'],row['cache']):
            assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
        saved=torch.load(row['source']['path'],weights_only=True,map_location='cpu')
        w=torch.load(row['cache']['path'],weights_only=True,map_location='cpu')['down']
        code=torch.zeros(9216,1152).scatter_(1,saved['code_indices'].long(),saved['code_values'])
        a,b=(code@saved['analysis_basis']).split(4608)
        functions.append((a,b,root@w));sources.append(dict(readers=row['source'],writer=row['cache']))
    total=json.loads((P/'FULLU_TRACE_METRIC_V1_AUDIT.json').read_text())['full_coefficient_metric']['total_energy']
    energies=[float(inner(f,f)) for f in functions]
    captures=[1-(energy-2*float(inner(native,f))+total)/total for f,energy in zip(functions,energies)]
    replay=[abs(value-row['after_capture']) for value,row in zip(captures,rows)]
    projection=[abs(value-energy/total) for value,energy in zip(captures,energies)]
    cross=float(inner(*functions));cosine=cross/(energies[0]*energies[1])**.5
    result=dict(predictions=dict(pred_a_instrument=max(replay+projection)<=1e-8 and min(energies)>0,
                                pred_b_function_stability=cosine>=.9),
        captures=captures,capture_replay_errors=replay,projection_identity_errors=projection,
        function_cosine=cosine,squared_function_difference_over_native=(sum(energies)-2*cross)/total,
        source=sources,seconds=time.perf_counter()-started,
        scope='Two locally converged reader dictionaries with exact conditional Down, whole coefficient-function stability; not raw coordinate alignment, text validation or circuit identification.')
    with (P/'REFITTED_READER_FUNCTION_STABILITY_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
