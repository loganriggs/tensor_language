"""Frozen unequal-duration cross-start comparison, not final convergence evidence.

A loss replay <=1e-8 and valid Gram bounds; B function cosine >=.9;
C native-residual cosine >=.9. Uses full weights and no text.
"""
import hashlib
import io
import json
import time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P, CK
from cancellation_group_v1_audit import load
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    out=P/'INTERMEDIATE_FUNCTION_STABILITY_V1_AUDIT.json'; assert not out.exists()
    start=time.perf_counter()
    parent=json.loads((P/'PROJECTED_SPARSE_DICTIONARY_FIT_V1_SEED_0.json').read_text())
    first=parent['cache']
    content=Path('/dev/shm/bilin18_projected_sparse_dictionary_fit_v1_s937.pt').read_bytes()
    frozen=Path('/dev/shm/bilin18_intermediate_function_stability_v1_s937.pt')
    with frozen.open('xb') as f:f.write(content)
    saved=torch.load(io.BytesIO(content),weights_only=True,map_location='cpu')
    second=dict(path=str(frozen),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),
                iteration=saved['history'][-1]['iteration'],fit_seconds=saved['history'][-1]['seconds'],loss=saved['loss'])
    del content
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T
    functions=[load(first,wh),load(second,wh)]
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    native=(l,r,wh@d)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    energies=[float(inner(f,f)/total) for f in functions]
    crosses=[float(inner(native,f)/total) for f in functions]
    cross=float(inner(*functions)/total)
    losses=[1+e-2*c for e,c in zip(energies,crosses)]
    residual_cross=1-sum(crosses)+cross
    cosine=cross/(energies[0]*energies[1])**.5
    residual_cosine=residual_cross/(losses[0]*losses[1])**.5
    drift=sum(energies)-2*cross
    replay=max(abs(losses[0]-parent['optimization']['final_loss']),abs(losses[1]-saved['loss']))
    result=dict(predictions=dict(pred_a_instrument=replay<=1e-8 and drift>=-1e-10 and
        abs(cosine)<=1+1e-8 and abs(residual_cosine)<=1+1e-8,
        pred_b_function_stability=cosine>=.9,pred_c_shared_residual=residual_cosine>=.9),
        captures=[1-v for v in losses],fitted_energies=energies,native_crosses=crosses,
        function_cross=cross,function_cosine=cosine,residual_cosine=residual_cosine,
        squared_function_difference=drift,loss_replay=replay,sources=[first,second],
        seconds=time.perf_counter()-start,
        scope='Independent starts, unequal durations, seed937 intermediate. No final or individual-unit stability verdict.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
