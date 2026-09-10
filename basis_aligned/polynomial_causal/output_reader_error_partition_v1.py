"""Exact answer-reader/complement norm partition from immutable phase receipts."""
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def partition(full_norm,margin_norm):
    # w=e_answer-e_foil has squared norm2, and lies in centered vocabulary space.
    parallel_squared=margin_norm**2/2
    full_squared=full_norm**2
    if parallel_squared>full_squared+1e-9*max(full_squared,1):
        raise ValueError('Inconsistent full-vector/answer-margin norms')
    orthogonal_squared=max(0.,full_squared-parallel_squared)
    return {'answer_axis_norm':math.sqrt(parallel_squared),
            'orthogonal_norm':math.sqrt(orthogonal_squared),
            'answer_axis_squared_norm_fraction':parallel_squared/max(full_squared,1e-30),
            'orthogonal_squared_norm_fraction':orthogonal_squared/max(full_squared,1e-30)}


def main():
    path=ROOT/'ABOUT_FOR_QUERY_PHASE_V1_RESULT.json';source=json.loads(path.read_text())
    assert source['predictions']['pred_a_instrument']
    reports={}
    for panel,r in source['reports'].items():
        reports[panel]={
            'reference_complete_head_effect':partition(r['reference_full_norm'],r['reference_margin_norm']),
            'phase_axis_approximation_error':partition(r['full_vector_error']*r['reference_full_norm'],
                                                     r['margin_error']*r['reference_margin_norm'])}
    # Independent centered four-token fixture, w=(1,-1,0,0).
    vector=(3.,-1.,2.,-4.);margin=vector[0]-vector[1]
    p=partition(math.sqrt(sum(v*v for v in vector)),margin)
    projected=(margin/2,-margin/2,0.,0.)
    residual=tuple(v-a for v,a in zip(vector,projected))
    error=abs(p['orthogonal_norm']-math.sqrt(sum(v*v for v in residual)))
    assert error<1e-12
    out={'schema':1,'source':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'reports':reports,'coordinate_control_error':error,'model_forwards':0,
         'scope':'Exact Euclidean contrast/complement geometry; no semantic label or selectivity of the complement established.'}
    with (ROOT/'OUTPUT_READER_ERROR_PARTITION_V1_RESULT.json').open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    for panel,r in reports.items():
        print(panel,'error outside answer axis',r['phase_axis_approximation_error']['orthogonal_squared_norm_fraction'])


if __name__=='__main__':main()
