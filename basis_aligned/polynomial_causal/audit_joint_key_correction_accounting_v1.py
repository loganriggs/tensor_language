"""Saved-write factorial accounting of numerator/denominator compensation."""
import hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    path=P/'REGIONAL_KEY_CORRECTION_V1_ARTIFACT.pt'
    w=torch.load(path,map_location='cpu',weights_only=True)['write_vertices'].double()
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows']
    # arms:0base,1native,2retained,3denominator,4numerator,5mixed,6complete.
    denominator=w[:,3]-w[:,2];numerator=w[:,4]-w[:,2]
    interaction=w[:,6]-w[:,4]-w[:,3]+w[:,2]
    required=w[:,6]-w[:,2];omitted_quadratic=w[:,6]-w[:,5]
    replay=float((required-denominator-numerator-interaction).norm()/required.norm())
    assert replay<=1e-12
    cells=[]
    for family in range(4):
        ids=[i for i,r in enumerate(rows) if r['family']==family];target=required[ids].flatten()
        terms={}
        for name,value in [('denominator',denominator),('numerator',numerator),('interaction',interaction),('omitted_quadratic',omitted_quadratic)]:
            v=value[ids].flatten()
            terms[name]=dict(norm_over_required=float(v.norm()/target.norm()),signed_projection_over_required=float((v@target)/(target@target)))
        cells.append(dict(family=family,terms=terms))
    result=dict(relative_identity_error=replay,cells=cells,artifact_sha=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exact conditional write factorial accounting. Norm ratios are not additive variance shares; signed projections can cancel. No intervention independence or native generator simplification is proved.')
    (P/'JOINT_KEY_CORRECTION_ACCOUNTING_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
