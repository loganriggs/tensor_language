"""Explain change/replacement split without fitting an offset.
A exact pair energy split <=1e-9. B physical difference-error energy fraction
<=.01 in each A1/A2 direction. C aggregate base/donor error cosine>=.9 eachcell.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;out=root/'NATIVE_RELATION_PAIR_REMAINDER_V1.json'
    assert not out.exists()
    paths=[root/'NATIVE_TOKEN_RELATION_FOLD_V1.pt',root/'FROZEN_BRANCH_MORPHOLOGY_V1_ENDPOINTS.pt',
           root/'FROZEN_BRANCH_MORPHOLOGY_V1_ROWS.json',root/'NATIVE_RELATION_PHYSICAL_V1_RESULT.json']
    physical=json.loads(paths[-1].read_text());assert physical['pred_a']
    binding=json.loads((root/'NATIVE_RELATION_PHYSICAL_V1_BINDING.json').read_text())['files']
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==binding[str(p)] for p in paths[:3])
    saved=torch.load(paths[0],weights_only=True,map_location='cpu')
    cache=torch.load(paths[1],weights_only=True,map_location='cpu');rows=json.loads(paths[2].read_text())['rows']
    r=saved['physical_readouts'][:3];writers=torch.linalg.solve(r@r.T,r).T
    metric=writers.T@writers;x=cache['ports']['input'].double()
    exact=cache['ports']['native_output'].double()@r.T
    approx=[]
    for j,c in enumerate(saved['compact'][:3]):
        approx.append((x@c['top16_square_readers'].T).square()@c['top16_square_coefficients']+
                      c['radial']*x.square().sum(1)+saved['folded_bias'][j])
    error=exact-torch.stack(approx,1)
    def inner(a,b):return torch.einsum('ni,ij,nj->',a,metric,b)
    cells=[];checks=[]
    for family in ('A1','A2'):
        for direction in ('base_to_suffix','suffix_to_base'):
            ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family and row['direction']==direction])
            eb,ed=error[2*ids],error[2*ids+1]
            common,difference=(eb+ed)/2,(ed-eb)/2
            ec,ee=inner(common,common),inner(difference,difference)
            total=inner(eb,eb)+inner(ed,ed)
            checks.append(float(abs(total-2*(ec+ee))/total))
            cosine=float(inner(eb,ed)/(inner(eb,eb)*inner(ed,ed)).sqrt())
            cells.append(dict(family=family,direction=direction,n=len(ids),
                difference_fraction=float(ee/(ec+ee)),base_donor_error_cosine=cosine,
                common_error_rms=float((ec/len(ids)).sqrt()),difference_error_rms=float((ee/len(ids)).sqrt())))
    a=max(checks)<=1e-9
    result=dict(pred_a=a,pred_b=a and all(r['difference_fraction']<=.01 for r in cells),
        pred_c=a and all(r['base_donor_error_cosine']>=.9 for r in cells),
        maximum_energy_identity_error=max(checks),cells=cells,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Paired error accounting in physical dual-writer norm on existing validation rows. '
        'No empirical offset inserted, no factors refitted, no repair of absolute replacement failure. '
        'Shared pair error does not establish a globally invariant feature, OOD robustness or native semantic hierarchy.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
