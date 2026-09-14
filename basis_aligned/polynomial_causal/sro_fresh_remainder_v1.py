"""Post-failure attribution of omitted R effects; no fitting or row selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,torch
from even_value_factorial_v1 import reconstruct


def main():
    p=Path(__file__).resolve().parent
    out=p/'SRO_FRESH_REMAINDER_V1_RESULT.json';assert not out.exists()
    source=p/'SRO_FRESH_CUE_V1_ARTIFACT.pt'
    a=torch.load(source,weights_only=True);cube=a['cube'];c=a['coefficients']
    rows=json.loads((p/'SRO_FRESH_CUE_V1_ROWS.json').read_text())['rows']
    reduced=torch.zeros_like(c);reduced[[1,2,4,5]]=c[[1,2,4,5]]
    prediction=reconstruct(reduced);effect=cube-cube[0]
    cells=[]
    for family in range(3):
        ix=[i for i,r in enumerate(rows) if r['family']==family]
        paired=effect[:,ix[::2],0]-effect[:,ix[1::2],0]
        pred=prediction[:,ix[::2],0]-prediction[:,ix[1::2],0]
        scale=paired[7].norm().clamp_min(1e-8)
        errors=(pred-paired).norm(dim=1)/scale
        pc=c[:,ix[::2],0]-c[:,ix[1::2],0]
        pair_errors=c[7,ix].norm(dim=0)/effect[7,ix].norm(dim=0).clamp_min(1e-8)
        omitted=pc[[2,3,6,7]].sum(0)
        projection=[float((pc[k]*omitted).sum()/omitted.square().sum().clamp_min(1e-16)) for k in [2,3,6,7]]
        cells.append(dict(family=family,pair_only_full_removal_errors=pair_errors.tolist(),
                          restored_R_cue_errors=errors.tolist(),
                          omitted_coefficient_norms_over_full=[float(pc[k].norm()/scale) for k in [2,3,6,7]],
                          signed_projection_on_omitted_full_effect=projection))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=all(max(r['pair_only_full_removal_errors'])<=.01 for r in cells),
                pred_b=all(max(r['restored_R_cue_errors'])<=.1 for r in cells),
                coefficient_order=['R','SR','RO','SRO'],families=cells,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope='Post-failure explanation on measured new-panel responses, not independent validation of a repaired reduction. Signed projections may exceed one or be negative; original failure and native capability miss remain.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
