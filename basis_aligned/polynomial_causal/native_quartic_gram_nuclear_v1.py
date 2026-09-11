"""Frozen first2 centered output modes perstart; no text/outcome selection.
A coefficientfeasible<=1e-9/finite. B rank<=halfcanonical and truncationerror<=1e-4 EACH.
C all primal-dual gaps/residuals<=1e-6. 4000ADMMsteps max/tol1e-7.
"""
import json
from pathlib import Path
import torch
from quartic_gram_map_v1 import layout,canonical,coefficients
from quartic_gram_nuclear_v1 import solve
from sparse_path_stability_atlas_v1 import digest


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);p=Path(__file__).parent
    out=p/'NATIVE_QUARTIC_GRAM_NUCLEAR_V1.json';ap=p/'NATIVE_QUARTIC_GRAM_NUCLEAR_V1.pt';assert not out.exists() and not ap.exists()
    control=json.loads((p/'QUARTIC_GRAM_NUCLEAR_V1_CONTROL.json').read_text());assert control['pred_a'] and control['pred_b']
    receipt=p/'SPARSE_QUARTIC_CEILING_V1_RESULT.json';prior=json.loads(receipt.read_text());assert prior['pred_a']
    source=p/'SPARSE_QUARTIC_CEILING_V1_MODES.pt';assert digest(source)==prior['artifact_sha256']
    modes=[m for m in torch.load(source,weights_only=True,map_location='cpu')['modes'] if m['metric']=='centered'];assert len(modes)==2
    mapping=layout(16);reports=[];saved=[]
    for m in modes:
        assert torch.equal(mapping['terms'],m['terms'])
        for index in [0,1]:
            c=m['scalar_coefficients'][index];base=canonical(c,mapping);be=torch.linalg.eigvalsh(base)
            base_rank=int((be.abs()>1e-6*be.abs().max()).sum())
            matrix,r=solve(c,mapping,iterations=4000,tolerance=1e-7)
            eig,vec=torch.linalg.eigh(matrix);keep=eig.abs()>1e-6*eig.abs().max()
            direct_error=float((coefficients(matrix,mapping)-c).norm()/c.norm())
            r.update(seed=m['seed'],mode=index,canonical_effective_rank=base_rank,canonical_nuclear_norm=float(be.abs().sum()),
                nuclear_ratio=r['nuclear_norm']/float(be.abs().sum()),direct_coefficient_error=direct_error,
                rank_half_and_accurate=r['effective_rank']<=base_rank/2 and r['truncated_coefficient_error']<=1e-4)
            reports.append(r);saved.append(dict(seed=m['seed'],mode=index,bank=m['bank'],output_writer=m['physical_writer'][:,index],
                gram=matrix,quadratic_coefficients=vec[:,keep],signed_weights=eig[keep],pairs=mapping['pairs']))
            print(json.dumps({k:v for k,v in r.items() if k!='history'}),flush=True)
    torch.save(dict(programs=saved),ap)
    result=dict(pred_a=all(r['direct_coefficient_error']<=1e-9 for r in reports) and all(torch.isfinite(s['gram']).all().item() for s in saved),
        pred_b=all(r['rank_half_and_accurate'] for r in reports),
        pred_c=all(-1e-10<=r['relative_gap']<=1e-6 and r['primal_residual']<=1e-6 and r['dual_residual']<=1e-6 for r in reports),
        reports=reports,source_receipt_sha256=digest(receipt),source_artifact_sha256=digest(source),artifact_sha256=digest(ap),script_sha256=digest(__file__),
        scope='Convex signed Gram nuclear surrogate on4weight-selected projected output modes. Not minimumrank, nativeglobalcapture, sharedmulti-outputdictionary or behavioral circuits.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2));assert result['pred_a']


if __name__=='__main__':main()
