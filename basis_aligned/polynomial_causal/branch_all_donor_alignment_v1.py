"""Exact donor-averaged linear effect and paired document bootstrap.

Predictions are in BRANCH_ALL_DONOR_ALIGNMENT_V1_PREREGISTRATION.md.
No model fitting; CPU analysis of the frozen scalar extraction.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent
    stem='BRANCH_ALL_DONOR_ALIGNMENT_V1'
    output=root/(stem+'_RESULT.json')
    assert not output.exists()
    path=root/(stem+'_SCALARS.pt')
    source=json.loads((root/(stem+'_EXTRACTION.json')).read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['scalar_sha256']
    assert source['pred_a']
    data=torch.load(path,weights_only=True,map_location='cpu')
    results={};checks=[]
    for label,record in data.items():
        a,g=record['amplitudes'],record['sensitivities']
        assert a.shape==g.shape==(3,128,2)
        generator=torch.Generator().manual_seed(5801)
        point=torch.zeros(2);boot=torch.zeros(2000,2);details={}
        for domain in sorted(set(record['domain_labels'])):
            ids=torch.tensor([i for i,name in enumerate(record['domain_labels']) if name==domain])
            n=len(ids);weight=n/128
            aa,gg=a[:,ids],g[:,ids]
            covariance=-(aa.mul(gg).mean(1)-aa.mean(1)*gg.mean(1))*n/(n-1)
            # Diagonal terms are identically zero; denominator excludes self donors.
            pairs=(gg[:,:,None,:]*(aa[:,None,:,:]-aa[:,:,None,:])).sum((1,2))/(n*(n-1))
            checks.append(float((pairs-covariance).abs().max()))
            sampled=torch.randint(n,(2000,n),generator=generator)
            ab,gb=aa[:,sampled],gg[:,sampled]
            covboot=-(ab.mul(gb).mean(2)-ab.mean(2)*gb.mean(2))*n/(n-1)
            point+=weight*covariance.mean(0)
            boot+=weight*covboot.mean(0)
            details[domain]=dict(documents=n,per_family_linear_effect=covariance.tolist(),
                mean_linear_effect=covariance.mean(0).tolist())
        ci=torch.quantile(boot,torch.tensor([.025,.975]),dim=0)
        assert bool(torch.isfinite(boot).all())
        results[label]=dict(mean_linear_effect=point.tolist(),paired_document_95=ci.tolist(),
            pred_b=bool((point>=.001).all()),pred_c=bool((ci[0]>0).all()),domains=details)
    a=max(checks)<=1e-12 and source['pred_a']
    result=dict(pred_a=a,pred_b=a and all(v['pred_b'] for v in results.values()),
        pred_c=a and all(v['pred_c'] for v in results.values()),
        all_pairs_identity_errors=checks,panels=results,
        source_sha256=source['scalar_sha256'],
        scope='Exact expected first-order interchange over all nonself donors within family/domain. '
              'Paired document bootstrap recomputes covariances; these previously inspected panels are not fresh confirmation. '
              'No full-dose all-donor result, learned circuit factors, semantic task gate, or extraction claim.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
