"""Nonself assignment controls common-reader donor swaps by private-partner distance.
A metric/permutation identities <=1e-9. B private cost <=.25 and common s
perturbation >=.10 of independent-donor expectations, each panel.
C absolute stratum mean product shift <=.25 of independent baseline, all cases.
No factor fitting; first-order effects descriptive until adequate finite replay.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;output=root/'BRANCH_PRIVATE_MATCHED_DONORS_V1.json'
    assert not output.exists()
    source=root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    scalars=root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_SCALARS.pt'
    record=torch.load(scalars,weights_only=True,map_location='cpu')
    node=torch.load(source,weights_only=True,map_location='cpu')['nodes'][1]
    partners=node['partners'].double()
    gram=partners.T@partners;chol=torch.linalg.cholesky(gram)
    dual=partners@torch.linalg.inv(gram)
    sources=[source,scalars];errors=[];results={};saved={}
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        path=root/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt';sources.append(path)
        x=torch.load(path,weights_only=True,map_location='cpu')['ports']['input'].double()
        s=x@node['reader'].double();p=x@partners
        z=torch.linalg.solve_triangular(chol,p.T,upper=False).T
        domains=record[label]['domain_labels'];g=record[label]['sensitivities'].reshape(384,2)
        assert float(((s[:,None]*p).reshape(3,128,2)-record[label]['amplitudes']).norm())==0
        donor=torch.empty(384,dtype=torch.long);details=[]
        private_cost=0.;private_ref=0.;shared_cost=0.;shared_ref=0.
        mean_shift=torch.zeros(2);mean_shift_ref=torch.zeros(2);effect=torch.zeros(2)
        for family in range(3):
            for domain in sorted(set(domains)):
                ids=torch.tensor([family*128+i for i,name in enumerate(domains) if name==domain])
                n=len(ids);zz=z[ids];ss=s[ids];pp=p[ids]
                costs=(zz[:,None]-zz[None,:]).square().sum(-1)
                cost_array=costs.numpy().copy();np.fill_diagonal(cost_array,np.inf)
                rows,columns=linear_sum_assignment(cost_array)
                assert np.array_equal(rows,np.arange(n))
                selected=torch.from_numpy(columns)
                assert bool((selected!=torch.arange(n)).all())
                assert torch.equal(selected.sort().values,torch.arange(n))
                donor[ids]=ids[selected]
                dp=pp[selected]-pp;ds=ss[selected]-ss
                selected_cost=costs[torch.arange(n),selected]
                independent_private=costs.sum()/(n*(n-1))
                independent_shared=(ss[:,None]-ss[None,:]).square().sum()/(n*(n-1))
                projected_shift=dp@dual.T
                errors.append(float((projected_shift.square().sum(1)-selected_cost).abs().max()))
                delta=ds[:,None]*pp
                expected_mean=-(ss[:,None]*pp).mean(0)+ss.mean()*pp.mean(0)
                expected_mean*=n/(n-1)
                weight=n/384
                private_cost+=weight*float(selected_cost.mean());private_ref+=weight*float(independent_private)
                shared_cost+=weight*float(ds.square().mean());shared_ref+=weight*float(independent_shared)
                mean_shift+=weight*delta.mean(0).abs();mean_shift_ref+=weight*expected_mean.abs()
                local_effect=(delta*g[ids]).mean(0);effect+=weight*local_effect
                details.append(dict(family=family,domain=domain,documents=n,
                    private_energy_ratio=float(selected_cost.mean()/independent_private),
                    common_energy_ratio=float(ds.square().mean()/independent_shared),
                    mean_product_shift=delta.mean(0).tolist(),independent_mean_shift=expected_mean.tolist(),
                    mean_linear_effect=local_effect.tolist()))
        assert torch.equal(donor.sort().values,torch.arange(384)) and bool((donor!=torch.arange(384)).all())
        rp,rs=private_cost/private_ref,shared_cost/shared_ref
        rm=mean_shift/mean_shift_ref
        results[label]=dict(private_distance_energy_ratio=rp,common_perturbation_energy_ratio=rs,
            absolute_mean_shift_ratio=rm.tolist(),mean_linear_effect=effect.tolist(),
            pred_b=rp<=.25 and rs>=.1,pred_c=bool((rm<=.25).all()),strata=details)
        saved[label]=dict(donors=donor,common_values=s,private_values=p,
            documents=record[label]['documents'],domain_labels=domains)
    a=max(errors)<=1e-9
    artifact=root/'BRANCH_PRIVATE_MATCHED_DONORS_V1.pt';assert not artifact.exists()
    torch.save(saved,artifact)
    result=dict(pred_a=a,pred_b=a and all(r['pred_b'] for r in results.values()),
        pred_c=a and all(r['pred_c'] for r in results.values()),metric_identity_errors=errors,
        panels=results,partner_gram_condition=float(torch.linalg.cond(gram)),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        scope='Minimum-cost private-matched donor permutation within family/domain. Validation row matching only; '
              'not conditional-distribution sampling, factor fitting, finite-dose effect or circuit identification.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','sources','metric_identity_errors')},indent=2))
    print(json.dumps({k:{j:v for j,v in r.items() if j!='strata'} for k,r in results.items()},indent=2))


if __name__=='__main__':main()
