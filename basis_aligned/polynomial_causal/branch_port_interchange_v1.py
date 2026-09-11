"""Split product donor interchange into common-reader, private-partner and mixed effects.
A product/pairwise/old-total replay. B common reader mean>=.001 for both branches
on both panels. C shared-reader paired bootstrap lower95>0 for all four cases.
"""
import hashlib
import json
from pathlib import Path
import torch


def roles(s, p, g):
    """Last two axes are documents and branch; any leading axes are preserved."""
    n = p.shape[-2]
    def covariance(a, b):
        return (a.mul(b).mean(-2)-a.mean(-2)*b.mean(-2))*n/(n-1)
    common = -covariance(s, p*g)
    private = -covariance(p, s*g)
    total = -covariance(s*p, g)
    return torch.stack((common, private, total-common-private, total), -1)


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    output = root/'BRANCH_PORT_INTERCHANGE_V1.json'
    assert not output.exists()
    scalar_path = root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_SCALARS.pt'
    source_path = root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    extraction = json.loads((root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_EXTRACTION.json').read_text())
    assert hashlib.sha256(scalar_path.read_bytes()).hexdigest()==extraction['scalar_sha256']
    assert extraction['pred_a']
    records = torch.load(scalar_path, weights_only=True, map_location='cpu')
    node = torch.load(source_path, weights_only=True, map_location='cpu')['nodes'][1]
    reference = json.loads((root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_RESULT.json').read_text())
    assert reference['pred_a']
    sources = [scalar_path, source_path]
    results = {}; product_errors = []; pair_errors = []; total_errors = []
    for label, suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        path = root/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt'
        panel_path = root/f'SHARED_NODE_PARENT1_{suffix}_ROWS.pt'
        sources += [path, panel_path]
        cache = torch.load(path, weights_only=True, map_location='cpu')
        panel = torch.load(panel_path, weights_only=True, map_location='cpu')
        x = cache['ports']['input'].double()
        record = records[label]
        assert record['documents']==panel['documents']
        s = (x @ node['reader'].double()).reshape(3,128,1)
        p = (x @ node['partners'].double()).reshape(3,128,2)
        g = record['sensitivities']
        product_errors.append(float((s*p-record['amplitudes']).norm()/record['amplitudes'].norm()))
        point = torch.zeros(2,4);boot = torch.zeros(2000,2,4);domains = {}
        generator = torch.Generator().manual_seed(5901)
        targets = panel['rows'][:,-1].reshape(3,128)
        conditional = torch.zeros(2,4); eligible = 0
        for domain in sorted(set(record['domain_labels'])):
            ids = torch.tensor([i for i, name in enumerate(record['domain_labels']) if name==domain])
            n = len(ids);ss,pp,gg = s[:,ids],p[:,ids],g[:,ids]
            part = roles(ss,pp,gg)
            ds = ss[:,None,:,:]-ss[:,:,None,:]
            dp = pp[:,None,:,:]-pp[:,:,None,:]
            changes = torch.stack((ds*pp[:,:,None,:],ss[:,:,None,:]*dp,ds*dp,
                ss[:,None,:,:]*pp[:,None,:,:]-ss[:,:,None,:]*pp[:,:,None,:]),-1)
            expected = (gg[:,:,None,:,None]*changes).sum((1,2))/(n*(n-1))
            pair_errors.append(float((expected-part).abs().max()))
            point += n/128*part.mean(0)
            chosen = torch.randint(n,(2000,n),generator=generator)
            boot += n/128*roles(ss[:,chosen],pp[:,chosen],gg[:,chosen]).mean(0)
            domains[domain] = dict(per_family=part.tolist(),mean=part.mean(0).tolist())
            for family in range(2):
                for token in sorted(set(targets[family,ids].tolist())):
                    matching = ids[targets[family,ids]==token]
                    if len(matching)<2:continue
                    conditional += len(matching)*roles(s[family,matching],p[family,matching],g[family,matching])
                    eligible += len(matching)
        ci = torch.quantile(boot,torch.tensor([.025,.975]),dim=0)
        total_errors.append(float((point[:,-1]-torch.tensor(reference['panels'][label]['mean_linear_effect'])).abs().max()))
        results[label] = dict(mean_linear_effect=point.tolist(),paired_document_95=ci.tolist(),
            pred_b=bool((point[:,0]>=.001).all()),pred_c=bool((ci[0,:,0]>0).all()),
            per_domain=domains,exact_token_conditioned_mean=(conditional/eligible).tolist(),
            exact_token_eligible_endpoints=eligible,exact_token_coverage=eligible/256)
        assert bool(torch.isfinite(point).all() and torch.isfinite(boot).all())
    a = max(product_errors)<=1e-9 and max(pair_errors)<=1e-12 and max(total_errors)<=1e-9
    result = dict(pred_a=a,pred_b=a and all(v['pred_b'] for v in results.values()),
        pred_c=a and all(v['pred_c'] for v in results.values()),
        intervention_order=['shared_reader','private_partner','mixed_term','complete_product'],
        product_errors=product_errors,pairwise_errors=pair_errors,old_total_errors=total_errors,panels=results,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        scope='First-order graph-port donor effects, native background retained. Shared-reader swap affects both consumers; '
              'branchwise attribution is descriptive. No global residual reader swap, full-dose port effects, new validation '
              'documents, learned factors, task semantics or circuit promotion.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','panels')},indent=2))
    print(json.dumps({k:{j:w for j,w in v.items() if j!='per_domain'} for k,v in results.items()},indent=2))


if __name__=='__main__': main()
