"""Post-result exact-token control for all-donor linear alignment.
A pair/covariance replay <=1e-12 and source checks. B >=80% eligible target-family
endpoints each panel. C conditional linear means >=.001 both branches/panels.
Only cells with >=2 distinct documents support nonself donor interchange.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent
    output=root/'BRANCH_EXACT_TOKEN_ALIGNMENT_V1.json'
    assert not output.exists()
    path=root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_SCALARS.pt'
    source=json.loads((root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_EXTRACTION.json').read_text())
    assert source['pred_a'] and hashlib.sha256(path.read_bytes()).hexdigest()==source['scalar_sha256']
    data=torch.load(path,weights_only=True,map_location='cpu')
    results={};errors=[];sources={str(path):source['scalar_sha256']}
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        panel_path=root/f'SHARED_NODE_PARENT1_{suffix}_ROWS.pt'
        sources[str(panel_path)]=hashlib.sha256(panel_path.read_bytes()).hexdigest()
        panel=torch.load(panel_path,weights_only=True,map_location='cpu')
        record=data[label];a,g=record['amplitudes'],record['sensitivities']
        assert record['documents']==panel['documents']
        targets=panel['rows'][:,-1].reshape(3,128)
        total=torch.zeros(2);eligible=0;cells=[];coverage={}
        for family in range(2):
            for domain in sorted(set(record['domain_labels'])):
                docs=[i for i,name in enumerate(record['domain_labels']) if name==domain]
                covered=0
                for token in sorted(set(targets[family,docs].tolist())):
                    assert token in panel['families'][family]
                    ids=torch.tensor([i for i in docs if int(targets[family,i])==token])
                    n=len(ids)
                    if n<2:continue
                    aa,gg=a[family,ids],g[family,ids]
                    cov=-(aa.mul(gg).mean(0)-aa.mean(0)*gg.mean(0))*n/(n-1)
                    pairs=(gg[:,None,:]*(aa[None,:,:]-aa[:,None,:])).sum((0,1))/(n*(n-1))
                    errors.append(float((pairs-cov).abs().max()))
                    total+=n*cov;eligible+=n;covered+=n
                    cells.append(dict(family=family,domain=domain,target_id=token,documents=n,mean_linear_effect=cov.tolist()))
                coverage[f'family{family}/{domain}']=dict(eligible=covered,total=len(docs),fraction=covered/len(docs))
        assert eligible>0
        mean=total/eligible
        results[label]=dict(eligible_endpoints=eligible,total_target_endpoints=256,coverage=eligible/256,
            mean_linear_effect=mean.tolist(),coverage_by_family_domain=coverage,cells=cells,
            pred_b=eligible/256>=.8,pred_c=bool((mean>=.001).all()))
    a=max(errors)<=1e-12
    b=a and all(v['pred_b'] for v in results.values())
    result=dict(pred_a=a,pred_b=b,pred_c=b and all(v['pred_c'] for v in results.values()),
        max_all_pairs_error=max(errors),panels=results,sources=sources,
        scope='Post-result first-order donor expectation conditional on identical target token and domain. '
              'Only covered cells counted; no uncertainty claim, new corpus, causal task semantics or fitted factors.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','sources')},indent=2))
    print(json.dumps({k:{j:w for j,w in v.items() if j!='cells'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
