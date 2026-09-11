"""Exact statistical accounting for single-port product mean shifts; no fitting.
A independent pairwise accounting <=1e-12. B mean shift explains >=75% of
negative branch0 common-port effect on both panels. C centered common-port
effect >=.001 for both branches on both panels. No behavioral rescue claim.
"""
import hashlib
import json
from pathlib import Path
import torch
from branch_port_interchange_v1 import roles


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;output=root/'BRANCH_PORT_MEAN_ACCOUNTING_V1.json'
    assert not output.exists()
    paths=[root/'BRANCH_ALL_DONOR_ALIGNMENT_V1_SCALARS.pt',root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',
           root/'BRANCH_PORT_INTERCHANGE_V1.json',root/'branch_port_interchange_v1.py']
    records=torch.load(paths[0],weights_only=True,map_location='cpu')
    node=torch.load(paths[1],weights_only=True,map_location='cpu')['nodes'][1]
    old=json.loads(paths[2].read_text());assert old['pred_a']
    errors=[];results={}
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        path=root/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt';paths.append(path)
        x=torch.load(path,weights_only=True,map_location='cpu')['ports']['input'].double()
        s=(x@node['reader'].double()).reshape(3,128,1)
        p=(x@node['partners'].double()).reshape(3,128,2)
        record=records[label];g=record['sensitivities']
        assert float((s*p-record['amplitudes']).norm())==0
        whole=torch.zeros(2,4);mean_piece=torch.zeros_like(whole);centered=torch.zeros_like(whole)
        details={}
        for domain in sorted(set(record['domain_labels'])):
            ids=torch.tensor([i for i,name in enumerate(record['domain_labels']) if name==domain])
            ss,pp,gg=s[:,ids],p[:,ids],g[:,ids];n=len(ids)
            total=roles(ss,pp,gg)
            shift=-(ss.mul(pp).mean(1)-ss.mean(1)*pp.mean(1))*n/(n-1)
            shifts=torch.stack((shift,shift,-2*shift,torch.zeros_like(shift)),-1)
            bias=gg.mean(1)[:,:,None]*shifts
            remainder=roles(ss,pp,gg-gg.mean(1,keepdim=True))
            ds=ss[:,None,:,:]-ss[:,:,None,:];dp=pp[:,None,:,:]-pp[:,:,None,:]
            changes=torch.stack((ds*pp[:,:,None,:],ss[:,:,None,:]*dp,ds*dp,
                ss[:,None,:,:]*pp[:,None,:,:]-ss[:,:,None,:]*pp[:,:,None,:]),-1)
            direct_shifts=changes.sum((1,2))/(n*(n-1))
            errors.extend([float((shifts-direct_shifts).abs().max()),float((total-bias-remainder).abs().max()),
                float(shifts[:,:,:3].sum(-1).abs().max())])
            whole+=n/128*total.mean(0);mean_piece+=n/128*bias.mean(0);centered+=n/128*remainder.mean(0)
            details[domain]=dict(per_family_mean_amplitude_shift=shifts.tolist(),
                per_family_mean_sensitivity=gg.mean(1).tolist(),mean_shift_effect=bias.mean(0).tolist(),
                centered_effect=remainder.mean(0).tolist())
        errors.append(float((whole-torch.tensor(old['panels'][label]['mean_linear_effect'])).abs().max()))
        ratio=float(mean_piece[0,0]/whole[0,0])
        results[label]=dict(total_effect=whole.tolist(),mean_shift_effect=mean_piece.tolist(),
            centered_sensitivity_effect=centered.tolist(),branch0_common_mean_shift_ratio=ratio,
            pred_b=bool(whole[0,0]<0 and ratio>=.75),pred_c=bool((centered[:,0]>=.001).all()),
            per_domain=details)
    a=max(errors)<=1e-12
    result=dict(pred_a=a,pred_b=a and all(v['pred_b'] for v in results.values()),
        pred_c=a and all(v['pred_c'] for v in results.values()),max_identity_error=max(errors),
        intervention_order=old['intervention_order'],panels=results,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        scope='Exact first-order statistical accounting. Empirical means are used only to decompose a measured effect; '
              'no model replacement, learned factors, full-dose claim or repair of the shared-port B/C miss.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','panels')},indent=2))
    print(json.dumps({k:{j:w for j,w in v.items() if j!='per_domain'} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
