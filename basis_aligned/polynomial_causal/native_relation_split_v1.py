"""Exact split of the frozen suffix program; no fitting or semantic promotion.
A independent leading-eigen/product, trace and full-execution replay<=1e-9.
The original radial/bias belongs to the leading component; remainder is traceless.
"""
import hashlib
import json
from pathlib import Path
import torch


def evaluate(artifact,x):
    radius=x.square().sum(-1)
    leading=[];remainder=[]
    for c in artifact['components']:
        leading.append((x@c['a'])*(x@c['b'])+c['leading_radial']*radius+c['bias'])
        remainder.append((x@c['rest_readers'].T).square()@c['rest_coefficients']+c['rest_radial']*radius)
    return torch.stack(leading,-1),torch.stack(remainder,-1)


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    p=Path(__file__).parent;source=p/'NATIVE_TOKEN_RELATION_FOLD_V1.pt'
    output=p/'NATIVE_RELATION_SPLIT_V1.json';artifact=p/'NATIVE_RELATION_SPLIT_V1.pt'
    assert not output.exists() and not artifact.exists()
    fold=json.loads((p/'NATIVE_TOKEN_RELATION_FOLD_V1.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==fold['artifact_sha256']
    saved=torch.load(source,weights_only=True,map_location='cpu');components=[];errors=[]
    torch.manual_seed(6301);x=torch.randn(19,1152);references=[]
    for j,c in enumerate(saved['compact'][:3]):
        v,coef=c['top16_square_readers'],c['top16_square_coefficients']
        ip,im=int(coef.argmax()),int(coef.argmin());assert coef[ip]>0 and coef[im]<0
        a,b=c['leading_product_a'],c['leading_product_b']
        expected=coef[ip]*torch.outer(v[ip],v[ip])+coef[im]*torch.outer(v[im],v[im])
        actual=(torch.outer(a,b)+torch.outer(b,a))/2
        errors.append(float((actual-expected).norm()/expected.norm()))
        keep=[i for i in range(16) if i not in (ip,im)];vr,cr=v[keep],coef[keep]
        trace_rest=(cr*vr.square().sum(1)).sum();trace_full=(coef*v.square().sum(1)).sum()
        leading_radial=c['radial']-(a@b)/1152;rest_radial=-trace_rest/1152
        errors.append(float(abs(leading_radial+rest_radial-(c['radial']-trace_full/1152))))
        components.append(dict(a=a,b=b,leading_radial=leading_radial,bias=saved['folded_bias'][j],
            rest_readers=vr,rest_coefficients=cr,rest_radial=rest_radial,relation=saved['labels'][j]))
        references.append((x@v.T).square()@coef+(c['radial']-trace_full/1152)*x.square().sum(1)+saved['folded_bias'][j])
    r=saved['physical_readouts'][:3];writers=torch.linalg.solve(r@r.T,r).T
    program=dict(components=components,readouts=r,writers=writers)
    lead,rest=evaluate(program,x);reference=torch.stack(references,1)
    errors.append(float((lead+rest-reference).norm()/reference.norm()))
    checks=[]
    for stem in ('FROZEN_BRANCH_MORPHOLOGY_V1','NATIVE_RELATION_HOLDOUT_V1'):
        cache=torch.load(p/(stem+'_ENDPOINTS.pt'),weights_only=True,map_location='cpu');xn=cache['ports']['input'].double()
        l,rn=evaluate(program,xn);direct=[]
        for j,c in enumerate(saved['compact'][:3]):
            v,coef=c['top16_square_readers'],c['top16_square_coefficients'];trace=(coef*v.square().sum(1)).sum()
            direct.append((xn@v.T).square()@coef+(c['radial']-trace/1152)*xn.square().sum(1)+saved['folded_bias'][j])
        direct=torch.stack(direct,1);error=float((l+rn-direct).norm()/direct.norm());errors.append(error)
        checks.append(dict(panel=stem,execution_error=error))
    torch.save(program,artifact)
    result=dict(pred_a=max(errors)<=1e-9,replay_errors=errors,natural_checks=checks,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        price=dict(variable_products=45,dense_input_readers=48,shared_radius_required=True,physical_writers=3,
                   native_background_retained=True),
        scope='Known signed-square pairing, exact split of one frozen program. Leading part includes original radial/bias; '
        'remaining part has its trace removed. No inferred semantic hierarchy, fitting, native full-layer replacement or behavioral result.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
