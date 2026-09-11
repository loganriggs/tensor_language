"""Exact private-square completion with preserved parent-port interventions.
A FP64/core/port identities<=1e-9 and planted naive-ablation tripwire.
B saves>=20 variable products with no array, coefficient multiply or addition increase.
C FP32 function and parent-effect replay<=1e-5 on fixed synthetic inputs.
"""
import hashlib
import json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v1 import execute


def shared_matrix(group):
    k=len(group['parent_ids']);h=group['lam'].new_zeros(k,k);offset=0
    for i in range(k):
        for j in range(i,k):
            h[i,j]=h[j,i]=group['shared_coeff'][offset]/(1 if i==j else 2)
            offset+=1
    return h


def packed(h):
    return torch.stack([h[i,j]*(1 if i==j else 2) for i in range(len(h)) for j in range(i,len(h))])


def evaluate(graph,patches,x,intervention=None,new=True):
    z=x@graph['readers'].to(x).T
    if intervention=='zero1':z[:,1]=0
    if intervention=='zero1_9':z[:,[1,9]]=0
    if intervention=='zero_all':z.zero_()
    if intervention=='swap1':z[:,1]=z[:,1].roll(1)
    pairs=graph['pairs'];products=z[:,pairs[:,0]]*z[:,pairs[:,1]]
    result=x.new_zeros(len(x),graph['groups'][0]['writer'].numel())
    for group,patch in zip(graph['groups'],patches):
        y=x@group['private'].to(x).T;k=len(group['parent_ids'])
        if new and patch['rewritten']:
            y=y+z[:,group['parent_ids']]@patch['mix'].to(x).T
            scalar=(y.square()*group['lam'].to(x)).sum(1)
            scalar=scalar+products[:,group['pair_ids']]@patch['shared_coeff'].to(x)
        else:
            scalar=(y.square()*group['lam'].to(x)).sum(1)
            scalar=scalar+products[:,group['pair_ids']]@group['shared_coeff'].to(x)
            if k:scalar=scalar+(z[:,group['parent_ids']]*(y@group['cross'].to(x).T)).sum(1)
        result+=scalar[:,None]*group['writer'].to(x)
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;output=root/'SCHUR_GRAPH_REWRITE_V1.json'
    assert not output.exists()
    path=root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt'
    graph=torch.load(path,weights_only=True,map_location='cpu')
    patches=[];core_errors=[];growth=[];skipped=[];min_pivot=[]
    old_products=len(graph['pairs']);new_products=old_products
    old_add=0;new_add=0;old_coeff=0;new_coeff=0
    for index,group in enumerate(graph['groups']):
        k=len(group['parent_ids']);l=len(group['lam']);t=len(group['pair_ids'])
        old_products+=l+k;new_products+=l
        old_add+=l-1 if k==0 else l+t+k*l-1
        old_coeff+=l+t+k*l
        if not k:
            patches.append(dict(rewritten=False));new_add+=l-1;new_coeff+=l;continue
        lam=group['lam'].double();cross=group['cross'].double();h=shared_matrix(group)
        ratio=float(lam.abs().min()/lam.abs().max());min_pivot.append(ratio)
        if ratio<=1e-12:
            skipped.append(index);patches.append(dict(rewritten=False));new_products+=k
            new_add+=l+t+k*l-1;new_coeff+=l+t+k*l;continue
        mix=cross.T/(2*lam[:,None]);schur=h-cross@mix/2
        patch=dict(rewritten=True,mix=mix,shared_coeff=packed(schur))
        patches.append(patch)
        old=torch.cat((torch.cat((h,cross/2),1),torch.cat((cross.T/2,torch.diag(lam)),1)),0)
        top=schur+mix.T@(lam[:,None]*mix)
        after=torch.cat((torch.cat((top,mix.T*lam[None]),1),torch.cat((lam[:,None]*mix,torch.diag(lam)),1)),0)
        core_errors.append(float((old-after).norm()/old.norm()))
        growth.append(dict(group=index,min_relative_pivot=ratio,max_mix=float(mix.abs().max()),
            schur_norm=float(schur.norm()),old_shared_norm=float(h.norm()),
            schur_to_whole_core_norm=float(schur.norm()/old.norm())))
        new_add+=k*l+l+t-1;new_coeff+=k*l+l+t
        assert mix.numel()==cross.numel() and patch['shared_coeff'].numel()==group['shared_coeff'].numel()
    x=torch.randn(32,1152,generator=torch.Generator().manual_seed(6001))
    x=x/x.square().mean(1,keepdim=True).sqrt()
    base_old=evaluate(graph,patches,x,new=False);base_new=evaluate(graph,patches,x,new=True)
    reference=execute(graph,x)
    reference_error=float((base_old-reference).norm()/reference.norm())
    checks={}
    for dtype in (torch.float64,torch.float32):
        xx=x.to(dtype);oldbase=evaluate(graph,patches,xx,new=False);newbase=evaluate(graph,patches,xx,new=True)
        per=[]
        for intervention in (None,'zero1','zero1_9','zero_all','swap1'):
            before=evaluate(graph,patches,xx,intervention,new=False)
            after=evaluate(graph,patches,xx,intervention,new=True)
            row=dict(intervention=intervention,output_relative_error=float((before-after).norm()/before.norm()))
            if intervention is not None:
                effect=oldbase-before;new_effect=newbase-after
                row['effect_relative_error']=float((effect-new_effect).norm()/effect.norm())
            per.append(row)
        checks[str(dtype)]=per
    # y^2+2zy=(y+z)^2-z^2: parent clamp must also enter the shifted square.
    z,y=1.,1.;native=y*y+2*z*y;clamped=y*y
    correct=(y+0)**2-0**2;naive=(y+z)**2
    tripwire=abs(correct-clamped)==0 and abs(naive-clamped)>=1
    e64=max(v for row in checks['torch.float64'] for k,v in row.items() if k.endswith('error'))
    e32=max(v for row in checks['torch.float32'] for k,v in row.items() if k.endswith('error'))
    a=not skipped and max(core_errors+[reference_error,e64])<=1e-9 and tripwire
    b=a and old_products-new_products>=20 and new_add<=old_add and new_coeff<=old_coeff
    artifact=root/'SCHUR_GRAPH_REWRITE_V1_PATCH.pt';assert not artifact.exists()
    torch.save(dict(patches=patches,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),artifact)
    result=dict(pred_a=a,pred_b=b,pred_c=a and e32<=1e-5,skipped_groups=skipped,
        max_core_error=max(core_errors),reference_executor_error=reference_error,checks=checks,growth=growth,
        price=dict(old_variable_products=old_products,new_variable_products=new_products,
            old_local_additions=old_add,new_local_additions=new_add,
            old_local_coefficient_multiplies=old_coeff,new_local_coefficient_multiplies=new_coeff,
            floating_count_change=0,integer_count_change=0,
            note='Compiled mix replaces cross; corrected shared coefficients replace old ones. Patch artifact stores only changed arrays. '
                 'Shared/private dense reads, group writers and native background remain unchanged and charged by the original graph.'),
        planted=dict(original=native,correct_clamped=correct,naive_drop_explicit_shared_only=naive,tripwire_held=tripwire),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),patch_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        scope='Exact algebraic compiler rewrite of the frozen surrogate, not a new factor fit, global minimum or semantic circuit. '
              'Parent interventions must be applied before the shifted-square mix. Synthetic FP32 checks are not model runtime benchmarks.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','growth')},indent=2))
    print(json.dumps(dict(max_fp64_error=e64,max_fp32_error=e32,max_mix=max(r['max_mix'] for r in growth),
        min_relative_pivot=min(min_pivot),max_schur_core_ratio=max(r['schur_to_whole_core_norm'] for r in growth))))


if __name__=='__main__':main()
