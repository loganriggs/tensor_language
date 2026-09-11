"""Batched exact coefficient energies for joint QK source spaces across positions."""
import json
from pathlib import Path
import torch
from joint_qk_source_ports_v1 import source_ports
from joint_qk_position_influence_v1 import coefficient


def norm(q11,q22,q12,k11,k22,k12):
    trace=lambda x:x.diagonal(dim1=-2,dim2=-1).sum(-1)
    aa=(q11*k11).sum((-2,-1));bb=(q22*k22).sum((-2,-1));ab=(q12*k12).sum((-2,-1))
    return (aa*bb+ab.square()+trace(k11@q12@k22@q12.transpose(-1,-2))
            +trace(k12@q22@k12.transpose(-1,-2)@q11))/4


def ports(query_grams,k1,k2,basis):
    full=(k1@k1.T,k2@k2.T,k1@k2.T)
    a,b=k1@basis,k2@basis;inside=(a@a.T,b@b.T,a@b.T)
    outside=tuple(x-y for x,y in zip(full,inside))
    total=norm(*query_grams,*full);inner=norm(*query_grams,*inside);outer=norm(*query_grams,*outside)
    return dict(total=total,inside=inner/total,mixed=(total-inner-outer)/total,outside=outer/total,touch=1-outer/total)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1291)
    q1,q2=torch.randn(7,4,9),torch.randn(7,4,9);k1,k2=torch.randn(4,11),torch.randn(4,11)
    basis=torch.linalg.qr(torch.randn(11,3)).Q
    grams=q1@q1.transpose(-1,-2),q2@q2.transpose(-1,-2),q1@q2.transpose(-1,-2)
    observed=ports(grams,k1,k2,basis);errors=[]
    for i in range(7):
        expected=source_ports(q1[i],q2[i],k1,k2,basis)
        errors.append(float(abs(observed['total'][i]/expected['total']-1)))
        errors.extend(float(abs(observed[key][i]-expected[key]/expected['total'])) for key in ['inside','mixed','outside'])
    stack=torch.cat((k1,k2),dim=0);small=coefficient(*grams,k1@k1.T,k2@k2.T,k1@k2.T)
    totals=(small*(stack@stack.T).T).sum((-2,-1))
    errors.append(float((totals-observed['total']).norm()/totals.norm()))
    mean=stack.T@(small/totals[:,None,None]).mean(0)@stack
    errors.append(abs(float(mean.trace())-1))
    out=dict(instrument_passed=max(errors)<1e-10,maximum_relative_error=max(errors),
             scope='Batched exact source port energies and equal-position influence normalization. No native position-shared result yet.')
    Path(__file__).with_name('JOINT_QK_POSITION_SPACE_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
