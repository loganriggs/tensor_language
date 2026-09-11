"""Exact full coefficient Gram for two sparse path programs, no dense input tensor."""
import torch
from sparse_path_program_v1 import edges


def feature_bank(program):
    bank=program['bank'];zero=torch.zeros_like(bank[0]);maps=[]
    assignments=[0,1] if program['mode']=='joint' else [0,0,1,1]
    for i,port in enumerate(assignments):
        maps.append(torch.cat([bank[i],zero],0) if port==0 else torch.cat([zero,bank[i]],0))
    return torch.cat(maps,1)


def gram(first,second,output_metric):
    f=feature_bank(first);g=feature_bank(second);input_cross=f.T@g
    a,b=edges(first);c,d=edges(second)
    ka=torch.where(a==b,2.,2**.5).to(f);kb=torch.where(c==d,2.,2**.5).to(f)
    feature=2*(input_cross[a[:,None],c[None,:]]*input_cross[b[:,None],d[None,:]]+
        input_cross[a[:,None],d[None,:]]*input_cross[b[:,None],c[None,:]])/(ka[:,None]*kb[None,:])
    output=first['physical_writer'].T@output_metric@second['physical_writer']
    return output*feature,feature,output
