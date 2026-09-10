"""Constant and affine calibration controls on fixed train/validation states.

No test rows, fitting labels or new native forwards. Affine least squares uses
centered training states and no ridge; report numerical conditioning/replay.
This is exploratory context for noncentered energy, not circuit identification.
"""
import json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    tic=time.perf_counter();torch.set_num_threads(2)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double();metric=u.T@u;del u
    bias=sd['transformer.h.17.mlp.Down_bias'].double()
    data=torch.load(P/'UNSUPERVISED_DATA_V2_STATES.pt',map_location='cpu',weights_only=True,mmap=True)
    def decode(split):
        rows=data[split+'_rows']
        x=(data['x'][rows].float()*data['x_scale'][rows]).reshape(-1,1152).double()
        y=(data['y'][rows].float()*data['y_scale'][rows]).reshape(-1,1152).double()-bias
        return x,y
    x,y=decode('train');mx,my=x.mean(0),y.mean(0)
    x-=mx;y-=my
    gram=x.T@x;cross=x.T@y
    print('Solving unregularized affine baseline',flush=True)
    coefficient=torch.linalg.solve(gram,cross)
    solve_error=float((gram@coefficient-cross).norm()/cross.norm())
    condition=float(torch.linalg.cond(gram))
    results={}
    for split in ['train','validation']:
        if split=='train':xx,yy=x,y
        else:
            xx,yy=decode(split);xx-=mx;yy-=my
        total=constant=affine=0.
        for first in range(0,len(xx),1024):
            bx,by=xx[first:first+1024],yy[first:first+1024]
            original=by+my;residual=by-bx@coefficient
            total+=float(((original@metric)*original).sum())
            constant+=float(((by@metric)*by).sum())
            affine+=float(((residual@metric)*residual).sum())
        results[split]=dict(states=len(xx),native_energy=total,
            constant_squared_relative_error=constant/total,affine_squared_relative_error=affine/total,
            affine_error_relative_to_mean_centered_energy=affine/constant)
    result=dict(schema='natural.state.baselines.v1',results=results,
        centered_input_gram_condition=condition,normal_equation_relative_residual=solve_error,
        price=dict(constant_parameter_numbers=1152,affine_parameter_numbers=1152*1152+1152,
            product128_parameter_numbers=3*1152*128,body_forwards=0),
        scope='Historically opened corpus, fixed row train/validation splits only. Test unopened. Full unembedding output metric before final RMS/tanh. Affine baseline is larger than product128; neither is an identified circuit.',
        wall_seconds=time.perf_counter()-tic)
    with (P/'NATURAL_STATE_BASELINES_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
