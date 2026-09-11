"""Polynomial scaling/block-rotation gauges and a non-gauge input rotation."""
import json,math
from pathlib import Path
import torch
from writer_compensated_tangent_v1 import measure
from chunked_bilinear_coefficient_v1 import dense
from conditional_writer_spectral_v1 import solve


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    e=torch.eye(2);j=torch.tensor([[0.,1.],[-1.,0.]])
    a=torch.stack((e[0],e[0],e[1]));b=torch.stack((e[0],e[1],e[1]));w=torch.eye(3)
    cases=[('reader_scaling',a,b,w,a*torch.tensor([.3,-.7,1.1])[:,None],torch.zeros_like(b)),
        ('complete_block_rotation',a,b,w,a@j,b@j),
        ('isolated_square_rotation',e[:1],e[:1],torch.ones(1,1),e[1:],e[1:])]
    rows=[]
    for name,aa,bb,ww,da,db in cases:
        correction,report=measure(aa,bb,ww,da,db)
        raw=dense(da,bb,ww)+dense(aa,db,ww)
        remaining=raw+dense(aa,bb,correction)
        exact=float(remaining.square().sum());scale=float(raw.square().sum())
        error=abs(exact-report['remaining_energy'])/max(scale,1e-30)
        rows.append(dict(name=name,dense_remaining_energy=exact,dense_replay_error=error,**report))
    theta=.3;rotation=torch.tensor([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
    finite=[]
    for name,aa,bb,ww in [('complete_block',a,b,w),('isolated_square',e[:1],e[:1],torch.ones(1,1))]:
        ar,br=aa@rotation,bb@rotation
        fitted,_,_,solver=solve(ww,aa,bb,ar,br)
        target=dense(aa,bb,ww)
        relative=float((dense(ar,br,fitted)-target).square().sum()/target.square().sum())
        finite.append(dict(name=name,angle=theta,relative_error=relative,writer_solve=solver))
    predictions=dict(pred_a_dense_projection=all(max(r['dense_replay_error'],r['projection_energy_identity_error'],r['writer_solve']['normal_residual'])<=1e-10 for r in rows),
        pred_b_known_gauges=all(abs(r['remaining_fraction'])<=1e-10 for r in rows[:2]),
        pred_c_isolated_change=rows[2]['remaining_fraction']>=.99,
        pred_d_finite_rotation=finite[0]['relative_error']<=1e-10 and finite[1]['relative_error']>=.01)
    result=dict(predictions=predictions,tangents=rows,finite_rotations=finite,
        scope='Small polynomial identifiability control. Complete-block coordinates can rotate without changing the computation; isolated square direction cannot. No native unit identification or behavioral claim.')
    with Path(__file__).with_name('WRITER_COMPENSATED_TANGENT_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(predictions.values())


if __name__=='__main__':main()
