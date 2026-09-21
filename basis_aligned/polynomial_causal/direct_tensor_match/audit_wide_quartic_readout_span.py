"""Diagnostic readout span floors on already-opened native quartic panels.
Oracle refits are in-sample lower bounds, never held-out validation.
"""
import json,time
from pathlib import Path
import torch
from empirical_quartic_dictionary import features,readout
P=Path(__file__).resolve().parent


def oracle(phi,y):
    scales=phi.square().mean(0).sqrt().clamp_min(1e-12);z=phi/scales
    u,s,vh=torch.linalg.svd(z,full_matrices=False);keep=s>s[0]*1e-10
    writer=(vh[keep].T@((u[:,keep].T@y)/s[keep,None]))/scales[:,None]
    error=float((phi@writer-y).norm()/y.norm())
    return writer,dict(error=error,rank=int(keep.sum()),smallest_relative_singular=float(s[-1]/s[0]))


def controls():
    torch.manual_seed(4600);z=torch.randn(61,7,dtype=torch.float64);c=torch.randn(7,3,dtype=torch.float64);noise=torch.randn(61,3,dtype=torch.float64);q=torch.linalg.qr(z).Q;noise=noise-q@(q.T@noise);y=z@c+noise
    _,r=oracle(z,y);expected=float(noise.norm()/y.norm());mix=torch.randn(7,7,dtype=torch.float64)+4*torch.eye(7,dtype=torch.float64);_,rr=oracle(z@mix,y);_,exact=oracle(z,z@c)
    assert abs(r['error']-expected)<1e-12 and abs(rr['error']-expected)<1e-12 and exact['error']<1e-12
    return dict(known_floor=expected,measured_floor=r['error'],basis_change_floor=rr['error'],exact_span_error=exact['error'])


def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();control=controls()
    data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);ys=[y.double() for y in data['targets']];xs=[r['rows'].double() for r in torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']];records=[]
    for width,initial in [(4,'INHERITED'),(4,'RANDOM'),(32,'INHERITED'),(32,'RANDOM')]:
        program=torch.load(P/f'WIDE_NATIVE_QUARTIC_{width}_{initial}_LONG_V1.pt',weights_only=True);phi=[features(x,program['U'].double(),program['V'].double()) for x in xs]
        c,scales,_=readout(phi[0],ys[0]);writer=c/scales[:,None];baseline=[float((p@writer-y).norm()/y.norm()) for p,y in zip(phi,ys)]
        local=[oracle(p,y)[1] for p,y in zip(phi,ys)]
        joint,pooled=oracle(torch.cat(phi),torch.cat(ys));joint_errors=[float((p@joint-y).norm()/y.norm()) for p,y in zip(phi,ys)]
        row=dict(width=width,initial=initial,calibration_readout_errors=baseline,individual_panel_oracles=local,pooled_oracle=pooled,pooled_panel_errors=joint_errors);records.append(row);print(json.dumps(row),flush=True)
    primary=next(r for r in records if r['width']==32 and r['initial']=='INHERITED')
    result=dict(controls=control,records=records,predictions=dict(second_panel_oracle_below_half_transferred_error=primary['individual_panel_oracles'][1]['error']<.5*primary['calibration_readout_errors'][1],one_pooled_writer_below_5percent_both=max(primary['pooled_panel_errors'])<.05),seconds=time.monotonic()-start,scope='Fixed learned feature directions. Separate/pooled in-sample readout lower bounds on two previously opened panels; rank tolerance1e-10. No new OOD data, no feature refit, no circuit adoption.')
    (P/'WIDE_QUARTIC_READOUT_SPAN_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=result['predictions'],seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
