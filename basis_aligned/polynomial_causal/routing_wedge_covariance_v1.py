"""Exact output/source Gram in the 36-dimensional head-routing wedge mode."""
import torch


def covariance(l,r,o,values,output_gram,heads):
    width=o.shape[1]//heads;lo=l@o;ro=r@o;g=values@values.T
    left=list(lo.split(width,dim=1));right=list(ro.split(width,dim=1))
    pairs=[(h,k) for h in range(heads) for k in range(h+1,heads)]
    result=l.new_zeros(len(pairs),len(pairs))
    def anchor(h):
        hs=slice(h*width,(h+1)*width);blocks=[]
        for u in range(heads):
            us=slice(u*width,(u+1)*width);gu=g[hs,us]
            a=left[h]@gu;b=right[h]@gu
            blocks.append((a@left[u].T,a@right[u].T,b@left[u].T,b@right[u].T))
        return blocks
    row=0
    for h in range(heads-1):
        hc=anchor(h)
        for k in range(h+1,heads):
            kc=anchor(k)
            for col in range(row,len(pairs)):
                u,v=pairs[col]
                # Order LL, LR, RL, RR. Each coefficient is sqrt(2) times
                # skew(B_hk), so the head-mode Gram has factor1/4 here.
                hu,hv,ku,kv=hc[u],hc[v],kc[u],kc[v]
                value=hu[0]*kv[3]-hv[1]*ku[2]
                value+=hu[1]*kv[2]-hv[0]*ku[3]
                value+=hu[2]*kv[1]-hv[3]*ku[0]
                value+=hu[3]*kv[0]-hv[2]*ku[1]
                scalar=(value*output_gram).sum()/4
                result[row,col]=scalar;result[col,row]=scalar
            del kc
            row+=1
        del hc
    return result,pairs


def controls():
    from attention_output_quadratic_pullback_v1 import dense_forms
    from shared_source_attention_quadratic_v1 import pullback,split_source_symmetry
    torch.manual_seed(425);dt=torch.float64
    l,r=torch.randn(8,6,dtype=dt),torch.randn(8,6,dtype=dt)
    w=torch.randn(7,8,dtype=dt);o=torch.randn(6,6,dtype=dt);v=torch.randn(6,8,dtype=dt)
    q=dense_forms(l,r,w);maps=torch.stack([o[:,h*2:(h+1)*2]@v[h*2:(h+1)*2] for h in range(3)])
    t=pullback(q,maps);_,minus=split_source_symmetry(t)
    actual,pairs=covariance(l,r,o,v,w.T@w,3)
    rows=torch.stack([(2**.5*minus[:,h,:,k,:]).flatten() for h,k in pairs]);expected=rows@rows.T
    relative=float((actual-expected).norm()/expected.norm());trace=float(abs(actual.trace()/minus.square().sum()-1))
    # The covariance eigensystem is the SVD of the explicit routing unfolding.
    ev=torch.linalg.eigvalsh(actual).flip(0);sv=torch.linalg.svdvals(rows).square()
    spectrum=float((ev-sv).norm()/sv.norm())
    return dict(dense_covariance_relative_error=relative,trace_relative_error=trace,dense_svd_relative_error=spectrum,passed=max(relative,trace,spectrum)<=1e-10)

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);r=controls();print(json.dumps(r,indent=2))
    Path(__file__).with_name('ROUTING_WEDGE_COVARIANCE_V1_CONTROL.json').write_text(json.dumps(r,indent=2)+'\n');assert r['passed']
