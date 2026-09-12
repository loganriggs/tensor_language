"""Exact last-bilinear execution on an affine source interface, with live RMS."""
import torch
EPS=1.1920928955078125e-7

def compile_suffix(pre,operator,left,right):
    # pre[N,D], operator[N,F,D]; delta lives in F-dimensional conditional source space.
    return dict(pre=pre,operator=operator,left0=pre@left.T,right0=pre@right.T,left=operator@left.T,right=operator@right.T,gram=operator@operator.transpose(-1,-2),linear=torch.einsum('nd,nfd->nf',pre,operator),norm0=pre.square().sum(-1))

def execute_suffix(delta,program,down,bias):
    p=program;dim=p['pre'].shape[-1]
    norm2=(p['norm0']+2*(delta*p['linear']).sum(-1)+torch.einsum('nf,nfg,ng->n',delta,p['gram'],delta))/dim+EPS
    l=p['left0']+torch.einsum('nf,nfm->nm',delta,p['left']);r=p['right0']+torch.einsum('nf,nfm->nm',delta,p['right'])
    y=p['pre']+torch.einsum('nf,nfd->nd',delta,p['operator'])
    return y+((l*r)/norm2[:,None])@down.T+bias

if __name__=='__main__':
    from pathlib import Path
    import json
    import torch.nn.functional as F
    torch.manual_seed(61208);torch.set_num_threads(2);pre=torch.randn(3,11,dtype=torch.float64);K=torch.randn(3,5,11,dtype=torch.float64);L=torch.randn(17,11,dtype=torch.float64);R=torch.randn_like(L);D=torch.randn(11,17,dtype=torch.float64);b=torch.randn(11,dtype=torch.float64);delta=torch.randn(3,5,dtype=torch.float64)
    p=compile_suffix(pre,K,L,R);h=execute_suffix(delta,p,D,b);y=pre+torch.einsum('nf,nfd->nd',delta,K);x=F.rms_norm(y,(11,),eps=EPS);ref=y+((x@L.T)*(x@R.T))@D.T+b
    error=float((h-ref).norm()/ref.norm());assert error<1e-12
    out=Path(__file__).with_name('AFFINE_BILINEAR_SUFFIX_V1_CONTROL.json');assert not out.exists();out.write_text(json.dumps(dict(relative_error=error,scope='Exact affine-port rational suffix, live input RMS; final RMS/readout remain subsequent operations.'),indent=2)+'\n');print(error)
