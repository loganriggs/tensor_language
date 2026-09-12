"""Known-answer closure with native-style RMS, reentry and joint factor edits."""
import hashlib,json
from pathlib import Path
import torch
from closed_feature_program_v1 import encode,execute
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(6133901)
    d,r,m,v=64,8,12,96;dtype=torch.float64;eps=torch.finfo(torch.float32).eps
    b=torch.linalg.qr(torch.randn(d,r,dtype=dtype),mode='reduced')[0]
    readout=torch.randn(v,d,dtype=dtype)/d**.5;blocks=[];dense=[]
    for j in range(18):
        left=.3*torch.randn(m,r,dtype=dtype)/r**.5;right=.3*torch.randn(m,r,dtype=dtype)/r**.5
        down=.15*torch.randn(r,m,dtype=dtype)/m**.5;bias=.01*torch.randn(r,dtype=dtype)
        reentry=torch.tensor([.97+.01*(j%3),.02+.01*(j%2)],dtype=dtype)
        blocks.append(dict(left=left,right=right,down=down,bias=bias,reentry=reentry))
        dense.append(dict(left=left@b.T,right=right@b.T,down=b@down,bias=b@bias,reentry=reentry))
    def reference(x0,edits):
        x=x0.clone()
        for j,block in enumerate(dense):
            lam,mu=block['reentry'];x=lam*x+mu*x0
            inp=x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
            product=(inp@block['left'].T)*(inp@block['right'].T)
            for layer,factor in edits:
                if layer==j:product[:,factor]=0
            x=x+product@block['down'].T+block['bias']
        logit=(x@readout.T)/(x.square().mean(-1,keepdim=True)+eps).sqrt()
        return x,30*torch.tanh(logit/30)
    def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
    masks=[(),((5,1),),((12,2),),((5,1),(12,2))];cells=[]
    for scale in (.3,1.,3.):
        x0=scale*torch.randn(64,d,dtype=dtype);encoded=encode(x0,b,readout)
        actual=[];compiled=[];features=[];norms=[]
        for edits in masks:
            x,y=reference(x0,edits);result=execute(encoded,blocks,edits)
            actual.append(y);compiled.append(result['logits'])
            features.append(rel(result['features'],x@b));norms.append(rel(result['norm2'],x.square().sum(-1)))
        a=torch.stack(actual);c=torch.stack(compiled)
        ie=a[3]-a[2]-a[1]+a[0];ice=c[3]-c[2]-c[1]+c[0]
        cells.append(dict(scale=scale,max_logits_error=max(rel(c[i],a[i]) for i in range(4)),max_feature_error=max(features),max_norm_error=max(norms),
                          effect_errors=[rel(c[i]-c[0],a[i]-a[0]) for i in range(1,4)],effect_norms=[float((a[i]-a[0]).norm()) for i in range(1,4)],
                          interaction_norm=float(ie.norm()),interaction_relative_error=rel(ice,ie),
                          missing_complement_norm_error=rel(execute(encoded,blocks,omit_perpendicular_norm=True)['logits'],a[0])))
    a=all(max(c['max_logits_error'],c['max_feature_error'],c['max_norm_error'])<=1e-11 for c in cells)
    bb=a and all(max(c['effect_errors'])<=1e-8 and min(c['effect_norms'])>1e-5 for c in cells)
    cc=bb and all(c['interaction_norm']>1e-9 and c['interaction_relative_error']<=1e-6 for c in cells) and max(c['missing_complement_norm_error'] for c in cells)>=1e-3
    price=dict(dense_weight_scalars=sum(t.numel() for block in dense for t in block.values())+readout.numel(),
               reduced_weight_scalars=sum(t.numel() for block in blocks for t in block.values())+b.numel()+readout.numel()+v*r,
               per_input_encoded_scalars=r+1+v,per_input_changing_features=r,
               caveat='Arbitrary full readout retains96 initial-complement values per input and original readout weights for encoding. This is not an8-scalar total state or a native compression measurement.')
    result={'pred_a':a,'pred_b':bb,'pred_c':cc};result.update(cells=cells,price=price,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha=hashlib.sha256((P/'closed_feature_program_v1.py').read_bytes()).hexdigest(),scope='Seeded common-frame positive control, not trained bilin18 or linguistic OOD. Candidate closure does not use intermediate native state; restrictive reader/writer support assumption is known by construction.')
    (P/'CLOSED_FEATURE_PROGRAM_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
