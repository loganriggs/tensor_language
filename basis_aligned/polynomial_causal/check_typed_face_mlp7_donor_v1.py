"""CPU algebra/control receipt for the proposed earlier donor-state boundary."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from typed_face_mlp7_donor_v1 import generate,source_gram
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(17092143)
    # Exact synthetic identity, with independent explicit ordered expansion.
    b,d,h=5,13,19
    p={name:torch.randn(shape,dtype=torch.float64) for name,shape in
       [('left',(h,d)),('right',(h,d)),('down',(d,h)),('bias',(d,)),('lambda8',(2,))]}
    g=torch.randn(b,d,dtype=torch.float64);e=torch.randn_like(g)
    z=F.rms_norm(g,(d,));hidden=(z@p['left'].T)*(z@p['right'].T)
    pieces=torch.stack([p['lambda8'][0]*g,p['lambda8'][0]*(hidden@p['down'].T+p['bias']),p['lambda8'][1]*e],1)
    denom=(source_gram(pieces).sum((-1,-2))+torch.finfo(g.dtype).eps).sqrt()
    expected=pieces.sum(1)/denom[:,None]
    err=float((generate(p,g,e)-expected).norm()/expected.norm())
    # Native opened sources: normalization identity and omitted-cross diagnostic.
    path=P/'TYPED_FACE_KEY_SOURCE_FRESH_V1_ARTIFACT.pt'
    a=torch.load(path,weights_only=True,map_location='cpu')['source_arrays']
    sources=a.squeeze(2).double();assert sources.shape==(16,3,1152)
    gram=source_gram(sources);full=gram.sum((-1,-2));direct=sources.sum(1).square().mean(-1)
    gram_error=float((full-direct).norm()/direct.norm())
    diagonal=gram.diagonal(dim1=-2,dim2=-1).sum(-1)
    # eps is FP32 because these source vectors came from the native FP32 model.
    eps=torch.finfo(torch.float32).eps
    scale_error=((full+eps)/(diagonal+eps)).sqrt()-1
    result={'synthetic_generator_relative_error':err,'native_source_gram_relative_error':gram_error,
      'pred_a':err<=1e-12,'pred_b':gram_error<=1e-12,
      'diagonal_only_normalizer_relative_scale_errors':scale_error.tolist(),
      'native_source_shape':list(a.shape),'panel_status':'opened fresh-transfer artifacts; algebra only',
      'scope':'Exact proposed MLP7 donor generator, no native weight replay yet. Cross-term diagnostic is a state normalization discrepancy, not a behavioral effect. Recipient native state and earlier donor g7 remain two ports; full native MLP weights must be charged.',
      'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (P/'TYPED_FACE_MLP7_DONOR_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['pred_a'] and result['pred_b'];print(json.dumps(result,indent=2))
if __name__=='__main__':main()
