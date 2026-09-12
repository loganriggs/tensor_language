"""Exact conditional joint-key evaluation on saved contextual update ports.

Bars: max FP64 key error <=1e-10 for native, attention-only, MLP-only,
and independently seeded amplitude edits at every saved prefix/layer.
No fitted amplitudes or behavioral claim. RMS epsilon is native FP32 epsilon.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    binding=json.loads((P/'REGIONAL_KEY_PREFIX_V3_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
    sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    records=torch.load(P/'REGIONAL_KEY_PREFIX_V3_ARTIFACT.pt',weights_only=True,map_location='cpu')
    eps=torch.finfo(torch.float32).eps; cells=[]; rng=torch.Generator().manual_seed(6121301)
    for record in records:
        j=record['layer']; z=torch.cat([record['anchor'][None],record['parts']],0).double()
        readers=[sd[f'transformer.h.{j}.attn.{name}.weight'].double().reshape(9,128,1152) for name in ['c_k','c_k2']]
        gram=z@z.T/1152
        projections=[torch.einsum('fd,hkd->fhk',z,k) for k in readers]
        n=z.shape[0]; native=torch.ones(n,dtype=torch.float64)
        attention=native.clone(); attention[2::2]=0
        mlp=native.clone(); mlp[1::2]=0
        synthetic=torch.randn(n,generator=rng,dtype=torch.float64); synthetic[0]=1
        for name,a in [('native',native),('attention',attention),('mlp',mlp),('synthetic',synthetic)]:
            r=a@z; rho2=a@gram@a+eps
            assert rho2>0
            errors=[]
            for k,proj in zip(readers,projections):
                p=torch.einsum('f,fhk->hk',a,proj)
                compiled=p/(p.square().mean(-1,keepdim=True)+eps*rho2).sqrt()
                direct=F.rms_norm(torch.einsum('d,hkd->hk',F.rms_norm(r,(1152,),eps=eps),k),(128,),eps=eps)
                errors.append(float((compiled-direct).norm()/direct.norm()))
            cells.append(dict(prefix=record['prefix'],layer=j,arm=name,relative_errors=errors,
                              port_count=n,conditional_stored_scalars=2*n*9*128+n*n))
    maximum=max(e for c in cells for e in c['relative_errors'])
    result=dict(passed=maximum<=1e-10,max_error=maximum,cells=cells,epsilon=eps,
                scope='Exact FP64 conditional key expression before fixed rotary maps. Saved native-generated ports are fixed; scalar edits do not recompute upstream model states. Native rounded execution was checked separately in prefix V3. No discovery, behavioral or independent extraction claim.')
    (P/'REGIONAL_KEY_PORTS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],max_error=maximum,cells=len(cells),max_conditional_scalars=max(c['conditional_stored_scalars'] for c in cells))))


if __name__=='__main__': main()
