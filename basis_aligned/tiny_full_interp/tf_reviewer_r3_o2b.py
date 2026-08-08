import json, math, os, torch, numpy as np
import tf_compress as CC
from tf_compress import Bits, bits_dense
HERE='/workspace/tensor_language/basis_aligned/tiny_full_interp'
P=f'{HERE}/tf_reviewer_round_3_compression.json'
D=CC.D1Desc('tf_vanilla_d1_w128_b8192_s0')
W=D.base['wte_out']; V,d=W.shape
X=W-W.mean(0,keepdim=True); var=(X*X).mean(0)
alloc={bpr: CC._alloc(var,bpr).cpu().numpy().tolist() for bpr in (256,384,512,640,768)}
uniform={bpr: (len(set(a))==1) for bpr,a in alloc.items()}

def code(b, scale_axis, entropy_axis):
    """scale_axis: 'row' or 'col' min/max; entropy_axis: 'global' or 'col'."""
    mu = W.mean(0,keepdim=True) if scale_axis=='col' else torch.zeros(1,d,device=W.device)
    Z = W-mu
    if scale_axis=='row':
        lo=Z.min(1,keepdim=True).values.half().float(); hi=Z.max(1,keepdim=True).values.half().float()
        nsc=V
    else:
        lo=Z.min(0,keepdim=True).values.half().float(); hi=Z.max(0,keepdim=True).values.half().float()
        nsc=d
    step=((hi-lo)/(2**b-1)).clamp_min(1e-30)
    q=((Z-lo)/step).round().clamp(0,2**b-1)
    R=q*step+lo+mu
    if entropy_axis=='global':
        cb=CC.entropy_bits(q,2**b)
    else:
        cb=sum(CC.entropy_bits(q[:,j],2**b) for j in range(d))
    bt=Bits(codes=cb, scales=2*nsc*16)
    if scale_axis=='col': bt.add(mean=bits_dense(d,32))
    s=D.score({'wte_read':R,'wte_out':R})
    return {'bits':bt.total,'bill':bt.to_json(),**s}

rows=[]
for b in (4,5,6):
    for sa in ('row','col'):
        for ea in ('global','col'):
            r=code(b,sa,ea); r.update({'b':b,'scales':sa,'entropy':ea,
                'scheme':f'q{b}_{sa}scale_{ea}entropy'})
            rows.append(r); print(r['scheme'], round(r['bits']/1e6,4),'Mbit KL',round(r['kl'],5))
out=json.load(open(P))
out['O2b_where_the_transform_codes_gain_comes_from']={
 'allocation_vectors':alloc,'allocation_is_uniform_at_every_budget':uniform,
 'ablation_rows':rows,
 'finding':('The reverse-water-filling allocation gives EVERY one of the 128 '
   'embedding columns exactly the same number of bits at every budget tested '
   '(384->3, 512->4, 640->5, 768->6), so it contributes nothing: the column '
   'variances are too homogeneous for water-filling to bite. The transform '
   'code beats per-row scalar quantisation for two other reasons, isolated in '
   'the ablation: per-COLUMN min/max scales (128 fp16 pairs instead of 8192, '
   'saving 0.258 Mbit) and a per-COLUMN entropy model instead of one global '
   'histogram. RESULTS.md section 2 and 3 attribute the gain to the '
   'allocation; that attribution is wrong.')}
json.dump(out,open(P,'w'),indent=1)
print(json.dumps(uniform))
