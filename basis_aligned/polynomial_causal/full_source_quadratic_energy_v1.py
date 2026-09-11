"""Exact shared-source symmetry energy for every output at once.

Contract output Gram in native product coordinates. Never form a token ×
(head×source)^2 tensor or restrict to a fitted output subspace.
"""
import torch


def energies(l,r,o,values,output_gram,heads):
    width=o.shape[1]//heads
    lo=l@o;ro=r@o;gram=values@values.T
    blocks=[];self_grams=[]
    for h in range(heads):
        hs=slice(h*width,(h+1)*width);a=lo[:,hs];b=ro[:,hs];g=gram[hs,hs]
        blocks.append((a,b,hs))
        self_grams.append((a@g@a.T,b@g@b.T,a@g@b.T))
    total=l.new_zeros(());partial_inner=l.new_zeros(());diagonal=l.new_zeros(())
    for h,(la,ra,hs) in enumerate(blocks):
        aa,bb,ab=self_grams[h]
        for k in range(h,heads):
            lb,rb,ks=blocks[k];cc,dd,cd=self_grams[k]
            norm=((aa*dd+bb*cc+2*ab*cd.T)*output_gram).sum()/4
            if h==k:
                total+=norm;partial_inner+=norm;diagonal+=norm
                continue
            g=gram[ks,hs]
            c=rb@g@la.T;d=lb@g@ra.T
            a=lb@g@la.T;b=rb@g@ra.T
            trace=((c*c.T+d*d.T+2*a*b.T)*output_gram).sum()/4
            total+=2*norm;partial_inner+=2*trace
    return dict(total=total,symmetric=(total+partial_inner)/2,
                antisymmetric=(total-partial_inner)/2,head_diagonal=diagonal)


def controls():
    from attention_output_quadratic_pullback_v1 import dense_forms
    from shared_source_attention_quadratic_v1 import pullback,split_source_symmetry
    torch.manual_seed(387);dt=torch.float64
    l,r=torch.randn(8,6,dtype=dt),torch.randn(8,6,dtype=dt)
    w=torch.randn(7,8,dtype=dt);o=torch.randn(6,6,dtype=dt);v=torch.randn(6,8,dtype=dt)
    q=dense_forms(l,r,w);maps=torch.stack([o[:,h*2:(h+1)*2]@v[h*2:(h+1)*2] for h in range(3)])
    t=pullback(q,maps);plus,minus=split_source_symmetry(t)
    found=energies(l,r,o,v,w.T@w,3)
    expected=dict(total=t.square().sum(),symmetric=plus.square().sum(),antisymmetric=minus.square().sum(),head_diagonal=sum(t[:,h,:,h,:].square().sum() for h in range(3)))
    errors={k:float(abs(found[k]-expected[k])/expected['total']) for k in expected}
    # Shared orthogonal source permutation preserves all parts, independent
    # head permutations preserve total and diagonal but can change the split.
    perm=torch.randperm(8);common=energies(l,r,o,v[:,perm],w.T@w,3)
    errors['common_permutation']=max(float(abs(common[k]-found[k])/found['total']) for k in found)
    independent=torch.cat([x[:,torch.randperm(8)] for x in v.split(2)],dim=0)
    other=energies(l,r,o,independent,w.T@w,3)
    errors['independent_total']=float(abs(other['total']/found['total']-1))
    return dict(errors=errors,passed=max(errors.values())<=1e-10)

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);r=controls();print(json.dumps(r,indent=2))
    Path(__file__).with_name('FULL_SOURCE_QUADRATIC_ENERGY_V1_CONTROL.json').write_text(json.dumps(r,indent=2)+'\n');assert r['passed']
