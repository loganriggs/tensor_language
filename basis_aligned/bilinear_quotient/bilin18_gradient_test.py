"""The section-226 unregistered observation, registered: within bilin12,
attention sharing RISES with depth (Spearman of the full 10-point profile
+0.7-ish) while MLP sharing FALLS. Is the opposite-gradient pattern a family
trait? Complete bilin18's attention profile (add attn0, attn2, attn10,
attn14, attn17 to the measured attn1/4/8/12/16; attn6 excluded as
borrowed-privacy, section 224) and compute both models' depth gradients.

REGISTERED PREDICTIONS: (a) bilin18 MLP profile Spearman(depth, LORO)
<= -0.5 (falls; from the 8 measured writers); (b) bilin12 attention Spearman
>= +0.5 (rises; from the 10 measured writers -- confirmatory); (c) the live
uncertain one: bilin18 attention Spearman >= +0.3 (opposite gradients are a
family trait); alternative <= 0: the rise is 12L-specific and the gradient
story dies. (d) measured nulls <= 0.1 on the five new writers."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_attn_landscape import loro_attn
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_gradient_test_results.json')

def spear(xs,ys):
    import statistics
    rx=sorted(range(len(xs)),key=lambda i:xs[i])
    ry=sorted(range(len(ys)),key=lambda i:ys[i])
    ra=[0]*len(xs); rb=[0]*len(ys)
    for r,i in enumerate(rx): ra[i]=r
    for r,i in enumerate(ry): rb[i]=r
    ma=statistics.mean(ra); mb=statistics.mean(rb)
    num=sum((a-ma)*(b-mb) for a,b in zip(ra,rb))
    den=(sum((a-ma)**2 for a in ra)*sum((b-mb)**2 for b in rb))**0.5
    return num/den

@torch.no_grad()
def main():
    t0=time.time()
    new={}
    for Wl in (0,2,10,14,17):
        med,rnd=loro_attn(Wl)
        new[Wl]=(med,rnd)
        print(f'attn{Wl:2d}: LORO {med:+.3f} (random {rnd:+.3f})',flush=True)
    b18_attn={1:0.485,4:0.292,8:0.347,12:0.474,16:0.384}
    b18_attn.update({k:v[0] for k,v in new.items()})
    b18_mlp={0:0.699,1:0.637,3:0.429,5:0.427,7:0.524,9:0.538,12:0.511}
    b12_attn={0:0.251,1:0.183,2:0.264,3:0.203,4:0.230,5:0.251,
              6:0.382,7:0.389,8:0.400,9:0.392}
    sa18=spear(list(b18_attn),list(b18_attn.values()))
    sm18=spear(list(b18_mlp),list(b18_mlp.values()))
    sa12=spear(list(b12_attn),list(b12_attn.values()))
    pa=sm18<=-0.5; pb=sa12>=0.5; pc=sa18>=0.3; alt=sa18<=0
    pd=all(v[1]<=0.1 for v in new.values())
    out={'new':{str(k):{'loro':v[0],'random':v[1]} for k,v in new.items()},
         'spearman':{'b18_attn':sa18,'b18_mlp':sm18,'b12_attn':sa12},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'alt_12L_specific':bool(alt),'pred_d':bool(pd)}
    print(f'\nSpearman(depth): b18 mlp {sm18:+.2f} | b18 attn {sa18:+.2f} | '
          f'b12 attn {sa12:+.2f}')
    print(f"(a) b18 mlp falls <=-0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) b12 attn rises >=+0.5: {'HELD' if pb else 'FAILED'}")
    print(f"(c) b18 attn rises >=+0.3: {'HELD' if pc else 'FAILED'}"
          f"{' (<=0: rise is 12L-specific)' if alt else ''}")
    print(f"(d) nulls <=0.1: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
