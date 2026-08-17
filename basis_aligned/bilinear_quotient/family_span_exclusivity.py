"""Is the private span exclusive on the WRITE side too? The reader population
declines to share a vocabulary over span 6:1-8 (bilin18) / 4:1-8 (bilin12).
If no other component writes into those directions either, the span is a
dedicated sub-channel -- a point-to-point wire from the fraction-1/3 writer
to the output end (section 214's direct channel), and "readers decline" would
reframe as "the stream reserves an address." Measure absolute write energy
into the private span for every component (MLP and attention outputs, all
layers), against each component's own matched-random-span baseline (3 random
8-dim spans from the owner's output distribution, components 9+).

REGISTERED PREDICTIONS: (a) the owner MLP's write energy into its span is
>= 5x every other MLP's (write-dominant); (b) every non-owner component's
write into the span is <= 2x its own random-span baseline (nobody else AIMS
at it -- exclusivity); alternative: specific components exceed 3x = the span
is co-written, name them; (c) same verdicts in bilin12 (family check)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'family_span_exclusivity_results.json')

@torch.no_grad()
def scan(name, owner):
    m2,_=load_elriggs(name, device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    nl=len(m2.transformer.h)
    caps={('mlp',li):[] for li in range(nl)}
    caps.update({('attn',li):[] for li in range(nl)})
    hs=[]
    for li in range(nl):
        def mkm(li=li):
            return lambda mo_,i_,o_: caps[('mlp',li)].append(
                o_.detach().reshape(-1,D).float())
        def mka(li=li):
            return lambda mo_,i_,o_: caps[('attn',li)].append(
                (o_[0] if isinstance(o_,tuple) else o_)
                .detach().reshape(-1,D).float())
        hs.append(m2.transformer.h[li].mlp.register_forward_hook(mkm()))
        hs.append(m2.transformer.h[li].attn.register_forward_hook(mka()))
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    caps={k:torch.cat(v) for k,v in caps.items()}
    Yo=caps[('mlp',owner)]; mu=Yo.mean(0)
    _,_,Vh=torch.linalg.svd((Yo-mu).float(), full_matrices=False)
    Q=orth(Vh[:8].T)
    g=torch.Generator(device=DEV).manual_seed(0)
    Rs=[orth(Vh[8:].T@torch.randn(Vh.shape[0]-8,8,device=DEV,generator=g))
        for _ in range(3)]
    tab={}
    for (typ,li),Y in caps.items():
        Yc=Y-Y.mean(0)
        e=float((Yc@Q).pow(2).sum(1).mean())
        er=sorted(float((Yc@R).pow(2).sum(1).mean()) for R in Rs)[1]
        tab[(typ,li)]=(e,er,e/max(er,1e-9))
    own=tab[('mlp',owner)][0]
    others=[(k,v) for k,v in tab.items() if k!=('mlp',owner)]
    mx_mlp=max(v[0] for k,v in others if k[0]=='mlp')
    aimers=[f'{k[0]}{k[1]}' for k,v in others if v[2]>3.0]
    pa=own>=5*mx_mlp
    pb=all(v[2]<=2.0 for k,v in others)
    for (typ,li),(e,er,r) in sorted(tab.items(),key=lambda x:-x[1][0])[:8]:
        print(f'{name} {typ}{li:2d}: E {e:9.1f} rndE {er:9.1f} ratio {r:5.2f}',
              flush=True)
    print(f'{name}: owner E {own:.1f} | max other-MLP E {mx_mlp:.1f} | '
          f'aimers>3x: {aimers if aimers else "none"}',flush=True)
    del m2; torch.cuda.empty_cache()
    return {'owner_E':own,'max_other_mlp_E':mx_mlp,'aimers':aimers,
            'pred_a':bool(pa),'pred_b':bool(pb),
            'tab':{f'{k[0]}{k[1]}':v for k,v in tab.items()}}

@torch.no_grad()
def main():
    t0=time.time()
    r18=scan('bilin18',6)
    r12=scan('bilin12',4)
    out={'bilin18':r18,'bilin12':r12}
    for nm,r in (('bilin18',r18),('bilin12',r12)):
        print(f"{nm}: (a) owner >=5x MLPs: {'HELD' if r['pred_a'] else 'FAILED'}"
              f" | (b) exclusivity <=2x: {'HELD' if r['pred_b'] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
