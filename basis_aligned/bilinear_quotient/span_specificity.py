"""SPAN SPECIFICITY -- 291 found cutting the 4 channel directions makes
mlp6's span-coefficient variance explode 4.6x. Is the explosion IN the
8-dim span specifically, or generic destabilization of mlp6's whole
output? Compare variance ratios (cut/base) for: the 8-dim span, the
next 8 PCA directions (9-16), 8 random directions, and the full output.
REGISTERED PREDICTIONS:
  (a) span ratio >= 2x the full-output ratio (the explosion targets the
      contested code) -- if FAILED, the destabilization is generic and
      290's geometric overlap was incidental;
  (b) random-8-direction ratio ~ full-output ratio (sanity)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'span_specificity_results.json'
CA,CB=300,512; R0,R1=120,300; LI=5

@torch.no_grad()
def main():
    t0=time.time()
    blk=m.transformer.h[LI]
    cur={}
    def hb(mo,args):
        x,v1,x0=args
        cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
    ds=[]; m6=[]
    h1=blk.register_forward_pre_hook(hb)
    h2=blk.mlp.register_forward_pre_hook(
        lambda mo,args: ds.append((args[0].float()
            -F.rms_norm(cur['xm'].float(),(D,))).reshape(-1,D)))
    h3=m.transformer.h[6].mlp.register_forward_hook(
        lambda mo,i_,o_: m6.append(o_.detach().float().reshape(-1,D)))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in (h1,h2,h3): h.remove()
    Dd=torch.cat(ds)
    _,_,Vh=torch.linalg.svd(Dd[:40000], full_matrices=False)
    V4=orth(Vh[:4].T)
    Y6=torch.cat(m6)
    _,_,V6h=torch.linalg.svd((Y6-Y6.mean(0))[:40000],full_matrices=False)
    S6=orth(V6h[:8].T)
    S16=orth(V6h[8:16].T)
    g=torch.Generator(device=DEV).manual_seed(0)
    R8=orth(torch.randn(D,8,device=DEV,generator=g))
    R4=orth(torch.randn(D,4,device=DEV,generator=g))
    def stats(Y):
        return {'span':float((Y@S6).var(0).sum()),
                'next8':float((Y@S16).var(0).sum()),
                'rand8':float((Y@R8).var(0).sum()),
                'full':float(Y.var(0).sum())}
    base=stats(Y6)
    def collect(P):
        m6b=[]
        hbb=blk.register_forward_pre_hook(hb)
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            d=x.float()-alt
            return ((x.float()-(d@P)@P.T).to(x.dtype),)+args[1:]
        hmm=blk.mlp.register_forward_pre_hook(hm)
        hc=m.transformer.h[6].mlp.register_forward_hook(
            lambda mo,i_,o_: m6b.append(o_.detach().float()
                                        .reshape(-1,D)))
        for i in range(CA,CB,8):
            bb=FW[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        for h in (hbb,hmm,hc): h.remove()
        return stats(torch.cat(m6b))
    cut=collect(V4)
    rat={k:cut[k]/base[k] for k in base}
    for k in base:
        print(f'{k:6s}: base {base[k]:.0f} cut {cut[k]:.0f} '
              f'ratio {rat[k]:.2f}',flush=True)
    pa=rat['span']>=2*rat['full']
    pb=abs(rat['rand8']-rat['full'])<=0.5*rat['full']
    out={'ratios':{k:round(v,3) for k,v in rat.items()},
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) span ratio >= 2x full ({rat['span']:.2f} vs "
          f"{rat['full']:.2f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) rand8 ~ full: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
