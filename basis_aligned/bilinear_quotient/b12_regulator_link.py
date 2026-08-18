"""BILIN12 REGULATOR LINK -- 300: bilin12's narrow mid junction (block
5, +3.52, top-4 = 85%) sits one block AFTER its private writer (mlp4,
the fraction-0.33 universal), so if the family mechanisms correspond it
should CONSUME the private code rather than regulate it. Tests: (1)
geometric overlap of the junction's top-4 channel with mlp4's output
span (8-dim, random floor 8/768 = 0.0104); (2) per-class CE damage of
the 4-direction cut; (3) control: random-4 overlap and cut.
REGISTERED PREDICTIONS:
  (a) channel-vs-mlp4-span mean cos^2 >= 0.07 (>= ~7x floor);
  (b) random-4 cut <= 10% of the top-4 cut cost;
  (c) per-class damage table reported (informational)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'b12_regulator_link_results.json'
R0,R1=120,300; CA=300

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('bilin12', device=DEV)
    NL=len(m2.transformer.h); D=m2.transformer.wte.weight.shape[1]
    print(f'bilin12: {NL} layers, d={D}',flush=True)
    cur={}
    def evalCE(hooks):
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m2.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return float(torch.cat(ces).mean())
    def mk_cut(li,P=None):
        blk=m2.transformer.h[li]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            if P is None:
                return (alt.to(x.dtype),)+args[1:]
            d=x.float()-alt
            return ((x.float()-(d@P)@P.T).to(x.dtype),)+args[1:]
        return [blk.register_forward_pre_hook(hb),
                blk.mlp.register_forward_pre_hook(hm)]
    from circuit_dictionary import classify, CLS
    base=evalCE([])
    LI=5
    blk=m2.transformer.h[LI]
    ds_=[]; m4=[]
    def hb2(mo,args):
        x,v1,x0=args
        cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
    h1=blk.register_forward_pre_hook(hb2)
    h2=blk.mlp.register_forward_pre_hook(
        lambda mo,args: ds_.append((args[0].float()
            -F.rms_norm(cur['xm'].float(),(D,))).reshape(-1,D)))
    h3=m2.transformer.h[4].mlp.register_forward_hook(
        lambda mo,i_,o_: m4.append(o_.detach().float().reshape(-1,D)))
    for i in range(CA,CA+80,4):
        bb=FW[i:i+4,:257].to(DEV)
        m2(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in (h1,h2,h3): h.remove()
    Dd=torch.cat(ds_); Y4=torch.cat(m4)
    _,_,Vh=torch.linalg.svd(Dd[:40000],full_matrices=False)
    V4=orth(Vh[:4].T)
    _,_,V4h=torch.linalg.svd((Y4-Y4.mean(0))[:40000],full_matrices=False)
    S4=orth(V4h[:8].T)
    ov=float((V4.T@S4).pow(2).sum())/4
    g=torch.Generator(device=DEV).manual_seed(0)
    Rr=orth(torch.randn(D,4,device=DEV,generator=g))
    ovr=float((Rr.T@S4).pow(2).sum())/4
    print(f'channel-vs-mlp4-span overlap {ov:.3f} (random {ovr:.3f}, '
          f'floor 0.0104)',flush=True)
    def cevec(P):
        hbb=blk.register_forward_pre_hook(hb2)
        hs=[hbb]
        if P is not None:
            def hm(mo,args):
                x=args[0]
                alt=F.rms_norm(cur['xm'].float(),(D,))
                d=x.float()-alt
                return ((x.float()-(d@P)@P.T).to(x.dtype),)+args[1:]
            hs.append(blk.mlp.register_forward_pre_hook(hm))
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for b2 in m2.transformer.h:
                x,v1=b2(x,v1,x0)
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    c0=cevec(None); c1=cevec(V4); cr=cevec(Rr)
    cut=float((c1-c0).mean()); ctl=float((cr-c0).mean())
    print(f'top-4 cut {cut:+.4f} | random-4 {ctl:+.4f}',flush=True)
    clsC=classify(R0,R1).reshape(-1).to(DEV)
    dmg=c1-c0; tot=float(dmg.sum())
    tab={}
    for k,nm in enumerate(CLS):
        sel=clsC==k
        tab[nm]={'damage_share':round(float(dmg[sel].sum())/tot,3),
                 'position_share':round(float(sel.float().mean()),3)}
        print(f'{nm:8s}: dmg {tab[nm]["damage_share"]:5.1%} pos '
              f'{tab[nm]["position_share"]:5.1%}',flush=True)
    pa=ov>=0.07; pb=ctl<=0.1*max(cut,1e-4)
    out={'overlap':round(ov,3),'overlap_rand':round(ovr,3),
         'cut':round(cut,4),'ctl':round(ctl,4),'per_class':tab,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) overlap >= 0.07: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random-4 <= 10%: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
