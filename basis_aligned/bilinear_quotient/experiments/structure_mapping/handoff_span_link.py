"""HANDOFF -> SPAN CAUSAL LINK -- 290 found the attn5->mlp5 channel
overlaps the private span 40-50x above chance and reads as a
clause-boundary axis. Causal closure: cut ONLY the top-4 channel
directions at mlp5's input and measure (i) the variance of mlp6's
output span coefficients (does the private code collapse?), (ii) which
function classes pay the CE damage (clause classes predicted: sentend,
comma, bclose, newline). Control: cut 4 random directions.
REGISTERED PREDICTIONS:
  (a) mlp6 span-coefficient variance drops >= 40% under the cut;
      random-4 control changes it <= 5%;
  (b) the four clause classes carry a damage share >= 1.5x their
      position share (damage concentrates where the semantics say);
  (c) per-class damage table reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'handoff_span_link_results.json'
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
    var_base=float((Y6@S6).var(0).sum())
    g=torch.Generator(device=DEV).manual_seed(0)
    R4=orth(torch.randn(D,4,device=DEV,generator=g))
    def span_var(P):
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
        Yb=torch.cat(m6b)
        return float((Yb@S6).var(0).sum())
    var_cut=span_var(V4); var_ctl=span_var(R4)
    drop=1-var_cut/var_base; dctl=abs(1-var_ctl/var_base)
    print(f'span variance: base {var_base:.1f} cut {var_cut:.1f} '
          f'({drop:+.1%}) | random-4 {var_ctl:.1f} ({dctl:.1%})',
          flush=True)
    # per-class CE damage on eval window
    clsC=classify(R0,R1).reshape(-1).to(DEV)
    def ce_vec(P):
        hbb=blk.register_forward_pre_hook(hb)
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
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for b2 in m.transformer.h:
                x,v1=b2(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    c0=ce_vec(None); c1=ce_vec(V4)
    dmg=c1-c0
    tot=float(dmg.sum())
    CLAUSE={'sentend','comma','bclose','newline'}
    tab={}
    csh=0.0; psh=0.0
    for k,nm in enumerate(CLS):
        sel=clsC==k
        share=float(dmg[sel].sum())/tot
        pos=float(sel.float().mean())
        tab[nm]={'damage_share':round(share,3),
                 'position_share':round(pos,3)}
        if nm in CLAUSE: csh+=share; psh+=pos
        print(f'{nm:8s}: dmg {share:5.1%} pos {pos:5.1%}',flush=True)
    pa=drop>=0.40 and dctl<=0.05
    pb=csh>=1.5*psh
    out={'span_var_drop':round(drop,3),'ctl_change':round(dctl,3),
         'total_damage':round(tot/len(dmg),4),'per_class':tab,
         'clause_damage_share':round(csh,3),
         'clause_position_share':round(psh,3),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'clause damage share {csh:.1%} vs position share {psh:.1%}')
    print(f"(a) span var drops >=40%, ctl <=5%: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) clause damage >= 1.5x position share: "
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
