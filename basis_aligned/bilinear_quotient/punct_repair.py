"""PUNCT REPAIR -- 460: the model over-continues at phrase
boundaries (457, n=100: intact top-1 is a wrong continuation 75%
of the time, ablation suppresses exactly that competitor by
-0.108 against +1.2e-06 for a random token), and five components
carry it (458/459: after recentering, competitor-minus-target
logit margins of +3.15, +4.90, +6.95, +7.61, +8.91 against the
clean control's +0.04 -- a hundred-fold separation).
Understanding a deficiency well enough to REPAIR it is the
strongest test available. Build a rank-1 correction with no
oracle: take the mean write direction of the five helping
components at helped-punctuation sites minus their mean write
elsewhere -- the "over-continuation direction" -- and subtract a
scaled multiple of it from the residual at EVERY position, on
FRESH FineWeb rows the direction was not fitted on.
Scales swept: 0.05, 0.1, 0.2, 0.4 of the residual norm. Control:
a random direction of identical norm at the same scales.
REGISTERED PREDICTIONS:
  (a) REPAIR: at its best scale the correction lowers overall CE
      on fresh rows (dCE < 0);
  (b) TARGETED: the improvement at punctuation targets is >= 3x
      the improvement elsewhere;
  (c) CONTROL: the random direction does not lower CE at any
      scale."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_repair_results.json'
TAGS=['r.18.2.0','r.13.2.1','r.11.1.2']
HELPERS=['a3','a6','a7','a8','m7']
SCALES=[0.05,0.1,0.2,0.4]
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    rows=cl.rows()
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    ispunct=lambda t:(lambda s: bool(s) and
                      not any(c.isalnum() for c in s))(
                          cl.d1(int(t)).strip())
    # --- fit the direction on CENSUS rows (never on fresh) ---
    def a7hooks():
        mu=mus['a7'].to(DEV)
        def fh(mo,i_,o_,mu=mu):
            y,v1=o_
            return (mu.expand_as(y).to(y.dtype),v1)
        return [MODS['a7'].register_forward_hook(fh)]
    d=cl.ce_sweep(a7hooks())-cl.base_ce()
    mem=sorted({g for t in TAGS
                for g in cl.leaf(t)['member'].tolist()})
    sites=[g for g in mem
           if ispunct(int(rows[g//256,g%256+1])) and float(d[g])<0]
    byrow={}
    for g in sites: byrow.setdefault(g//256,[]).append(g%256)
    rowsel=sorted(byrow)[:40]
    S=torch.zeros(D,device=DEV); ns=0
    E_=torch.zeros(D,device=DEV); ne=0
    for i in range(0,len(rowsel),4):
        rid=torch.tensor(rowsel[i:i+4])
        bb=rows[rid,:257].to(DEV); idx=bb[:,:-1].contiguous()
        outs={}
        hs=[MODS[k].register_forward_hook(
            (lambda k: lambda mo,i_,o_: outs.__setitem__(
                k,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))(k)) for k in HELPERS]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        tot=sum(outs[k] for k in HELPERS)
        for b,r in enumerate(rid.tolist()):
            ps=set(byrow[r])
            for pos in range(T):
                if pos in ps: S+=tot[b,pos]; ns+=1
                else: E_+=tot[b,pos]; ne+=1
    v=(S/max(ns,1))-(E_/max(ne,1))
    u=v/v.norm().clamp_min(1e-6)
    print(f'over-continuation direction fitted on {ns} sites '
          f'(norm {float(v.norm()):.1f})',flush=True)
    g=torch.Generator(device=DEV).manual_seed(19)
    r0=torch.randn(D,generator=g,device=DEV); r0=r0/r0.norm()
    # --- evaluate on FRESH rows ---
    fresh=cl.fineweb_rows(NFRESH)
    pm=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            pm[r,q]=ispunct(int(fresh[r,q+1]))
    def run(vec,scale):
        tot_p=tot_n=0.0; np_=nn_=0
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            hs=[]
            if vec is not None:
                def ph(mo_,a_,vec=vec,scale=scale):
                    xx=a_[0]
                    nrm=xx.float().norm(dim=-1,keepdim=True)
                    return (xx-(scale*nrm*vec[None,None,:])
                            .to(xx.dtype),)+tuple(a_[1:])
                hs.append(m.transformer.h[8]
                          .register_forward_pre_hook(ph))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(-1,T).cpu()
            mk=pm[i:i+4]
            tot_p+=float(ce[mk].sum()); np_+=int(mk.sum())
            tot_n+=float(ce[~mk].sum()); nn_+=int((~mk).sum())
            for h in hs: h.remove()
        return tot_p/max(np_,1),tot_n/max(nn_,1)
    bp,bn=run(None,0)
    res={'baseline':{'punct':round(bp,4),'nonpunct':round(bn,4)}}
    print(f'baseline: punct {bp:.4f} nonpunct {bn:.4f}',flush=True)
    for nm,vec in (('repair',u),('random',r0)):
        res[nm]={}
        for s in SCALES:
            p,n=run(vec,s)
            res[nm][str(s)]={'dce_punct':round(p-bp,4),
                             'dce_nonpunct':round(n-bn,4),
                             'dce_all':round(((p-bp)*1+(n-bn)*6)/7,4)}
            print(f"{nm} scale {s}: punct {p-bp:+.4f} nonpunct "
                  f"{n-bn:+.4f}",flush=True)
    best=min(res['repair'],key=lambda s:res['repair'][s]['dce_all'])
    br=res['repair'][best]
    pa=br['dce_all']<0
    pb=(br['dce_punct']<0 and
        abs(br['dce_punct'])>=3*abs(br['dce_nonpunct']))
    pc=all(res['random'][s]['dce_all']>=0 for s in res['random'])
    out={'results':res,'best_scale':best,'fitted_on_sites':ns,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'best repair scale {best}: {br}')
    for nm,v_ in (('a','repair lowers CE on fresh rows'),
                  ('b','improvement concentrates at punctuation'),
                  ('c','random direction does not help')):
        print(f"({nm}) {v_}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
