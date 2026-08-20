"""PUNCT ORACLE CEILING -- 461: the rank-1 repair FAILED (460) and
failed informatively: subtracting the fitted over-continuation
direction made CE WORSE at every scale, and worse at punctuation
(+0.057 at the smallest scale) than a random direction of the same
norm (+0.009). So the bias is NOT a fixed additive vector that can
be subtracted -- unlike the layer-5 sink constant, which IS one
(435). The over-continuation is input-dependent: it lives in how
those components respond at boundary positions, not in a constant
they add.
So measure the headroom instead of guessing a fix. Two arms on
FRESH FineWeb rows:
  oracle  : mean-ablate the five helping components ONLY at
            positions whose true next token is punctuation
            (an upper bound on what any detector could buy)
  proxy   : the same gate driven by a NON-oracle signal available
            at inference -- the current token ends a word
            (next token starts with a space or is punctuation is
            NOT knowable; we use "current token is alphabetic and
            the one before it started with a space", a purely
            causal cue)
REGISTERED PREDICTIONS:
  (a) CEILING EXISTS: the oracle arm lowers overall CE on fresh
      rows (dCE < 0);
  (b) SIZE: the oracle gain at punctuation positions is >= 0.05
      nats;
  (c) PRACTICAL: the causal proxy captures >= 30% of the oracle's
      overall gain -- if not, the deficiency is real but not
      cheaply fixable, which is the honest finding."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_oracle_ceiling_results.json'
HELPERS=['a3','a6','a7','a8','m7']
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    ispunct=lambda t:(lambda s: bool(s) and
                      not any(c.isalnum() for c in s))(
                          cl.d1(int(t)).strip())
    fresh=cl.fineweb_rows(NFRESH)
    pm=torch.zeros(NFRESH,T,dtype=torch.bool)
    proxy=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            pm[r,q]=ispunct(int(fresh[r,q+1]))
            cur=cl.d1(int(fresh[r,q]))
            prev=cl.d1(int(fresh[r,q-1])) if q>0 else ''
            proxy[r,q]=(cur.strip().isalpha()
                        and prev.startswith(' '))
    print(f'punct targets {int(pm.sum())}, proxy gate fires '
          f'{int(proxy.sum())} of {pm.numel()}',flush=True)
    def run(gate):
        tp=tn=0.0; np_=nn_=0
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=[]
            if gate is not None:
                gm=gate[i:i+4].to(DEV)[:,:,None]
                for k in HELPERS:
                    mu=mus[k].to(DEV)
                    if k[0]=='a':
                        def fh(mo,i_,o_,mu=mu,gm=gm):
                            y,v1=o_
                            rep=mu.expand_as(y).to(y.dtype)
                            return (torch.where(gm,rep,y),v1)
                    else:
                        def fh(mo,i_,o_,mu=mu,gm=gm):
                            rep=mu.expand_as(o_).to(o_.dtype)
                            return torch.where(gm,rep,o_)
                    hs.append(MODS[k].register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            mk=pm[i:i+4]
            tp+=float(ce[mk].sum()); np_+=int(mk.sum())
            tn+=float(ce[~mk].sum()); nn_+=int((~mk).sum())
            for h in hs: h.remove()
        return tp/max(np_,1),tn/max(nn_,1)
    bp,bn=run(None)
    op,on=run(pm)
    xp,xn=run(proxy)
    w=int(pm.sum())/pm.numel()
    def allce(p,n): return p*w+n*(1-w)
    base=allce(bp,bn); orc=allce(op,on); prx=allce(xp,xn)
    og=orc-base; pg=prx-base
    pa=og<0
    pb=(bp-op)>=0.05
    pc=(og<0 and pg<0 and (pg/og)>=0.30)
    out={'baseline':{'punct':round(bp,4),'nonpunct':round(bn,4),
                     'all':round(base,4)},
         'oracle':{'punct':round(op,4),'nonpunct':round(on,4),
                   'dce_all':round(og,4),
                   'dce_punct':round(op-bp,4)},
         'proxy':{'punct':round(xp,4),'nonpunct':round(xn,4),
                  'dce_all':round(pg,4),
                  'dce_punct':round(xp-bp,4),
                  'fraction_of_oracle':round(pg/og,3)
                      if og<0 else None},
         'proxy_fire_rate':round(float(proxy.float().mean()),3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'baseline all {base:.4f} | oracle {og:+.4f} '
          f'(punct {op-bp:+.4f}) | proxy {pg:+.4f}')
    for nm,v in (('a','oracle lowers CE'),
                 ('b','oracle gain at punct >=0.05'),
                 ('c','causal proxy captures >=30%')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
