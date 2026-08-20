"""INDUCTION REDUNDANCY -- 486: the fork resolved in favour of the
program's flagship claim (485). Windowing the nine induction-band
heads to four tokens costs +0.318 at match positions -- 75% of the
+0.426 that deleting them outright costs -- against +0.180 for
nine random control heads. So their distant reads really do carry
most of their function.
But that sits against 484, where windowing whole LAYERS containing
those same heads cost between -0.06 and +0.01 at match. Each of
those layers contains one band head plus eight others. The natural
explanation is REDUNDANCY ACROSS THE BAND: losing one head's
distant reads is covered by the other eight, and only removing
them together bites.
Measure it: window the band heads CUMULATIVELY, one at a time in a
fixed order, and watch how the cost grows.
REGISTERED PREDICTIONS:
  (a) SUPERLINEAR: windowing all nine costs >= 3x the MEAN cost of
      windowing each head singly;
  (b) SINGLES ARE CHEAP: at least 7 of the 9 heads cost <= 0.05 at
      match positions when windowed alone;
  (c) CONTROL: the same cumulative curve over nine random non-band
      heads is closer to linear (its all-nine cost is < 3x its
      singles mean)."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'induction_redundancy_results.json'
NR=16
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    g=torch.Generator().manual_seed(17)
    ctrl=[]
    while len(ctrl)<9:
        lj=int(torch.randint(0,18,(1,),generator=g))
        hd=int(torch.randint(0,9,(1,),generator=g))
        if (lj,hd) not in IND and (lj,hd) not in ctrl:
            ctrl.append((lj,hd))
    print(f'control heads: {ctrl}',flush=True)
    def run(heads,mode):
        byl={}
        for lj,hd in heads: byl.setdefault(lj,[]).append(hd)
        tm=tn=0.0; nm_=nn_=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            for lj,hds in byl.items():
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,hds=hds,mode=mode):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=r2(at.c_q),r2(at.c_k)
                    q2,k2=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    ar=torch.arange(T,device=DEV)
                    win=tril*((ar[:,None]-ar[None,:])<K).float()
                    pat=(sc*sc2)*tril
                    patw=(sc*sc2)*win
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                    zw=torch.einsum('bhqk,bkhd->bhqd',patw,
                                    vm.float())
                    for h in hds:
                        z[:,h]=0 if mode=='delete' else zw[:,h]
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            mk=torch.zeros(B,T,dtype=torch.bool)
            for b in range(B):
                toks=ROWS[i+b,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]
                    if t in last and last[t]+1<q and q>=8:
                        mk[b,q]=True
                    last[t]=q
            tm+=float(ce[mk].sum()); nm_+=int(mk.sum())
            tn+=float(ce[~mk].sum()); nn_+=int((~mk).sum())
            for h in hs: h.remove()
        return tm/max(nm_,1),tn/max(nn_,1)
    bm,bn=run([],'window')
    def curve(heads,label):
        singles=[]; cum=[]
        for j,h in enumerate(heads):
            pm,_=run([h],'window')
            singles.append(round(pm-bm,4))
            pm2,_=run(heads[:j+1],'window')
            cum.append(round(pm2-bm,4))
            print(f'{label} +{h}: single {singles[-1]:+.4f} '
                  f'cumulative {cum[-1]:+.4f}',flush=True)
        return singles,cum
    si,ci=curve(IND,'ind')
    sc,cc=curve(ctrl,'ctrl')
    mi=sum(si)/len(si); mc=sum(sc)/len(sc)
    pa=ci[-1]>=3*max(mi,1e-6)
    pb=sum(1 for v in si if v<=0.05)>=7
    pc=cc[-1]<3*max(mc,1e-6)
    res={'ind_singles':si,'ind_cumulative':ci,
         'ctrl_singles':sc,'ctrl_cumulative':cc,
         'ind_singles_mean':round(mi,4),
         'ctrl_singles_mean':round(mc,4),
         'ind_ratio_all_over_mean':round(ci[-1]/max(mi,1e-6),2),
         'ctrl_ratio_all_over_mean':round(cc[-1]/max(mc,1e-6),2)}
    out={'baseline_match_ce':round(bm,4),**res,
         'control_heads':ctrl,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f"ind: singles mean {mi:+.4f}, all nine {ci[-1]:+.4f} "
          f"(ratio {res['ind_ratio_all_over_mean']}) | ctrl ratio "
          f"{res['ctrl_ratio_all_over_mean']}")
    for nm,v in (('a','band is superlinear (>=3x singles mean)'),
                 ('b','>=7 of 9 singles cost <=0.05'),
                 ('c','control curve is closer to linear')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
