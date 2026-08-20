"""WINDOW AT MATCH POSITIONS -- 484: on AVERAGE, every attention
layer in this model tolerates a 4-token read window (482: 17 of 18
cost under 0.09 nats), and the one exception -- layer 5 at +1.112
-- turns out to be the position-0 sink losing its constant, not a
content read (483: allowing position 0 drops it to +0.077).
That sits oddly with the induction band. Induction requires
reading a DISTANT earlier occurrence, and this program closed that
circuit end to end across layers 1-8 (376-408). Yet windowing
those layers costs +0.014 to +0.038 on average.
The likely reconciliation is that induction barely moves AVERAGE
loss -- it matters at match positions, which are a small minority.
Test it by scoring the same sweep where induction actually
operates: positions whose token has appeared before.
REGISTERED PREDICTIONS:
  (a) MATCH POSITIONS PAY: at match positions, at least one layer
      in 1-8 costs >= 0.20 under the 4-token window;
  (b) BAND LOCALISED: the largest match-position cost among
      layers 1-8 falls on a documented induction layer
      (1, 2, 3, 5, 6, 7 or 8);
  (c) SINK IS DIFFERENT: at match positions, layer 5's cost is
      NOT fixed by allowing position 0 (unlike the average case,
      where it dropped 93%)."""


import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'window_at_match_results.json'
NR=16
LAYERS=list(range(18)); K=4

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    import sys
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    def run(LJ,active,k=K,allow0=False):
        tm=tn=0.0; nm_=nn_=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            at=m.transformer.h[LJ].attn
            if active:
                def fh(mo_,args,o_,k=k,at=at):
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
                    mask=torch.tril(torch.ones(T,T,device=DEV))
                    if k is not None:
                        ar=torch.arange(T,device=DEV)
                        win=((ar[:,None]-ar[None,:])<k).float()
                        if allow0:
                            w0=torch.zeros(T,T,device=DEV)
                            w0[:,0]=1.0
                            win=torch.clamp(win+w0,max=1.0)
                        mask=mask*win
                    pat=(sc*sc2)*mask
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
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
    bm,bn=run(0,False)
    res={}
    for LJ in LAYERS:
        pm,pn=run(LJ,True)
        res[LJ]={'match':round(pm-bm,4),'nonmatch':round(pn-bn,4)}
        print(f'layer {LJ}: match {res[LJ]["match"]:+.4f} '
              f'non-match {res[LJ]["nonmatch"]:+.4f}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    p5,_=run(5,True,allow0=True)
    l5_plus0=round(p5-bm,4)
    band=[l for l in range(1,9)]
    pa=any(res[l]['match']>=0.20 for l in band)
    worst=max(band,key=lambda l:res[l]['match'])
    pb=worst in (1,2,3,5,6,7,8)
    pc=l5_plus0>0.15
    out={'baseline_match_ce':round(bm,4),'window':K,
         'dce_by_layer':res,'worst_band_layer':worst,
         'worst_band_match_cost':res[worst]['match'],
         'layer5_match_with_pos0':l5_plus0,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'worst band layer {worst} at '
          f'{res[worst]["match"]:+.4f} | layer5 match with pos0 '
          f'{l5_plus0:+.4f}')
    for nm,v in (('a','a band layer costs >=0.20 at match'),
                 ('b','the worst is a documented induction layer'),
                 ('c','position 0 does NOT fix layer 5 at match')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
