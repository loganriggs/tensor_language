"""RELAY EDGE CAUSAL -- 407: relay_heads named the movers --
a4.h7 (prev 0.765) relays for 5.5; a6.h3 (prev 0.468) relays for
ALL THREE deep heads 7.3/8.3/8.4. Causal test in the LIVE model:
zero one head's contribution and measure (i) dCE at match
positions, (ii) each band head's top-read shift rate (fraction of
match positions where its real argmax read changes). Controls:
zero a low-relay-lift head from the same layer (a6.h0 lift
-0.008; a4.h2 lift -0.001).
REGISTERED PREDICTIONS:
  (a) SELECTIVE PATTERN DAMAGE: deleting a6.h3 shifts the deep
      trio's top reads at >=3x the rate it shifts the early
      band's (1.4, 2.5);
  (b) deleting a4.h7 shifts 5.5's top reads at >=3x its effect
      on the early band;
  (c) matched-layer control: a6.h3's match-position dCE >= 3x
      a6.h0's;
  (d) report: dCE at match vs off-match for all four deletions
      (relay deletion should lean toward match positions)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'relay_edge_causal_results.json'
NR=32
BAND=[(1,4),(2,5),(5,5),(7,3),(8,3),(8,4)]
ARMS={'a6h3':(6,3),'a6h0':(6,0),'a4h7':(4,7),'a4h2':(4,2)}

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    stats={a:{'m':0.0,'nm':0,'o':0.0,'no':0,
              'shift':{f'{li}.{hd}':[0,0] for li,hd in BAND}}
           for a in ARMS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        mmask=torch.zeros(B,T,dtype=torch.bool)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                ism=t in last and last[t]+1<q
                last[t]=q
                if ism and q>=8: mmask[b,q]=True
        def run(kill=None):
            caps={}
            hs=[]
            for li,hd in BAND:
                def ph(mo_,args,li=li): caps[li]=args[0]
                hs.append(m.transformer.h[li].attn
                          .register_forward_pre_hook(ph))
            if kill is not None:
                kl,kh=kill
                at=m.transformer.h[kl].attn
                def fh(mo_,args,o_,at=at,kh=kh):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                    kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                    qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                    q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),
                                  (128,))
                    k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),
                                  (128,))
                    q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',
                                     q2.float(),k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    z[:,kh]=0
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            pats={}
            for li,hd in BAND:
                at=m.transformer.h[li].attn
                X=caps[li]
                cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                qf=F.rms_norm(at.c_q(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                kf=F.rms_norm(at.c_k(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                qf=are(qf[:,:,None],cos,sin)[:,:,0]
                kf=are(kf[:,:,None],cos,sin)[:,:,0]
                q2=are(q2[:,:,None],cos,sin)[:,:,0]
                k2=are(k2[:,:,None],cos,sin)[:,:,0]
                tril=torch.tril(torch.ones(T,T,device=DEV))
                pats[f'{li}.{hd}']=(torch.einsum(
                    'bqd,bkd->bqk',qf.float(),kf.float())
                    *torch.einsum('bqd,bkd->bqk',q2.float(),
                                  k2.float()))*tril
            for h in hs: h.remove()
            return ce,pats
        ce0,pat0=run(None)
        for a,kill in ARMS.items():
            ce1,pat1=run(kill)
            d=ce1-ce0
            st=stats[a]
            st['m']+=float(d[mmask].sum()); st['nm']+=int(mmask.sum())
            st['o']+=float(d[~mmask].sum())
            st['no']+=int((~mmask).sum())
            for k in pat0:
                sh=st['shift'][k]
                for b in range(B):
                    for q in range(T):
                        if not mmask[b,q]: continue
                        k0=int(pat0[k][b,q,:q].abs().argmax()) \
                            if q>0 else 0
                        k1=int(pat1[k][b,q,:q].abs().argmax()) \
                            if q>0 else 0
                        sh[0]+=int(k0!=k1); sh[1]+=1
        print(f'batch {i} done',flush=True)
    out={}
    for a,st in stats.items():
        out[a]={'dce_match':round(st['m']/max(st['nm'],1),4),
                'dce_off':round(st['o']/max(st['no'],1),4),
                'shift':{k:round(v[0]/max(v[1],1),3)
                         for k,v in st['shift'].items()}}
        print(f"{a}: dce match {out[a]['dce_match']} off "
              f"{out[a]['dce_off']} | shifts {out[a]['shift']}",
              flush=True)
    deep=['7.3','8.3','8.4']; early=['1.4','2.5']
    s63=out['a6h3']['shift']
    dsh=sum(s63[k] for k in deep)/3
    esh=sum(s63[k] for k in early)/2
    pa=dsh>=3*max(esh,1e-4)
    s47=out['a4h7']['shift']
    pb=s47['5.5']>=3*max((s47['1.4']+s47['2.5'])/2,1e-4)
    pc=abs(out['a6h3']['dce_match'])>= \
        3*max(abs(out['a6h0']['dce_match']),1e-4)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','a6.h3 shifts deep trio >=3x early'),
                 ('b','a4.h7 shifts 5.5 >=3x early'),
                 ('c','a6.h3 match-dCE >=3x a6.h0')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
