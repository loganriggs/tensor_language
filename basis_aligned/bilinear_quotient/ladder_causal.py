"""LADDER CAUSAL -- 402: relay depth bounded (all nine band heads
>=0.93 at k=2). Cash in causally: run the LIVE model with every
band head's pattern replaced by the pattern computed from the
k=2 reconstructed code (real values, real everything else), and
price it in CE at match positions. This is the end-to-end
statement: the induction triggers of bilin18 are computable from
wte + MLP chain + two attention moves, and the model RUNS on the
computed triggers.
Arms: intact / ladder-trigger (k=2 patterns on all 9 band heads)
/ shuffled-trigger (k=2 chains built with row-shuffled value
sources -- structure preserved, content broken) / deletion
(band heads' pattern zeroed; scale anchor).
REGISTERED PREDICTIONS (dCE at match positions vs intact):
  (a) ladder-trigger costs <= 0.05 (same order as the 4-read
      code's +0.0073, far under deletion's +0.60);
  (b) ladder-trigger <= 10% of deletion's cost;
  (c) shuffled-trigger >= 6x ladder-trigger (content matters);
  (d) off-match positions: ladder-trigger costs <= 0.02 (the
      substitution is surgical)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ladder_causal_results.json'
NR=64; KREP=2
HEADS=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]
BYLAYER={}
for li,hd in HEADS: BYLAYER.setdefault(li,[]).append(hd)

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    maxli=max(l for l,_ in HEADS)
    tot={a:{'m':0.0,'nm':0,'o':0.0,'no':0} for a in
         ('ladder','shuf','delete')}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        B=4
        # match-position mask
        mmask=torch.zeros(B,T,dtype=torch.bool)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                ism=t in last and last[t]+1<q
                last[t]=q
                if ism and q>=8: mmask[b,q]=True
        # pass 1: capture
        mout={}; pre={}
        hs=[]
        for lj in range(maxli):
            def mh(mo,i_,o_,lj=lj): mout[lj]=o_.detach().float()
            def phj(mo_,args,lj=lj): pre[lj]=(args[0],args[1])
            hs.append(m.transformer.h[lj].mlp
                      .register_forward_hook(mh))
            hs.append(m.transformer.h[lj].attn
                      .register_forward_pre_hook(phj))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ce0=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                            reduction='none').view(B,T).cpu()
        for h in hs: h.remove()
        def a_relay(j,src,shuf=False):
            at=m.transformer.h[j].attn
            Xj,v1j=pre[j]
            Xs=F.rms_norm(src,(D,)) \
                .to(m.transformer.wte.weight.dtype)
            if shuf: Xs=Xs[torch.tensor([1,2,3,0])]
            v=at.c_v(Xs).view(B,T,9,128)
            vm=v if v1j is None else \
                (1-at.lamb)*v+at.lamb*v1j.view_as(v)
            cos,sin=at.rotary(at.c_q(Xj).view(B,T,9,128))
            qf=F.rms_norm(at.c_q(Xj).view(B,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(Xj).view(B,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(Xj).view(B,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(Xj).view(B,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            patm=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',patm,vm.float())
            return at.c_proj(z.transpose(1,2).contiguous()
                             .view(B,T,-1).to(Xj.dtype)).float()
        def build(vals,shuf=False):
            xr=E.clone(); out={}; resid={}
            for lj in range(maxli+1):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                resid[lj]=xr.clone()
                out[lj]=F.rms_norm(xr,(D,))
                if lj<maxli:
                    if vals is not None:
                        xr=xr+a_relay(lj,vals[lj],shuf)
                    xr=xr+mout[lj]
            return out,resid
        def chain_codes(shuf=False):
            _,resid=build(None)
            codes=None
            for k in range(1,KREP+1):
                codes,resid=build(resid,shuf and k==KREP)
            return codes
        codesL=chain_codes(False)
        codesS=chain_codes(True)
        # pass 2 variants: substitute band-head patterns
        def run_sub(codes,delete=False):
            hooks=[]
            for li,hds in BYLAYER.items():
                at=m.transformer.h[li].attn
                def fh(mo_,args,out,at=at,li=li,hds=hds):
                    y,v1r=out
                    X=args[0]
                    v1x=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1x.view_as(v)
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
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    patm=(sc*sc2)*tril
                    z=torch.einsum('bhqk,bkhd->bhqd',patm,
                                   vm.float())
                    for hd in hds:
                        if delete:
                            z[:,hd]=0; continue
                        c=codes[li]
                        a9,b9=hd*128,(hd+1)*128
                        fq1=F.rms_norm(c@at.c_q.weight.float()
                                       [a9:b9].T,(128,))
                        fk1=F.rms_norm(c@at.c_k.weight.float()
                                       [a9:b9].T,(128,))
                        fq2=F.rms_norm(c@at.c_q2.weight.float()
                                       [a9:b9].T,(128,))
                        fk2=F.rms_norm(c@at.c_k2.weight.float()
                                       [a9:b9].T,(128,))
                        fq1=are(fq1[:,:,None],cos,sin)[:,:,0]
                        fk1=are(fk1[:,:,None],cos,sin)[:,:,0]
                        fq2=are(fq2[:,:,None],cos,sin)[:,:,0]
                        fk2=are(fk2[:,:,None],cos,sin)[:,:,0]
                        fpat=(torch.einsum('bqd,bkd->bqk',fq1,fk1)
                              *torch.einsum('bqd,bkd->bqk',
                                            fq2,fk2))/(128*128)
                        fpat=fpat*tril
                        z[:,hd]=torch.einsum('bqk,bkd->bqd',
                                             fpat,vm[:,:,hd]
                                             .float())
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hooks.append(at.register_forward_hook(fh))
            xx=F.rms_norm(m.transformer.wte(idx),(D,))
            xr=xx; x0r=xx; v1r=None
            for blk in m.transformer.h: xr,v1r=blk(xr,v1r,x0r)
            lgr=(30*torch.tanh(m.lm_head(F.rms_norm(xr,(D,)))
                               /30)).float()
            cer=F.cross_entropy(lgr.view(-1,lgr.size(-1)),tg,
                                reduction='none').view(B,T).cpu()
            for h in hooks: h.remove()
            return cer
        for a,cer in (('ladder',run_sub(codesL)),
                      ('shuf',run_sub(codesS)),
                      ('delete',run_sub(None,delete=True))):
            d=cer-ce0
            tot[a]['m']+=float(d[mmask].sum())
            tot[a]['nm']+=int(mmask.sum())
            tot[a]['o']+=float(d[~mmask].sum())
            tot[a]['no']+=int((~mmask).sum())
        print(f'batch {i} done',flush=True)
    out={}
    for a,s in tot.items():
        out[a]={'dce_match':round(s['m']/max(s['nm'],1),4),
                'dce_off':round(s['o']/max(s['no'],1),4)}
        print(f"{a}: match {out[a]['dce_match']} off "
              f"{out[a]['dce_off']}",flush=True)
    lm=out['ladder']['dce_match']; dm=out['delete']['dce_match']
    sm=out['shuf']['dce_match']
    pa=lm<=0.05
    pb=lm<=0.10*max(dm,1e-6)
    pc=sm>=6*max(lm,1e-6)
    pd=out['ladder']['dce_off']<=0.02
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc),'pred_d':bool(pd)})
    for nm,v in (('a','ladder-trigger <=0.05 at match'),
                 ('b','<=10% of deletion'),
                 ('c','shuffled >=6x ladder'),
                 ('d','off-match <=0.02')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
