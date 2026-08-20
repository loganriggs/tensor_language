"""SHIFT METRIC AUDIT -- 422: the crossover (421) found BOTH the
pattern swap and the value swap as damaging as deletion
(0.33/0.20/0.22 and 0.336/0.199/0.19 vs 0.336/0.191/0.185). Two
readings: either a6.h3's downstream role needs the exact
(pattern x values) product, or the top-read-shift METRIC saturates
under any perturbation of that magnitude. Audit before concluding
(house rule: a null that passes disqualifies the headline).
Arms at layer 6 head 3: zero / patswap / valswap / permute (z3's
positions shuffled -- same norm, same value distribution, wrong
alignment) / gauss (random write matched to ||z3|| per position).
Metrics: (i) trio top-read shift, as before; (ii) rank correlation
of the trio's full read distribution with intact (graceful); (iii)
dCE at match positions (function, not bookkeeping).
REGISTERED PREDICTIONS:
  (a) SATURATION: the permute control's trio shift >= 0.15 -- if
      HELD, the argmax-shift metric is non-specific at this
      magnitude and every shift number in 416-421 must be read as
      "a perturbation of size ||h3 write||", not as content
      attribution (caveat to be stated in the ledger and report);
  (b) CE DISCRIMINATES: zero's match dCE exceeds gauss's by
      >= 0.05 nats (the functional metric still separates arms);
  (c) GRACEFUL METRIC: patswap's read-distribution rank
      correlation with intact exceeds gauss's by >= 0.10."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'shift_metric_audit_results.json'
NR=16
TRIO=[(7,3),(8,3),(8,4)]
ARMS=['zero','patswap','valswap','permute','gauss']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={a:{'shift':{f'{li}.{hd}':[0,0] for li,hd in TRIO},
             'corr':{f'{li}.{hd}':[] for li,hd in TRIO},
             'ce':0.0,'nce':0} for a in ARMS}
    g=torch.Generator(device=DEV).manual_seed(11)
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
        def run(mode):
            caps={}; hs=[]
            for li,_ in TRIO:
                def ph(mo_,args,li=li): caps[li]=args[0]
                hs.append(m.transformer.h[li].attn
                          .register_forward_pre_hook(ph))
            if mode is not None:
                at=m.transformer.h[6].attn
                def fh(mo_,args,o_,at=at,mode=mode):
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
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    z3=z[:,3].clone()
                    if mode=='zero': z[:,3]=0
                    elif mode=='patswap':
                        z[:,3]=torch.einsum('bqk,bkd->bqd',
                            pat[:,0],vm[:,:,3].float())
                    elif mode=='valswap':
                        z[:,3]=torch.einsum('bqk,bkd->bqd',
                            pat[:,3],vm[:,:,0].float())
                    elif mode=='permute':
                        pm=torch.randperm(T,generator=g,
                                          device=DEV)
                        z[:,3]=z3[:,pm]
                    elif mode=='gauss':
                        r=torch.randn(z3.shape,generator=g,
                                      device=DEV,dtype=z3.dtype)
                        r=r/r.norm(dim=-1,keepdim=True) \
                            .clamp_min(1e-6)
                        z[:,3]=r*z3.norm(dim=-1,keepdim=True)
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
            for li,hd in TRIO:
                at2=m.transformer.h[li].attn
                X=caps[li]
                cos,sin=at2.rotary(at2.c_q(X).view(B,T,9,128))
                qf=F.rms_norm(at2.c_q(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                kf=F.rms_norm(at2.c_k(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                q2=F.rms_norm(at2.c_q2(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                k2=F.rms_norm(at2.c_k2(X).view(B,T,9,128),
                              (128,))[:,:,hd]
                qf=are(qf[:,:,None],cos,sin)[:,:,0]
                kf=are(kf[:,:,None],cos,sin)[:,:,0]
                q2=are(q2[:,:,None],cos,sin)[:,:,0]
                k2=are(k2[:,:,None],cos,sin)[:,:,0]
                pats[f'{li}.{hd}']=((torch.einsum(
                    'bqd,bkd->bqk',qf.float(),kf.float())
                    *torch.einsum('bqd,bkd->bqk',q2.float(),
                                  k2.float()))
                    *torch.tril(torch.ones(T,T,device=DEV))).cpu()
            for h in hs: h.remove()
            return ce,pats
        ce0,pat0=run(None)
        for a in ARMS:
            ce1,pat1=run(a)
            d=ce1-ce0
            acc[a]['ce']+=float(d[mmask].sum())
            acc[a]['nce']+=int(mmask.sum())
            for k in pat0:
                sh=acc[a]['shift'][k]
                for b in range(B):
                    for q in range(T):
                        if not mmask[b,q]: continue
                        p0=pat0[k][b,q,:q].abs()
                        p1=pat1[k][b,q,:q].abs()
                        sh[0]+=int(int(p0.argmax())
                                   !=int(p1.argmax()))
                        sh[1]+=1
                        if q>=16:
                            r0=p0.argsort().argsort().float()
                            r1=p1.argsort().argsort().float()
                            acc[a]['corr'][k].append(float(
                                torch.corrcoef(torch.stack(
                                    [r0,r1]))[0,1]))
        print(f'batch {i} done',flush=True)
    out={}
    for a in ARMS:
        sh={k:round(v[0]/max(v[1],1),3)
            for k,v in acc[a]['shift'].items()}
        co={k:round(sum(v)/max(len(v),1),3)
            for k,v in acc[a]['corr'].items()}
        out[a]={'trio_shift':sh,'trio_rankcorr':co,
                'dce_match':round(acc[a]['ce']
                                  /max(acc[a]['nce'],1),4)}
        print(f"{a}: shift {sh} | rankcorr {co} | dCE "
              f"{out[a]['dce_match']}",flush=True)
    mean=lambda d:sum(d.values())/len(d)
    pa=mean(out['permute']['trio_shift'])>=0.15
    pb=(out['zero']['dce_match']-out['gauss']['dce_match'])>=0.05
    pc=(mean(out['patswap']['trio_rankcorr'])
        -mean(out['gauss']['trio_rankcorr']))>=0.10
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc)})
    for nm,v in (('a','permute control also >=0.15 (metric saturates)'),
                 ('b','CE separates zero from gauss by >=0.05'),
                 ('c','rank corr separates patswap from gauss')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
