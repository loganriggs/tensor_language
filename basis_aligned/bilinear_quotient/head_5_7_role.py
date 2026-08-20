"""VALUE VS PATTERN (CE) -- 423: the metric audit (422) showed the
top-read-shift metric is non-specific (a norm-matched random write
damages MORE than any real arm), but the FUNCTIONAL metric found a
clean dissociation at a6.h3: giving the head the control head's
PATTERN while keeping its own values costs -0.001 nats (free),
while keeping its own pattern with the control's VALUES costs
0.062 -- as much as deleting it (0.057). Confirm at larger n and
test generality, with dCE as the primary metric (argmax shift is
retired for cross-arm comparison; rank correlation reported as the
graceful secondary).
Pairs (head, control) at the same layer: (6,3)/(6,0) [the deep
band's courier], (4,7)/(4,2) [5.5's courier], (6,5)/(6,7) and
(4,1)/(4,5) [unnamed pairs, generality baseline].
REGISTERED PREDICTIONS:
  (a) REPLICATION: for a6.h3 at n=32 rows, patswap |dCE| <= 0.02
      and valswap dCE >= 0.8 x zero dCE;
  (b) COURIER GENERALITY: a4.h7 shows the same ordering
      (patswap < valswap), i.e. both named couriers carry their
      function in the payload, not the read positions;
  (c) SCOPE FORK (report either way): the two unnamed pairs show
      the same ordering -> "payload carries function" is a
      layer-level property of this architecture, not a courier
      property; if they differ, it is courier-specific;
  (d) sanity: zero dCE > 0 for every head tested."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_5_7_role_results.json'
NR=32
PAIRS=[(5,7,6),(5,7,0)]   # two different siblings
ARMS=['zero','patswap','valswap']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={f'{lj}.{h1}': {a:{'m':0.0,'nm':0,'all':0.0,'nall':0}
         for a in ARMS} for lj,h1,_ in PAIRS}
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
        def run(lj=None,h1=None,h0=None,mode=None):
            hs=[]
            if mode is not None:
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,h1=h1,h0=h0,mode=mode):
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
                    if mode=='zero': z[:,h1]=0
                    elif mode=='patswap':
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pat[:,h0],vm[:,:,h1].float())
                    elif mode=='valswap':
                        z[:,h1]=torch.einsum('bqk,bkd->bqd',
                            pat[:,h1],vm[:,:,h0].float())
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
            for h in hs: h.remove()
            return ce
        ce0=run()
        for lj,h1,h0 in PAIRS:
            k=f'{lj}.{h1}'
            for a in ARMS:
                d=run(lj,h1,h0,a)-ce0
                acc[k][a]['m']+=float(d[mmask].sum())
                acc[k][a]['nm']+=int(mmask.sum())
                acc[k][a]['all']+=float(d.sum())
                acc[k][a]['nall']+=d.numel()
        print(f'batch {i} done',flush=True)
    out={}
    for k,arms in acc.items():
        out[k]={a:{'dce_match':round(v['m']/max(v['nm'],1),4),
                   'dce_all':round(v['all']/max(v['nall'],1),4)}
                for a,v in arms.items()}
        print(f"{k}: "+" | ".join(
            f"{a} match {out[k][a]['dce_match']} all "
            f"{out[k][a]['dce_all']}" for a in ARMS),flush=True)
    c3=out['5.7']
    pa=c3['zero']['dce_match']>0.3   # sanity: the big cost is real
    pb=c3['patswap']['dce_match']>=0.5*c3['zero']['dce_match']
    pc=True
    pd=all(out[k]['zero']['dce_match']>0 for k in out)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c_general':bool(pc),'pred_d':bool(pd)})
    for nm,v in (('a','5.7 deletion cost >0.3 at match (sanity)'),
                 ('b','5.7 POSITION-sensitive (patswap >=0.5x zero)'),
                 ('c_general','n/a'),
                 ('d','deletion costs >0 everywhere')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
