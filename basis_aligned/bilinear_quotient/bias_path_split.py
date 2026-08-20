"""BIAS PATH SPLIT -- 442: reading the bias vector through the
unembedding says it PUSHES function words and punctuation
(' and', ',', ' (', ' or', ' in', ' to', '-'), but the causal
test says the opposite: deleting head 5.7 RAISES exactly those
tokens' logits by +0.211 against +0.074 for a random control
(2.8x). Direct-path reading and total effect disagree in SIGN.
Resolve it by splitting the paths. Arms:
  full      : intact
  delete    : head 5.7's write zeroed
  direct    : head 5.7 zeroed, but its constant added straight to
              the final residual before the unembedding (so the
              vector reaches the logits WITHOUT passing through
              layers 6-17)
  indirect  : head 5.7 zeroed at layer 5 but its constant added
              back to the residual at layer 6 (so downstream
              computation sees it, as normally)
REGISTERED PREDICTIONS:
  (a) DOWNSTREAM WORK: the direct arm recovers < 30% of the
      head's CE contribution -- the bias earns its 0.92 nats by
      being consumed by later layers, not by shifting logits;
  (b) the indirect arm recovers >= 80% (it is the real path);
  (c) SIGN: under the direct arm the top-20 tokens' logits move
      UP relative to deletion (matching the unembedding read),
      confirming the readout measures only the direct path."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_path_split_results.json'
NR=32

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    prev=json.load(open(PT+'bias_semantics_results.json'))
    toptok=prev['top20_tokens']
    W=m.lm_head.weight.float()
    enc=cl.enc()
    topi=[enc.encode_single_token(t) if hasattr(
              enc,'encode_single_token') else None
          for t in toptok]
    topi=[t for t in topi if t is not None]
    acc={a:[0.0,0] for a in ('delete','direct','indirect')}
    lg_acc={a:0.0 for a in ('full','delete','direct')}
    nb=0
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4; cap={}
        h=m.transformer.h[LJ].attn.register_forward_pre_hook(
            lambda mo_,args: cap.__setitem__('X',args[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        h.remove()
        # the head's own write, and its per-batch constant
        at=m.transformer.h[LJ].attn; X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        def rot(w):
            return are(F.rms_norm(w(X).view(B,T,9,128),
                       (128,))[:,:,HD][:,:,None],cos,sin)[:,:,0]
        qf,kf=rot(at.c_q),rot(at.c_k); q2,k2=rot(at.c_q2),rot(at.c_k2)
        pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
             *torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())) \
            *torch.tril(torch.ones(T,T,device=DEV))
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()*(1-at.lamb)
        Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
        const=(torch.einsum('bqk,bkd->bqd',pat,v)@Wp.T) \
            .mean(dim=(0,1))            # one vector
        def run(mode):
            hs=[]
            if mode is not None:
                def fh(mo_,args,o_,at=at):
                    y,v1r=o_
                    X2=args[0]
                    v1b=args[1] if args[1] is not None else v1r
                    vv=at.c_v(X2).view(B,T,9,128)
                    vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
                    c2,s2=at.rotary(at.c_q(X2).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X2).view(B,T,9,128),(128,)),c2,s2)
                    qq,kk=r2(at.c_q),r2(at.c_k)
                    q22,k22=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                                    kk.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                                     k22.float())/128
                    p2=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
                    zz[:,HD]=0
                    return (at.c_proj(zz.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X2.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
                if mode=='indirect':
                    b6=m.transformer.h[LJ+1]
                    def ph(mo_,args,b6=b6):
                        return (args[0]+const.to(args[0].dtype),
                                )+tuple(args[1:])
                    hs.append(b6.register_forward_pre_hook(ph))
            xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0b=xx
            v1b=None
            for blk in m.transformer.h: xx,v1b=blk(xx,v1b,x0b)
            if mode=='direct': xx=xx+const.to(xx.dtype)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(xx,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h_ in hs: h_.remove()
            return ce,lg.mean(dim=(0,1)).cpu()
        base,lgf=run(None)
        for a in ('delete','direct','indirect'):
            c,lg=run(a)
            acc[a][0]+=c-base; acc[a][1]+=1
            if a in ('delete','direct'):
                lg_acc[a]+=float(lg[topi].mean())
        lg_acc['full']+=float(lgf[topi].mean()); nb+=1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()}
    LG={a:round(v/max(nb,1),4) for a,v in lg_acc.items()}
    dele=CE['delete']
    rec=lambda a: round(1-CE[a]/dele,3) if dele else None
    pa=(rec('direct') is not None and rec('direct')<0.30)
    pb=(rec('indirect') is not None and rec('indirect')>=0.80)
    pc=LG['direct']>LG['delete']
    out={'dce':CE,'recovered_fraction':{'direct':rec('direct'),
         'indirect':rec('indirect')},'top20_mean_logit':LG,
         'n_top_tokens':len(topi),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('dCE:',CE,'| recovered:',out['recovered_fraction'])
    print('top-20 mean logit:',LG)
    for nm,v in (('a','direct path recovers <30%'),
                 ('b','indirect path recovers >=80%'),
                 ('c','direct arm raises the pushed tokens')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
