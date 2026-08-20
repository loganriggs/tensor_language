"""BIAS CONSUMER -- 445: injection depth is a CLIFF, not a decay
(444/445 data): replacing head 5.7's write with its constant IN
PLACE is free (-0.005), but delivering the same constant anywhere
after layer 5's MLP costs 1.02-1.37 -- worse than deleting the
head (0.915). Two readings remain open: either mlp5 is the
consumer that must see the bias, or adding ANY constant late is
simply harmful. Disentangle with subtraction arms and a junk
control (the head stays intact, so "who needs it" is asked
directly).
Arms (const = the head's own mean write):
  delete        : head zeroed (reference, ~0.92)
  mlp5_blind    : head zeroed + const added at block 6 input
                  (everyone after layer 5 sees it, mlp5 does not)
  after_blind   : head intact + const SUBTRACTED at block 6 input
                  (mlp5 sees it, nobody later does)
  junk_add      : head zeroed + a norm-matched RANDOM vector added
                  at block 6 (is late injection harmful per se?)
  junk_sub      : head intact + random vector subtracted there
REGISTERED PREDICTIONS:
  (a) MLP5 IS THE CONSUMER: after_blind costs <= 0.20 nats --
      once mlp5 has seen the bias, removing it from the rest of
      the stack is nearly free;
  (b) JUNK CONTROL: junk_add costs >= 0.5 x mlp5_blind, i.e. much
      of the late-injection penalty is generic damage rather than
      missing consumption -- report the split either way;
  (c) sanity: delete ~0.92 and junk_sub < junk_add."""


import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_consumer_results.json'
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
    POINTS=['mlp5_blind','after_blind','junk_add','junk_sub']
    acc={a:[0.0,0] for a in ['delete']+POINTS}
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
        # /128 per QK factor, as the model does (omitted in the
        # first version: made the injected constant 16384x too
        # large, writeup 443)
        pat=((torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())/128)
             *(torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())/128)) \
            *torch.tril(torch.ones(T,T,device=DEV))
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()*(1-at.lamb)
        Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
        const=(torch.einsum('bqk,bkd->bqd',pat,v)@Wp.T) \
            .mean(dim=(0,1))            # one vector
        def run(mode):
            hs=[]
            if mode in ('delete','mlp5_blind','junk_add',
                        'inplace'):
                def fh(mo_,args,o_,at=at,mode=mode):
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
                    if mode=='inplace':
                        zz[:,HD]=zz[:,HD].mean(dim=(0,1),
                                               keepdim=True)
                    else:
                        zz[:,HD]=0
                    return (at.c_proj(zz.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X2.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
                if mode in ('mlp5_blind','after_blind',
                            'junk_add','junk_sub'):
                    gj=torch.Generator(device=DEV).manual_seed(77)
                    rnd=torch.randn(const.shape,generator=gj,
                                    device=DEV)
                    rnd=rnd/rnd.norm()*const.norm()
                    vec={'mlp5_blind':const,'after_blind':-const,
                         'junk_add':rnd,'junk_sub':-rnd}[mode]
                    def ph(mo_,args,vec=vec):
                        return (args[0]+vec.to(args[0].dtype),
                                )+tuple(args[1:])
                    hs.append(m.transformer.h[LJ+1]
                              .register_forward_pre_hook(ph))
            xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0b=xx
            v1b=None
            for blk in m.transformer.h: xx,v1b=blk(xx,v1b,x0b)

            lg=(30*torch.tanh(m.lm_head(F.rms_norm(xx,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h_ in hs: h_.remove()
            return ce,lg.mean(dim=(0,1)).cpu()
        base,lgf=run(None)
        for a in ['delete']+POINTS:
            c,lg=run(a)
            acc[a][0]+=c-base; acc[a][1]+=1
            if a in ('delete','final'):
                lg_acc[a if a=='delete' else 'direct'] \
                    +=float(lg[topi].mean())
        lg_acc['full']+=float(lgf[topi].mean()); nb+=1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()}
    LG={a:round(v/max(nb,1),4) for a,v in lg_acc.items()}
    pa=CE['after_blind']<=0.20
    pb=CE['junk_add']>=0.5*CE['mlp5_blind']
    pc=(CE['delete']>0.5 and CE['junk_sub']<CE['junk_add'])
    out={'dce':CE,'order':POINTS,'top20_mean_logit':LG,
         'n_top_tokens':len(topi),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('dCE:',{p:CE[p] for p in POINTS},
          '| delete',CE['delete'])
    for nm,v in (('a','after_blind <=0.20 (mlp5 is the consumer)'),
                 ('b','junk_add >=0.5x mlp5_blind (generic damage)'),
                 ('c','delete ~0.92 and junk_sub < junk_add')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
