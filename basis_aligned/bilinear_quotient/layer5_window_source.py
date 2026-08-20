"""LAYER-5 WINDOW SOURCE -- 483: restricting each attention layer
in turn to a 4-token read window found ONE layer that matters
(482): layer 5 costs +1.112 nats while every other layer costs
at most +0.086. That looks like "layer 5 carries this model's
long-range dependence" -- but there is an obvious confound I can
see before claiming it. Head 5.7 is an attention SINK that reads
POSITION 0 for 99.8% of queries (432), and position 0 is exactly
what a 4-token sliding window cuts off. So layer 5's cost may be
the sink losing its constant, not long-range content reading.
Disambiguate with three arms at layer 5:
  win4        : the 4-token window (reference, +1.112)
  win4_plus0  : the same window PLUS position 0 always allowed
  sink_only   : window applied ONLY to head 5.7
  others_only : window applied to all layer-5 heads EXCEPT 5.7
REGISTERED PREDICTIONS:
  (a) SINK EXPLAINS IT: win4_plus0 costs <= 0.15 nats;
  (b) LOCALISED TO 5.7: sink_only reproduces >= 0.5 of the
      +1.112;
  (c) THE REST ARE LOCAL: others_only costs <= 0.15.
If all three hold, 482's headline is really "the model's only
non-local read is a constant fetch", and the apparent convergence
with the induction band is coincidence -- which must then be said
plainly."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; SINK=7; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'layer5_window_source_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    def run(mode):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if mode is not None:
                def fh(mo_,args,o_,mode=mode):
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
                    win=((ar[:,None]-ar[None,:])<K).float()
                    if mode=='win4_plus0':
                        w0=torch.zeros(T,T,device=DEV)
                        w0[:,0]=1.0
                        win=torch.clamp(win+w0,max=1.0)
                    full=tril
                    lim=tril*win
                    pat=(sc*sc2)
                    out=torch.empty_like(pat)
                    for h in range(9):
                        if mode=='sink_only':
                            mk=lim if h==SINK else full
                        elif mode=='others_only':
                            mk=full if h==SINK else lim
                        else:
                            mk=lim
                        out[:,h]=pat[:,h]*mk
                    z=torch.einsum('bhqk,bkhd->bhqd',out,vm.float())
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            tot+=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                 reduction='none').mean().item()
            cnt+=1
            for h in hs: h.remove()
        return tot/max(cnt,1)
    base=run(None)
    res={a:round(run(a)-base,4) for a in
         ('win4','win4_plus0','sink_only','others_only')}
    print('dCE:',res,flush=True)
    pa=res['win4_plus0']<=0.15
    pb=res['sink_only']>=0.5*res['win4']
    pc=res['others_only']<=0.15
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','allowing position 0 fixes it (<=0.15)'),
                 ('b','head 5.7 alone reproduces >=50%'),
                 ('c','the other eight heads are local (<=0.15)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
