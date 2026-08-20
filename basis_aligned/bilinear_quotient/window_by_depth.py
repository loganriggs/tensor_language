"""WINDOW BY DEPTH -- 482: both of the first two blocks are
essentially LOCAL. Block 0 needs the current token plus one
previous (+0.004, 480); block 1 needs four positions for +0.014
and even two for +0.080, refuting my prediction that layer 1 is
where context widens (481b, cost >= 0.30 expected).
So where does long-range context actually enter this model?
Measure it directly: restrict ONE attention layer at a time to a
4-token read window (all other layers intact) and price it. The
layer whose restriction is expensive is the layer that genuinely
needs distant reads.
Layers swept: 0-17, all of them.
REGISTERED PREDICTIONS:
  (a) LOCAL FRONT: layers 0, 1 and 2 each cost <= 0.05 under a
      4-token window;
  (b) A TRANSITION EXISTS: at least one layer costs >= 0.30;
  (c) INDUCTION LAYERS PAY: the largest cost falls in layers 5-8,
      where the induction band lives (376-408) and long-range
      matching is the documented function."""

import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'window_by_depth_results.json'
NR=16
LAYERS=list(range(18)); K=4

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    import sys
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    def run(LJ,active,k=K):
        tot=0.0; cnt=0
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
                        mask=mask*((ar[:,None]-ar[None,:])<k
                                   ).float()
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
            tot+=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                 reduction='none').mean().item()
            cnt+=1
            for h in hs: h.remove()
        return tot/max(cnt,1)
    base=run(0,False)
    res={}
    for LJ in LAYERS:
        res[LJ]=round(run(LJ,True)-base,4)
        print(f'layer {LJ}: dCE {res[LJ]:+.4f}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    pa=all(res[l]<=0.05 for l in (0,1,2))
    pb=any(v>=0.30 for v in res.values())
    worst=max(res,key=res.get)
    pc=5<=worst<=8
    out={'baseline_ce':round(base,4),'window':K,'dce_by_layer':res,
         'worst_layer':worst,'worst_cost':res[worst],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'worst layer {worst} at {res[worst]:+.4f}')
    for nm,v in (('a','layers 0-2 local (<=0.05)'),
                 ('b','some layer costs >=0.30'),
                 ('c','the worst layer is in 5-8 (induction)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
