"""BLOCK-1 CONTEXT WINDOW -- 481: with the block-0 lambda mix
corrected, the front of the model is a BIGRAM function (480):
truncating attn0's reads to the current token plus ONE previous
position costs +0.0041 nats, four positions costs -0.042 (free),
and the untruncated fold is exact at +0.0000. Only self-only
(k=1) is expensive at +0.537.
How fast does the required context widen with depth? Apply the
same measurement one block up. attn1 reads block-0 outputs, which
are themselves bigram functions, so block 1 could in principle
stay narrow -- or the whole point of layer 1 could be to widen it.
Truncate attn1's reads to the last k positions (its pattern
computed normally from the real residual, only the read window
restricted) and sweep.
REGISTERED PREDICTIONS:
  (a) SANITY: untruncated costs <= 0.02;
  (b) WIDER THAN BLOCK 0: k = 2 costs >= 0.30 at block 1, against
      +0.004 at block 0 -- layer 1 is where context widens;
  (c) STILL BOUNDED: k = 16 costs <= 0.10."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=1
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'block1_window_results.json'
NR=16
KS=[None,32,16,8,4,2,1]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    at=m.transformer.h[LJ].attn
    import sys
    are=sys.modules[type(at).__module__].apply_rotary_emb
    def run(k,active):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if active:
                def fh(mo_,args,o_,k=k):
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
    base=run(None,False)
    res={}
    for k in KS:
        lbl='full' if k is None else f'k{k}'
        res[lbl]=round(run(k,True)-base,4)
        print(f'{lbl}: dCE {res[lbl]:+.4f}',flush=True)
    pa=abs(res['full'])<=0.02
    pb=res['k2']>=0.30
    pc=res['k16']<=0.10
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','untruncated is exact'),
                 ('b','k=2 costs >=0.30 (wider than block 0)'),
                 ('c','k=16 costs <=0.10 (still bounded)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
