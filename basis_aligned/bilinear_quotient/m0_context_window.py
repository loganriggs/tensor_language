"""M0 CONTEXT WINDOW -- 480: forcing attn0's weight onto the
PREVIOUS token and recomputing m0 costs +1.255 -- WORSE than
ignoring attn0 altogether (+1.018, 479). So m0's missing context
is not the previous token, even though one layer-0 head reads
offset -1 at 66%; the other eight read elsewhere, and collapsing
them all onto -1 is worse than nothing.
What m0 needs is attn0's actual output -- which is itself an exact
weights-only bigram table (477). So m0 IS exactly computable from
tokens; the open quantity is HOW WIDE a prefix it needs. Measure
it: rebuild attn0's pattern from weights and tokens (the exact
fold), truncate its reads to the last k positions, and sweep k.
Arms: k = full (sanity, should be exact), 16, 8, 4, 2, 1.
REGISTERED PREDICTIONS:
  (a) SANITY: the untruncated fold costs <= 0.02 nats (it is an
      identity up to numerics; a miss means a bug);
  (b) NARROW WINDOW: k = 4 costs <= 0.20 nats -- m0's context
      need is local;
  (c) MONOTONE: cost decreases as k grows, and k = 1 costs
      >= 1.00 (consistent with 479's +1.255)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m0_context_window_results.json'
NR=16
KS=[None,16,8,4,2,1]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    at0=m.transformer.h[0].attn
    mlp0=m.transformer.h[0].mlp
    L0=mlp0.Left.weight.float(); R0=mlp0.Right.weight.float()
    D0=mlp0.Down.weight.float(); B0=mlp0.Down_bias.detach().float()
    def mlp0_manual(x):
        return ((x@L0.T)*(x@R0.T))@D0.T+B0
    def run(k,active):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if active:
                def fh(mo,i_,o_,idx=idx,k=k):
                    E=F.rms_norm(m.transformer.wte(idx),(D,))
                    cos,sin=at0.rotary(at0.c_q(E).view(B,T,9,128))
                    def rf(w):
                        return are(F.rms_norm(
                            w(E).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=rf(at0.c_q),rf(at0.c_k)
                    q2,k2=rf(at0.c_q2),rf(at0.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    mask=torch.tril(torch.ones(T,T,device=DEV))
                    if k is not None:
                        ar=torch.arange(T,device=DEV)
                        win=((ar[:,None]-ar[None,:])<k)
                        mask=mask*win.float()
                    pat=(sc*sc2)*mask
                    v=at0.c_v(E).view(B,T,9,128).float()
                    vm=(1-at0.lamb)*v+at0.lamb*v
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm)
                    a0=at0.c_proj(z.transpose(1,2).contiguous()
                                  .view(B,T,-1).to(E.dtype)).float()
                    # block-0 lambda mix: the residual entering
                    # the MLP is (lam0+lam1)*E + attn_out, and
                    # lam0+lam1 = 12.19 here. Using 1.0*E (the
                    # first version) under-weighted the embedding
                    # by 12x and made the "exact" fold cost 0.55.
                    lam=m.transformer.h[0].lambdas.detach().float()
                    xin=F.rms_norm(float(lam.sum())*E.float()+a0,
                                   (D,))
                    return mlp0_manual(xin).to(o_.dtype)
                hs.append(mlp0.register_forward_hook(fh))
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
    pb=res['k4']<=0.20
    seq=[res['k1'],res['k2'],res['k4'],res['k8'],res['k16'],
         res['full']]
    pc=(all(seq[i]>=seq[i+1]-1e-4 for i in range(len(seq)-1))
        and res['k1']>=1.00)
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','untruncated fold is exact (<=0.02)'),
                 ('b','k=4 costs <=0.20 (local context)'),
                 ('c','monotone in k and k=1 >= 1.00')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
