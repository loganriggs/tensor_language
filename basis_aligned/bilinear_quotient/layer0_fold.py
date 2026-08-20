"""LAYER-0 FOLD -- 477: head 0.3, the model's second costliest
(+0.112 nats), folds EXACTLY to a lookup (476): replacing its
pattern with a weights+tokens+rotary reconstruction AND its values
with a weights-only per-token table over the full 50304-token
vocabulary costs -0.0 nats, while the token-shuffled table costs
+0.147.
At layer 0 exact foldability is architecturally guaranteed --
attention there reads only token embeddings -- so the interesting
question is not whether ONE head folds but whether the WHOLE first
attention layer does, at once, and where the boundary lies.
Arms:
  layer0_fold   : all nine heads of layer 0 replaced by their
                  weights-only pattern + per-token value tables
  layer0_shuf   : the same with token identities permuted (null)
  layer1_fold   : the same construction applied to layer 1, whose
                  input is NOT just token embeddings (boundary
                  test -- this should fail)
REGISTERED PREDICTIONS:
  (a) WHOLE LAYER: folding all nine layer-0 heads costs
      <= 0.01 nats;
  (b) NULL: the shuffled version costs >= 0.20;
  (c) BOUNDARY: the same fold at layer 1 costs >= 0.10, because
      layer 1 reads layer-0 outputs rather than raw tokens."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'layer0_fold_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    V=m.lm_head.weight.shape[0]
    def value_table(li):
        at=m.transformer.h[li].attn
        tab=torch.zeros(V,9,128,device=DEV)
        for i in range(0,V,4096):
            tt=torch.arange(i,min(i+4096,V),device=DEV)
            e=F.rms_norm(m.transformer.wte(tt),(D,))
            tab[i:i+4096]=at.c_v(e).view(-1,9,128).float()
        return tab
    tab0=value_table(0); tab1=value_table(1)
    g=torch.Generator().manual_seed(31)
    perm=torch.randperm(V,generator=g).to(DEV)
    print(f'value tables built ({V} tokens)',flush=True)
    def run(mode):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if mode is not None:
                li=1 if mode=='layer1_fold' else 0
                tab=tab1 if li==1 else tab0
                src=tab[perm] if mode=='layer0_shuf' else tab
                at=m.transformer.h[li].attn
                def fh(mo_,args,o_,at=at,src=src,idx=idx):
                    y,v1r=o_
                    X=args[0]
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    Ef=F.rms_norm(m.transformer.wte(idx),(D,))
                    cf,sf=at.rotary(at.c_q(Ef).view(B,T,9,128))
                    def rf(w):
                        return are(F.rms_norm(
                            w(Ef).view(B,T,9,128),(128,)),cf,sf)
                    qf,kf=rf(at.c_q),rf(at.c_k)
                    q2f,k2f=rf(at.c_q2),rf(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2f.float(),
                                     k2f.float())/128
                    fp=(sc*sc2)*tril
                    rep=src[idx]                    # B,T,9,128
                    z=torch.einsum('bhqk,bkhd->bhqd',fp,
                                   rep.float())
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
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
    res={a:round(run(a)-base,5) for a in
         ('layer0_fold','layer0_shuf','layer1_fold')}
    print('dCE:',res,flush=True)
    pa=abs(res['layer0_fold'])<=0.01
    pb=res['layer0_shuf']>=0.20
    pc=res['layer1_fold']>=0.10
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','all nine layer-0 heads fold (<=0.01)'),
                 ('b','shuffled null costly (>=0.20)'),
                 ('c','layer 1 does NOT fold (>=0.10)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
