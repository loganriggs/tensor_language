"""HEAD 0.3 EXACT FOLD -- 476: head 0.3 (the model's second
costliest, +0.112 nats) is a previous-token head with a self
component -- offset -1 carries 65.8% of its top reads, offset 0
carries 28.3% -- and its pattern is EXACTLY token-determined: the
weights+tokens+rotary fold reproduces the real top read 1.000 of
the time (475a, 475b).
475c failed for a reason visible in my own code, not in the model:
the value table was built only from tokens that appeared as top
reads at sampled query positions, so most positions hit an empty
slot and contributed zero. At layer 0 the value is a pure function
of the token -- v = c_v(rms_norm(wte(t))) -- so the correct table
is computable from WEIGHTS ALONE over the whole vocabulary, with
no data and no holes.
Arms:
  exact   : replace the head's per-position values with the
            weights-only per-token table
  full    : replace BOTH pattern and values with their weights-only
            reconstructions (a complete fold of the head)
  shuffled: the same table with token identities permuted (null)
REGISTERED PREDICTIONS:
  (a) EXACT: the weights-only value table costs <= 0.005 nats --
      at layer 0 this should be an identity up to numerics;
  (b) FULLY FOLDABLE: replacing pattern AND values costs
      <= 0.01 nats, i.e. the whole head is a lookup;
  (c) NULL: the shuffled-token table costs >= 0.05."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=0; HD=3
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_0_3_exact_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    at=m.transformer.h[LJ].attn
    V=m.lm_head.weight.shape[0]
    # weights-only value table over the whole vocabulary
    tab=torch.zeros(V,128,device=DEV)
    for i in range(0,V,4096):
        tt=torch.arange(i,min(i+4096,V),device=DEV)
        e=F.rms_norm(m.transformer.wte(tt),(D,))
        tab[i:i+4096]=at.c_v(e).view(-1,9,128)[:,HD].float()
    g=torch.Generator().manual_seed(31)
    perm=torch.randperm(V,generator=g).to(DEV)
    print(f'value table built for {V} tokens '
          f'(norm {float(tab.norm(dim=-1).mean()):.1f})',flush=True)
    def run(mode):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if mode is not None:
                def fh(mo_,args,o_,at=at,mode=mode):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    vv=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*vv+at.lamb*v1.view_as(vv)
                    c2,s2=at.rotary(at.c_q(X).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X).view(B,T,9,128),(128,)),c2,s2)
                    qq,kk=r2(at.c_q),r2(at.c_k)
                    q22,k22=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                                    kk.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                                     k22.float())/128
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    p2=(sc*sc2)*tril
                    z=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
                    toks=idx
                    src=tab if mode!='shuffled' else tab[perm]
                    rep=src[toks]                    # B,T,128
                    if mode in ('exact','shuffled'):
                        z[:,HD]=torch.einsum('bqk,bkd->bqd',
                                             p2[:,HD],rep)
                    elif mode=='full':
                        # pattern rebuilt from tokens+rotary only
                        Ef=F.rms_norm(m.transformer.wte(toks),(D,))
                        cf,sf=at.rotary(at.c_q(Ef).view(B,T,9,128))
                        def rf(w):
                            return are(F.rms_norm(
                                w(Ef).view(B,T,9,128),
                                (128,))[:,:,HD][:,:,None],cf,sf
                                )[:,:,0]
                        qf,kf=rf(at.c_q),rf(at.c_k)
                        q2f,k2f=rf(at.c_q2),rf(at.c_k2)
                        fp=((torch.einsum('bqd,bkd->bqk',qf.float(),
                                          kf.float())/128)
                            *(torch.einsum('bqd,bkd->bqk',
                                           q2f.float(),
                                           k2f.float())/128))*tril
                        z[:,HD]=torch.einsum('bqk,bkd->bqd',fp,rep)
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
         ('exact','full','shuffled')}
    print('dCE:',res,flush=True)
    pa=abs(res['exact'])<=0.005
    pb=abs(res['full'])<=0.01
    pc=res['shuffled']>=0.05
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','weights-only value table is exact'),
                 ('b','the whole head folds to a lookup'),
                 ('c','shuffled-token null is costly')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
