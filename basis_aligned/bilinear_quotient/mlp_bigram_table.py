"""MLP BIGRAM TABLE -- 479: per-token tables for the front MLPs
are expensive (478: m0 +1.018, m1 +1.775, m2 +0.744 nats;
shuffled m0 +3.188), which refutes the "m0 is a token table"
framing this program has been carrying and puts a price on the
contextual part of its input.
But we now know exactly what that contextual part IS. mlp0's input
is rms_norm(wte(t) + attn0_out), and attn0 is itself a bigram
table (477) dominated by the previous token (476: head 0.3 reads
offset -1 at 66%). So m0 should be nearly a function of the TOKEN
PAIR (t, t_prev) rather than of t alone.
Arms:
  unigram : m0 replaced by its per-token table (reference, +1.018)
  bigram  : m0 recomputed on rms_norm(wte(t) + attn0-with-all-
            weight-on-offset-(-1)), i.e. a pure function of
            (t, t_prev)
  shuf    : the same with the previous token shuffled (null)
REGISTERED PREDICTIONS:
  (a) PAIR BEATS TOKEN: the bigram form costs <= 0.30 nats;
  (b) MARGIN: it beats the unigram table by >= 0.50;
  (c) NULL: shuffling the previous token costs >= 1.00."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp_bigram_table_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    V=m.lm_head.weight.shape[0]
    mlp0=m.transformer.h[0].mlp
    tab=torch.zeros(V,D,device=DEV)
    for i in range(0,V,4096):
        tt=torch.arange(i,min(i+4096,V),device=DEV)
        e=F.rms_norm(m.transformer.wte(tt),(D,))
        tab[i:i+4096]=mlp0(e).float()
    at0=m.transformer.h[0].attn
    g=torch.Generator().manual_seed(7)
    def run(mode):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if mode is not None:
                if mode=='unigram':
                    def fh(mo,i_,o_,idx=idx):
                        return tab[idx].to(o_.dtype)
                    hs.append(mlp0.register_forward_hook(fh))
                else:
                    prev=torch.roll(idx,1,dims=1)
                    prev[:,0]=idx[:,0]
                    if mode=='shuf':
                        pi=torch.randperm(T,generator=g).to(DEV)
                        prev=prev[:,pi]
                    def fh(mo,i_,o_,idx=idx,prev=prev):
                        # attn0 output if ALL weight sat on the
                        # previous token: v(prev) through c_proj
                        e_prev=F.rms_norm(
                            m.transformer.wte(prev),(D,))
                        v=at0.c_v(e_prev).view(B,T,9,128).float()
                        vm=(1-at0.lamb)*v+at0.lamb*v
                        a0=at0.c_proj(vm.reshape(B,T,-1)
                                      .to(e_prev.dtype)).float()
                        e_cur=F.rms_norm(
                            m.transformer.wte(idx),(D,)).float()
                        xin=F.rms_norm(e_cur+a0,(D,))
                        return mlp0(xin.to(o_.dtype))
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
    base=run(None)
    res={a:round(run(a)-base,4) for a in
         ('unigram','bigram','shuf')}
    print('dCE:',res,flush=True)
    pa=res['bigram']<=0.30
    pb=(res['unigram']-res['bigram'])>=0.50
    pc=res['shuf']>=1.00
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','bigram form costs <=0.30'),
                 ('b','beats the unigram table by >=0.50'),
                 ('c','shuffled previous token costs >=1.00')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
