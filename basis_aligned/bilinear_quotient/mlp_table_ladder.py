"""MLP TABLE LADDER -- 478: the whole first attention layer folds
exactly (477: all nine heads replaced by weights-only token-pair
patterns and per-token value tables, dCE -0.0; shuffled null
+0.237; the same construction at layer 1 costs +1.470, a sharp
boundary).
Attention at layer 0 is tableable because it reads only token
embeddings. The MLPs are the other half of the front of the model,
and mlp0 was identified early as the model's identity-code
generator. But mlp0's input is rms_norm(E + attn0_out), NOT the
token alone -- 393 showed exactly that contextual term mattered
for the induction trigger. So: how far up does per-token
TABLEABILITY reach in the MLP chain, and what does ignoring the
contextual part actually cost?
Arms: replace m0, m1, m2 in turn with a per-token table computed
from weights alone, T[t] = mlp_i(rms_norm(wte(t))), plus a
token-shuffled null for m0.
REGISTERED PREDICTIONS:
  (a) m0 IS NEARLY A TABLE: the per-token table for m0 costs
      <= 0.10 nats -- it is the identity-code generator and its
      contextual input is a correction, not its substance;
  (b) LADDER: cost rises monotonically with depth, m0 < m1 < m2;
  (c) NULL: the shuffled-token table for m0 costs >= 0.50."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp_table_ladder_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    V=m.lm_head.weight.shape[0]
    tabs={}
    for li in (0,1,2):
        mlp=m.transformer.h[li].mlp
        tb=torch.zeros(V,D,device=DEV)
        for i in range(0,V,4096):
            tt=torch.arange(i,min(i+4096,V),device=DEV)
            e=F.rms_norm(m.transformer.wte(tt),(D,))
            tb[i:i+4096]=mlp(e).float()
        tabs[li]=tb
        print(f'table m{li} built (norm '
              f'{float(tb.norm(dim=-1).mean()):.1f})',flush=True)
    g=torch.Generator().manual_seed(31)
    perm=torch.randperm(V,generator=g).to(DEV)
    def run(mode):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            hs=[]
            if mode is not None:
                li=int(mode[1]); shuf=mode.endswith('shuf')
                src=tabs[li][perm] if shuf else tabs[li]
                def fh(mo,i_,o_,src=src,idx=idx):
                    return src[idx].to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(fh))
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
         ('m0','m1','m2','m0shuf')}
    print('dCE:',res,flush=True)
    pa=res['m0']<=0.10
    pb=(res['m0']<res['m1']<res['m2'])
    pc=res['m0shuf']>=0.50
    out={'baseline_ce':round(base,4),'dce':res,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    for nm,v in (('a','m0 is nearly a per-token table (<=0.10)'),
                 ('b','cost rises with depth m0<m1<m2'),
                 ('c','shuffled m0 table costly (>=0.50)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
