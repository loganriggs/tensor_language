"""SUFFIX CODE -- computational-grade attempt on a NOVEL circuit
(user standard: executable code replicating behavior). Target: the
place-suffix|newline family leaf r.3.0 (trigger-grade at 7.9x,
class=newline). THE CODE, written down:
    def circuit(prev, cur): return TABLE[prev, cur]
where TABLE[p,c] = the circuit's newline-logit contribution measured
in a bare 2-token context [p,c]: logit_nl(real) - logit_nl(bundles
ablated), one number per corpus pair, derived from weights +
architecture alone (no full-context data). The computational claim:
the circuit's action in FULL context equals its 2-token value --
context-freeness of the mechanism.
Validation: in full context, per member position, measured
contribution = logit_nl(real) - logit_nl(ablated).
REGISTERED PREDICTIONS:
  (a) corr(TABLE[prev,cur], measured contribution) >= 0.6 over
      member positions;
  (b) code error: median |TABLE - measured| <= 50% of the median
      |measured| (scale-accurate, not just rank-accurate);
  (c) CONTROL: pair-shuffled table corr <= 0.2;
  (d) the context-dependence gap (1 - explained variance) is
      reported as the honest residual."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; NL=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'suffix_code_results.json'
TAG='r.3.0'

@torch.no_grad()
def main():
    t0=time.time()
    lf=cl.leaf(TAG); mem=lf['member']
    probes=lf['top_probes']
    print(f'{TAG}: n={lf["n_members"]} probes {probes}',flush=True)
    ROWS=cl.rows()
    tok2d=ROWS[:,:256]
    prev=torch.roll(tok2d,1,dims=1); prev[:,0]=-1
    flatP=prev.reshape(-1); flatC=tok2d.reshape(-1)
    mp=flatP[mem]; mc=flatC[mem]
    okm=mp>=0
    mem=mem[okm]; mp=mp[okm]; mc=mc[okm]
    upairs,inv=torch.unique(torch.stack([mp,mc],1),dim=0,
                            return_inverse=True)
    print(f'{len(mem)} members, {len(upairs)} unique pairs',flush=True)
    def nl2(hooks):
        outs=[]
        for i in range(0,len(upairs),512):
            bb=upairs[i:i+512].to(DEV)
            x=F.rms_norm(m.transformer.wte(bb),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            outs.append(lg[:,1,NL].cpu())
        for h in hooks: h.remove()
        return torch.cat(outs)
    base2=nl2([])
    abl2=nl2(cl.leaf_hooks(probes))
    TABLE=base2-abl2
    print(f'TABLE built: mean {float(TABLE.mean()):+.3f} '
          f'sd {float(TABLE.std()):.3f}',flush=True)
    # full-context measured contribution at member positions
    def nl_full(hooks):
        vals=torch.zeros(54272)
        for i in range(0,212,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            vals[i*256:(i+4)*256]=lg[...,NL].reshape(-1).cpu()
        for h in hooks: h.remove()
        return vals
    fr=nl_full([])
    fa=nl_full(cl.leaf_hooks(probes))
    measured=(fr-fa)[mem]
    code=TABLE[inv]
    corr=float(torch.corrcoef(torch.stack([code,measured]))[0,1])
    err=float((code-measured).abs().median())
    scale=float(measured.abs().median())
    g=torch.Generator().manual_seed(3)
    codes=TABLE[torch.randperm(len(TABLE),generator=g)][inv]
    corrs=float(torch.corrcoef(torch.stack([codes,measured]))[0,1])
    ev=1-float(((code-measured).var())/measured.var().clamp_min(1e-6))
    pa=corr>=0.6; pb=err<=0.5*scale; pc_=abs(corrs)<=0.2
    out={'tag':TAG,'n_members':len(mem),'n_pairs':len(upairs),
         'corr':round(corr,3),'shuffled_corr':round(corrs,3),
         'median_err':round(err,3),'median_scale':round(scale,3),
         'explained_var':round(ev,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':True}
    print(f'corr {corr:.3f} | shuffled {corrs:.3f} | err {err:.3f} '
          f'vs scale {scale:.3f} | explained var {ev:.2%}')
    print(f"(a) corr >=0.6: {'HELD' if pa else 'FAILED'}")
    print(f"(b) scale-accurate: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled <=0.2: {'HELD' if pc_ else 'FAILED'}")
    print(f"(d) context gap: {1-ev:.2%}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
