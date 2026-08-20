"""PUNCT HELDOUT -- 438: r.13.2.1's punctuation claim survived
three adversarial attacks (reviewer CONFIRM, 436): 39/49 vs a
random rank-matched subspace's 21/49, and a base-CE confound that
runs the WRONG WAY (help-rate rises with base CE). The reviewer's
standing objection is the right one: the test lives on the same
corpus the leaf was discovered in, and its alpha corrects only
within-leaf sub-tests, not the search across leaves and classes.
Settle it on text the census never saw: fresh FineWeb rows
(cl.fineweb_rows, the training distribution, deduped against the
eval store), scoring the claim in its GENERALIZED form -- does
ablating this bundle lower CE at punctuation targets relative to
non-punctuation targets?
Arms: the leaf's own bundle vs a rank-matched random subspace in
the same components. Statistic: (mean dCE at punct targets) minus
(mean dCE at non-punct targets), with a label-permutation null.
REGISTERED PREDICTIONS:
  (a) GENERALIZES: on fresh rows the own-bundle punct-minus-
      nonpunct dCE difference is NEGATIVE (punct helped more) with
      permutation p <= 0.01;
  (b) SPECIFIC: the random-subspace arm's difference is not
      significant at p <= 0.01;
  (c) sanity: both arms produce a non-trivial overall dCE
      (|mean| > 0.01) so the ablations are doing something."""
import json, ast, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256; TAG='r.13.2.1'
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_heldout_results.json'
NROWS=64

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    rows=cl.fineweb_rows(NROWS)
    print(f'fresh rows: {tuple(rows.shape)}',flush=True)
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    rk={}
    for p in probes:
        key=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
        n=(p[3][1]-p[3][0]) if p[0]=='pca' else 8
        rk[key]=rk.get(key,0)+n
    def own_hooks(): return cl.proj_hooks(lf['top_probes'])
    def rnd_hooks(seed):
        hs=[]
        for key,n in rk.items():
            gg=torch.Generator(device=DEV).manual_seed(seed)
            P=orth(torch.randn(D,n,generator=gg,device=DEV))
            mod=MODS[key]
            if key[0]=='a':
                def fh(mo,i_,o_,P=P):
                    y,v1=o_
                    yf=y.float().reshape(-1,D)
                    return ((yf-(yf@P)@P.T).view(y.shape)
                            .to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,P=P):
                    yf=o_.float().reshape(-1,D)
                    return (yf-(yf@P)@P.T).view(o_.shape) \
                        .to(o_.dtype)
            hs.append(mod.register_forward_hook(fh))
        return hs
    def ce_all(mk):
        ces=[]
        for i in range(0,NROWS,4):
            bb=rows[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            hs=mk() if mk else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                       reduction='none').view(4,T).cpu())
            for h in hs: h.remove()
        return torch.cat(ces)
    base=ce_all(None)
    # punctuation mask on TARGET tokens
    pm=torch.zeros(NROWS,T,dtype=torch.bool)
    for r in range(NROWS):
        for q in range(T):
            s=cl.d1(int(rows[r,q+1])).strip()
            pm[r,q]=bool(s) and not any(c.isalnum() for c in s)
    print(f'punct targets: {int(pm.sum())} of {pm.numel()}',
          flush=True)
    g=torch.Generator().manual_seed(7)
    def score(d):
        a=float(d[pm].mean()); b=float(d[~pm].mean())
        obs=a-b
        null=[]
        flat=d.reshape(-1); k=int(pm.sum())
        for _ in range(2000):
            idxp=torch.randperm(flat.numel(),generator=g)[:k]
            msk=torch.zeros(flat.numel(),dtype=torch.bool)
            msk[idxp]=True
            null.append(float(flat[msk].mean()
                              -flat[~msk].mean()))
        nn=torch.tensor(null)
        p=float((nn<=obs).float().mean()) if obs<0 else \
            float((nn>=obs).float().mean())
        return {'punct':round(a,4),'nonpunct':round(b,4),
                'diff':round(obs,4),'p_perm':round(p,4),
                'overall':round(float(d.mean()),4)}
    own=score(ce_all(own_hooks)-base)
    rnds=[score(ce_all(lambda s=s: rnd_hooks(s))-base)
          for s in (301,302,303)]
    pa=(own['diff']<0 and own['p_perm']<=0.01)
    pb=all(r['p_perm']>0.01 or r['diff']>=0 for r in rnds)
    pc=(abs(own['overall'])>0.01 and
        all(abs(r['overall'])>0.01 for r in rnds))
    out={'own':own,'random_subspaces':rnds,'n_rows':NROWS,
         'n_punct':int(pm.sum()),'pred_a':bool(pa),
         'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('own:',own); print('random:',rnds)
    for nm,v in (('a','own punct effect generalizes (p<=0.01)'),
                 ('b','random subspaces do not'),
                 ('c','both ablations non-trivial')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
