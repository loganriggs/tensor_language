"""PUNCT UNSELECTED -- 462: the oracle ceiling (461) came back
BACKWARDS -- gating the five components' mean-ablation to true
punctuation positions on fresh rows makes CE WORSE (+0.117 at
punctuation, +0.027 overall), when the whole arc predicted better.
That exposes a selection problem in my own chain: 457's competitor
statistics were computed at positions chosen BECAUSE ablation
helped there (d < 0). Conditioning on the outcome and then
reporting that the outcome holds is circular, and the report's
"75% of the time" figure inherits it.
This run measures the same quantities WITHOUT selection, on fresh
FineWeb rows, and separates the two interventions that the arc
conflated (a 16-dim probe BUNDLE versus five WHOLE components).
REGISTERED PREDICTIONS:
  (a) UNSELECTED OVER-CONTINUATION: across ALL punctuation
      targets on fresh rows, the intact model's top-1 is a
      non-punctuation token in >= 60% of cases (if this holds the
      claim survives without selection; if not it was an artifact
      of conditioning and the report must be corrected further);
  (b) BUNDLE REPLICATES: ablating r.13.2.1's 16-dim bundle on
      fresh rows lowers CE at punctuation relative to
      non-punctuation (440 was +0.024 in that direction);
  (c) INTERVENTIONS DIFFER: whole-component ablation of the five
      helpers does NOT show that dissociation on the same rows --
      confirming 461's reversal is real and the two interventions
      are not interchangeable."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_unselected_results.json'
HELPERS=['a3','a6','a7','a8','m7']
BUNDLE_TAG='r.13.2.1'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    ispunct=lambda t:(lambda s: bool(s) and
                      not any(c.isalnum() for c in s))(
                          cl.d1(int(t)).strip())
    fresh=cl.fineweb_rows(NFRESH)
    pm=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            pm[r,q]=ispunct(int(fresh[r,q+1]))
    def comp_hooks():
        hs=[]
        for k in HELPERS:
            mu=mus[k].to(DEV)
            if k[0]=='a':
                def fh(mo,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,mu=mu):
                    return mu.expand_as(o_).to(o_.dtype)
            hs.append(MODS[k].register_forward_hook(fh))
        return hs
    def bundle_hooks():
        return cl.proj_hooks(cl.leaf(BUNDLE_TAG)['top_probes'])
    stats={'top1_nonpunct':0,'n_punct':0,'comp_dp':0.0,
           'bundle_dp':0.0,'rand_dp':0.0}
    ce={'base':[0.0,0.0,0,0],'bundle':[0.0,0.0,0,0],
        'comp':[0.0,0.0,0,0]}
    g0=torch.Generator().manual_seed(5)
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]
        def fwd(mk):
            hs=mk() if mk else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                              reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
            return lg,c
        lg0,c0=fwd(None)
        lgb,cb=fwd(bundle_hooks)
        lgc,cc=fwd(comp_hooks)
        p0=F.softmax(lg0,-1); pb=F.softmax(lgb,-1)
        pc=F.softmax(lgc,-1)
        mk=pm[i:i+4]
        for nm,c in (('base',c0),('bundle',cb),('comp',cc)):
            ce[nm][0]+=float(c[mk].sum()); ce[nm][2]+=int(mk.sum())
            ce[nm][1]+=float(c[~mk].sum())
            ce[nm][3]+=int((~mk).sum())
        for b in range(B):
            for q in range(T):
                if not mk[b,q]: continue
                stats['n_punct']+=1
                t1=int(lg0[b,q].argmax())
                stats['top1_nonpunct']+=int(not ispunct(t1))
                stats['comp_dp']+=float(pc[b,q,t1]-p0[b,q,t1])
                stats['bundle_dp']+=float(pb[b,q,t1]-p0[b,q,t1])
                rt=int(torch.randint(0,50257,(1,),generator=g0))
                stats['rand_dp']+=float(pb[b,q,rt]-p0[b,q,rt])
        print(f'batch {i} done',flush=True)
    n=max(stats['n_punct'],1)
    fr=stats['top1_nonpunct']/n
    out={'n_punct_targets':n,'top1_nonpunct_frac':round(fr,3),
         'mean_dprob_competitor_bundle':round(
             stats['bundle_dp']/n,5),
         'mean_dprob_competitor_component':round(
             stats['comp_dp']/n,5),
         'mean_dprob_random_bundle':round(stats['rand_dp']/n,8)}
    for nm in ('base','bundle','comp'):
        p=ce[nm][0]/max(ce[nm][2],1); q=ce[nm][1]/max(ce[nm][3],1)
        out[nm]={'punct':round(p,4),'nonpunct':round(q,4)}
    dbp=out['bundle']['punct']-out['base']['punct']
    dbn=out['bundle']['nonpunct']-out['base']['nonpunct']
    dcp=out['comp']['punct']-out['base']['punct']
    dcn=out['comp']['nonpunct']-out['base']['nonpunct']
    out['bundle_dissociation']=round(dbp-dbn,4)
    out['component_dissociation']=round(dcp-dcn,4)
    pa=fr>=0.60
    pb=(dbp-dbn)<0
    pc=(dcp-dcn)>=0
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc),'runtime_s':time.time()-t0})
    print(f"UNSELECTED: {n} punct targets, intact top-1 is "
          f"non-punct {fr:.3f}")
    print(f"bundle: punct {dbp:+.4f} nonpunct {dbn:+.4f} "
          f"(dissociation {dbp-dbn:+.4f})")
    print(f"components: punct {dcp:+.4f} nonpunct {dcn:+.4f} "
          f"(dissociation {dcp-dcn:+.4f})")
    print(f"competitor dprob: bundle {out['mean_dprob_competitor_bundle']}"
          f" component {out['mean_dprob_competitor_component']}"
          f" random {out['mean_dprob_random_bundle']}")
    for nm,v in (('a','unselected over-continuation >=60%'),
                 ('b','bundle dissociation replicates'),
                 ('c','component intervention differs')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
