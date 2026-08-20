"""A14 PATHWAY -- 468: with the punctuation arc closed, return to
the program's ONE confirmed mechanism claim. r.3.0.2's machinery
(a15, a16, a17) shows writer a14 enriched at 2.37 [2.06-2.85],
2.43 [2.09-2.96] and 2.36 [2.05-2.84] -- bootstrap-stable
(ENRICHED_STABLE2 against thresholds 1.34-1.89), survives
projecting out the stream centre (450: 3.02 against 3.08), and
reviewer-confirmed with the caveat that the a15 leg alone
reproduces on an unrelated leaf (425). The a16/a17 legs are the
family-specific part.
Enrichment is a correlational statement about input composition.
Test whether the pathway is CAUSAL.
Arms (mean-ablation, operator-C):
  a14 alone      -- does removing the enriched writer damage this
                    leaf's members selectively?
  a13 control    -- an adjacent-depth component that is NOT the
                    enriched writer
  bundle alone   -- the leaf's own probes (reference profile)
  a14 + bundle   -- subadditive if they act on one pathway
REGISTERED PREDICTIONS:
  (a) CAUSAL: ablating a14 gives concentration >= 2 on r.3.0.2's
      members (member |dCE| over off-slice |dCE|);
  (b) SPECIFIC: the a13 control gives concentration < 1.5;
  (c) ONE PATHWAY: the joint arm's member damage is < 1.3x the
      larger of the two individual arms (subadditive)."""
import json, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'a14_pathway_results.json'
TAG='r.3.0.2'

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def comp_hooks(keys):
        hs=[]
        for key in keys:
            mu=mus[key].to(DEV); mod=MODS[key]
            if key[0]=='a':
                def fh(mo,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
            else:
                def fh(mo,i_,o_,mu=mu):
                    return mu.expand_as(o_).to(o_.dtype)
            hs.append(mod.register_forward_hook(fh))
        return hs
    def bundle_hooks():
        return cl.proj_hooks(cl.leaf(TAG)['top_probes'])
    arms={'a14':lambda: comp_hooks(['a14']),
          'a13':lambda: comp_hooks(['a13']),
          'bundle':bundle_hooks,
          'a14+bundle':lambda: comp_hooks(['a14'])+bundle_hooks()}
    res={}
    for nm,mk in arms.items():
        torch.cuda.empty_cache()
        d=cl.ce_sweep(mk())-cl.base_ce()
        s=cl.sign_stats(TAG,d)
        res[nm]={'concentration':s['concentration'],
                 'abs_dce_members':s['abs_dce_members'],
                 'abs_dce_offslice':s['abs_dce_offslice'],
                 'dce_members':s['dce_members']}
        print(f"{nm}: conc {s['concentration']} members "
              f"{s['abs_dce_members']} off {s['abs_dce_offslice']}",
              flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    pa=res['a14']['concentration']>=2
    pb=res['a13']['concentration']<1.5
    big=max(res['a14']['abs_dce_members'],
            res['bundle']['abs_dce_members'])
    pc=res['a14+bundle']['abs_dce_members']<1.3*big
    out={'arms':res,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),
         'joint_over_larger':round(
             res['a14+bundle']['abs_dce_members']/max(big,1e-6),3),
         'runtime_s':time.time()-t0}
    print(f"joint / larger individual: {out['joint_over_larger']}")
    for nm,v in (('a','a14 ablation is selective (conc>=2)'),
                 ('b','a13 control is not (conc<1.5)'),
                 ('c','a14 and the bundle share one pathway')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
