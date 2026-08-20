"""M14 PATHWAY -- 473: the census-scale screen plus specificity
check found exactly ONE genuinely leaf-specific input mechanism in
60 leaves (471/472): m14 -> m15, carried at min ratio 2.333 and
2.277 by the SIBLING leaves r.1.2.2 and r.1.2.0, against peer
leaves sharing m15 at 0.934, 1.167 and 0.691. (The third positive,
m15 -> m17 on r.6.2.0, is a layer property -- a peer scores
higher.)
Escalate it to a causal claim, with the bar design corrected by
468's failures: no absolute control threshold (adjacent components
are partly selective everywhere), and no assumption of one shared
pathway.
Arms per sibling leaf: m14 alone, m13 control, the leaf's own
bundle, and m14 + bundle jointly.
REGISTERED PREDICTIONS:
  (a) CAUSAL ON BOTH: ablating m14 gives concentration >= 2 on
      BOTH r.1.2.2 and r.1.2.0;
  (b) RELATIVE SPECIFICITY: m14's concentration exceeds the m13
      control's on both leaves (a relative bar -- 468 showed an
      absolute one is wrong);
  (c) ONE CIRCUIT, TWO VIEWS: the two siblings' m14 concentrations
      agree within 25% of each other, as two views of one circuit
      should."""

import json, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m14_pathway_results.json'
TAGS=['r.1.2.2','r.1.2.0']

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
    ALL={}
    for TAG in TAGS:
        def bh(TAG=TAG):
            return cl.proj_hooks(cl.leaf(TAG)['top_probes'])
        arms={'m14':lambda: comp_hooks(['m14']),
              'm13':lambda: comp_hooks(['m13']),
              'bundle':bh,
              'm14+bundle':lambda: comp_hooks(['m14'])+bh()}
        res={}
        for nm,mk in arms.items():
            torch.cuda.empty_cache()
            d=cl.ce_sweep(mk())-cl.base_ce()
            st=cl.sign_stats(TAG,d)
            res[nm]={'concentration':st['concentration'],
                     'abs_dce_members':st['abs_dce_members'],
                     'abs_dce_offslice':st['abs_dce_offslice']}
            print(f"{TAG} {nm}: conc {st['concentration']} "
                  f"members {st['abs_dce_members']}",flush=True)
        ALL[TAG]=res
        json.dump(ALL,open(OUT,'w'),indent=1)
    res=ALL[TAGS[0]]
    c14=[ALL[t]['m14']['concentration'] for t in TAGS]
    c13=[ALL[t]['m13']['concentration'] for t in TAGS]
    pa=all(c>=2 for c in c14)
    pb=all(a>b for a,b in zip(c14,c13))
    pc=(abs(c14[0]-c14[1])/max(max(c14),1e-6))<=0.25
    add={}
    for t in TAGS:
        r=ALL[t]
        big=max(r['m14']['abs_dce_members'],
                r['bundle']['abs_dce_members'])
        add[t]=round(r['m14+bundle']['abs_dce_members']
                     /max(big,1e-6),3)
    out={'leaves':ALL,'m14_concentrations':c14,
         'm13_concentrations':c13,'joint_over_larger':add,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'm14 {c14} vs m13 {c13} | joint/larger {add}')
    for nm,v in (('a','m14 selective on both siblings (>=2)'),
                 ('b','m14 beats the m13 control on both'),
                 ('c','the two siblings agree within 25%')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
