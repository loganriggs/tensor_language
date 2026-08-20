"""BAND UNIT -- 474: the census's one leaf-specific enrichment
(m14 -> m15, siblings r.1.2.2/r.1.2.0) FAILS causal specificity
(473): ablating m14 gives concentration 4.33 and 4.29, but the
adjacent m13 -- which the table does NOT flag -- gives 4.44 and
4.89. Same story as a14/a13 (468). Enrichment ranks magnitude well
(469: rho 0.76) but does not isolate a single writer from its
neighbours.
So maybe the writer is the wrong UNIT. If selectivity is smooth
over adjacent layers, the natural object is a contiguous BAND, and
the testable question is whether that band has a sharp boundary.
Arms on both sibling leaves (mean-ablation):
  m14, m13 alone                    (reference, from 473)
  band m13+m14
  band m12+m13+m14                  (does extending help?)
  distant band m8+m9 (width 2)      (control)
REGISTERED PREDICTIONS:
  (a) BAND BEATS PARTS: m13+m14 damages members more than either
      alone on both siblings;
  (b) SHARP BOUNDARY: adding m12 adds < 20% more member damage
      than m13+m14 alone;
  (c) LOCALITY: the distant band m8+m9 causes less than half the
      member damage of m13+m14."""
import json, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'band_unit_results.json'
TAGS=['r.1.2.2','r.1.2.0']

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def hooks(keys):
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
    ARMS={'m14':['m14'],'m13':['m13'],
          'band_13_14':['m13','m14'],
          'band_12_13_14':['m12','m13','m14'],
          'distant_8_9':['m8','m9']}
    ALL={}
    for nm,keys in ARMS.items():
        torch.cuda.empty_cache()
        d=cl.ce_sweep(hooks(keys))-cl.base_ce()
        for TAG in TAGS:
            st=cl.sign_stats(TAG,d)
            ALL.setdefault(TAG,{})[nm]={
                'concentration':st['concentration'],
                'abs_dce_members':st['abs_dce_members'],
                'abs_dce_offslice':st['abs_dce_offslice']}
        print(f"{nm}: "+" | ".join(
            f"{t} conc {ALL[t][nm]['concentration']} mem "
            f"{ALL[t][nm]['abs_dce_members']}" for t in TAGS),
            flush=True)
        json.dump(ALL,open(OUT,'w'),indent=1)
    def mem(t,a): return ALL[t][a]['abs_dce_members']
    pa=all(mem(t,'band_13_14')>max(mem(t,'m13'),mem(t,'m14'))
           for t in TAGS)
    pb=all((mem(t,'band_12_13_14')-mem(t,'band_13_14'))
           <0.20*mem(t,'band_13_14') for t in TAGS)
    pc=all(mem(t,'distant_8_9')<0.5*mem(t,'band_13_14')
           for t in TAGS)
    out={'leaves':ALL,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),
         'band_over_parts':{t:round(mem(t,'band_13_14')
                            /max(mem(t,'m14'),1e-6),3)
                            for t in TAGS},
         'extension_gain':{t:round((mem(t,'band_12_13_14')
                           -mem(t,'band_13_14'))
                           /max(mem(t,'band_13_14'),1e-6),3)
                           for t in TAGS},
         'runtime_s':time.time()-t0}
    print(f"band/parts {out['band_over_parts']} | extension gain "
          f"{out['extension_gain']}")
    for nm,v in (('a','band beats either part'),
                 ('b','sharp boundary (m12 adds <20%)'),
                 ('c','distant band <half the damage')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
