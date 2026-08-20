"""PUNCT CARRIER -- 453: the three punctuation claims are ONE
shared effect, not three (452/453: ablating all three bundles
jointly produces LESS excess than any bundle alone -- 0.25/0.27/
0.20 joint against 0.31/0.29/0.23 individual -- and two of three
random-subspace controls are clean). The bundles differ but their
components overlap: r.18.2.0 uses a7,a9; r.13.2.1 uses a7,a6,a3;
r.11.1.2 uses a8,a3,a4. Locate the carrier by ablating single
whole components (mean-ablation, the operator-C form used
throughout this program) and scoring the same punctuation
population test on each of the three leaves' member sets.
Components tested: the shared ones (a3, a7, a8) and the
non-shared ones (a4, a6, a9), plus two components in NO bundle
(a12, m7) as controls.
REGISTERED PREDICTIONS:
  (a) A CARRIER EXISTS: at least one single component reproduces
      a punct excess >= 0.15 on at least two of the three leaves;
  (b) IT IS SHARED: that component is one of a3, a7, a8;
  (c) CONTROLS CLEAN: a12 and m7 produce no punct effect at
      p <= 0.0083 on any leaf."""
import json, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_carrier_results.json'
TAGS=['r.18.2.0','r.13.2.1','r.11.1.2']
COMPS=['a3','a7','a8','a4','a6','a9','a12','m7']

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def comp_hooks(key):
        mu=mus[key].to(DEV); mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]
    res={}
    for key in COMPS:
        try:
            torch.cuda.empty_cache()
            d=cl.ce_sweep(comp_hooks(key))-cl.base_ce()
            row={}
            for tag in TAGS:
                st=cl.story_test_class(tag,d,'punct',True,
                                       n_tests=12)
                p=st['population']
                row[tag]={'n':p['n'],'hits':p['hits'],
                          'p':p['p_value'],
                          'excess':round(p['hits']/max(p['n'],1)
                                         -p['base_rate_help'],3),
                          'ROBUST_V2':st['ROBUST_V2']}
            res[key]=row
            print(f"{key}: "+" | ".join(
                f"{t} exc {row[t]['excess']} p {row[t]['p']}"
                for t in TAGS),flush=True)
            json.dump(res,open(OUT,'w'),indent=1)
        except Exception as e:
            print(f'{key}: SKIPPED ({type(e).__name__}: {e})',
                  flush=True)
            torch.cuda.empty_cache()
    def nbig(key):
        return sum(1 for t in TAGS
                   if res.get(key,{}).get(t,{}).get('excess',0)
                   >=0.15)
    carriers=[k for k in COMPS if k in res and nbig(k)>=2]
    pa=len(carriers)>0
    pb=any(k in ('a3','a7','a8') for k in carriers)
    pc=all(res.get(k,{}).get(t,{}).get('p',1)>0.0083
           for k in ('a12','m7') if k in res for t in TAGS)
    out={'components':res,'carriers':carriers,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'carriers (excess>=0.15 on >=2 leaves): {carriers}')
    for nm,v in (('a','a single-component carrier exists'),
                 ('b','the carrier is a shared component'),
                 ('c','a12 and m7 controls clean')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
