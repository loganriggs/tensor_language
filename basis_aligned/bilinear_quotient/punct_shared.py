"""PUNCT SHARED -- 451: three independent leaves now carry
ROBUST_V2 punctuation claims -- r.18.2.0 (36/43), r.13.2.1
(39/49, generalized to fresh FineWeb and survived a
random-subspace control), r.11.1.2 (36/51, p=0.0007) -- with
different machinery (a7/a9; a7/a6/a3; a8/a3/a4). Either these are
three views of ONE effect, or punctuation prediction is helped by
ablating many different things. Decide it.
Arms:
  each leaf's own bundle, alone            (individual effects)
  all three bundles ablated jointly        (subadditive => one
                                            shared effect;
                                            additive => three)
  rank-matched random subspaces per leaf   (control, extends
                                            436 to all three)
Also measured: pairwise principal-angle overlap between the three
leaves' probe subspaces, against random subspaces of the same
ranks.
REGISTERED PREDICTIONS:
  (a) SHARED: the joint ablation's punct hit-rate excess over
      base is <= 1.3x the largest individual excess (subadditive);
  (b) GEOMETRY: mean pairwise subspace overlap between the three
      bundles is >= 2x the random-subspace overlap;
  (c) CONTROLS: no random-subspace arm reaches p <= 0.0083 on the
      punct population test."""
import json, ast, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_shared_results.json'
TAGS=['r.18.2.0','r.13.2.1','r.11.1.2']

def subspace(tag):
    lf=cl.leaf(tag)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    rk={}
    for p in probes:
        key=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
        n=(p[3][1]-p[3][0]) if p[0]=='pca' else 8
        rk[key]=rk.get(key,0)+n
    return rk

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def rnd_hooks(rk,seed):
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
    res={}
    # individual + control arms, scored on each leaf's own punct pop
    for tag in TAGS:
        d=cl.leaf_ablate(tag)
        own=cl.story_test_class(tag,d,'punct',True,n_tests=12)
        rk=subspace(tag)
        dr=cl.ce_sweep(rnd_hooks(rk,404))-cl.base_ce()
        rnd=cl.story_test_class(tag,dr,'punct',True,n_tests=12)
        res[tag]={'own':own['population'],
                  'own_ROBUST':own['ROBUST_V2'],
                  'random':rnd['population'],
                  'random_ROBUST':rnd['ROBUST_V2'],'ranks':rk}
        print(f"{tag}: own {own['population']} | random "
              f"{rnd['population']}",flush=True)
    # joint ablation: every bundle at once
    allhooks=[]
    for tag in TAGS:
        allhooks+=cl.proj_hooks(cl.leaf(tag)['top_probes'])
    dj=cl.ce_sweep(allhooks)-cl.base_ce()
    joint={}
    for tag in TAGS:
        j=cl.story_test_class(tag,dj,'punct',True,n_tests=12)
        joint[tag]=j['population']
        print(f"joint scored on {tag}: {j['population']}",
              flush=True)
    def excess(p): return p['hits']/max(p['n'],1) \
        -p['base_rate_help']
    ind=[excess(res[t]['own']) for t in TAGS]
    jex=[excess(joint[t]) for t in TAGS]
    pa=max(jex)<=1.3*max(ind)
    pc=all(res[t]['random']['p_value']>0.0083 for t in TAGS)
    out={'per_leaf':res,'joint':joint,
         'individual_excess':[round(x,3) for x in ind],
         'joint_excess':[round(x,3) for x in jex],
         'pred_a':bool(pa),'pred_b':None,'pred_c':bool(pc),
         'note':'geometry leg (b) omitted: probe subspaces are '
                'slice-conditioned PCA blocks recomputed per leaf '
                'and not directly comparable as fixed bases; '
                'recorded as not-run rather than faked',
         'runtime_s':time.time()-t0}
    print(f'individual excess {out["individual_excess"]} | joint '
          f'{out["joint_excess"]}')
    for nm,v in (('a','joint <=1.3x largest individual (shared)'),
                 ('c','no random-subspace control reaches p<=0.0083')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    print('(b) geometry leg: NOT RUN (see note)')
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
