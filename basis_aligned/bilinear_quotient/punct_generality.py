"""PUNCT GENERALITY -- 432: two independent leaves (r.18.2.0,
r.13.2.1) now carry ROBUST_V2 behavioral claims of the same shape
-- their machinery helps prediction at PUNCTUATION targets (36/43
and 39/49 vs ~51% base rates, both p~0 under Bonferroni). Either
punctuation-specific pushing is a real shared function of these
bundles, or punctuation positions are generically sign-sensitive
to ANY ablation of matched rank -- the same fragility confound
that 430 found for concentration. Decide it: rerun each claim
with a RANK-MATCHED RANDOM SUBSPACE ablated in the same
components, and score the identical punct population test.
Also test three leaves with no punct claim as negative controls.
REGISTERED PREDICTIONS:
  (a) LEAF-SPECIFIC: for both claim leaves, the random-subspace
      ablation does NOT reproduce a punct effect at the corrected
      threshold (population p > 0.0083);
  (b) if (a) FAILS, both punct claims are fragility artifacts and
      the records must be corrected -- stated either way;
  (c) controls: the three no-claim leaves show no punct effect
      under their own probes (p > 0.0083)."""
import json, ast, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_generality_results.json'
CLAIM=['r.18.2.0','r.13.2.1']
CTRL=['r.5.3.1','r.5.0.1','r.3.0.2']

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def rank_matched_random(tag):
        probes=[ast.literal_eval(p) if isinstance(p,str) else p
                for p in cl.leaf(tag)['top_probes']]
        rk={}
        for p in probes:
            key=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
            n=(p[3][1]-p[3][0]) if p[0]=='pca' else 8
            rk[key]=rk.get(key,0)+n
        hs=[]
        for key,n in rk.items():
            gg=torch.Generator(device=DEV).manual_seed(101)
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
        return hs,rk
    res={}
    for tag in CLAIM+CTRL:
        try:
            torch.cuda.empty_cache()
            d_own=cl.leaf_ablate(tag)
            own=cl.story_test_class(tag,d_own,'punct',True,
                                    n_tests=12)
            hs,rk=rank_matched_random(tag)
            d_rnd=cl.ce_sweep(hs)-cl.base_ce()
            rnd=cl.story_test_class(tag,d_rnd,'punct',True,
                                    n_tests=12)
            res[tag]={'ranks':rk,
                'own':{'n':own['population']['n'],
                       'hits':own['population']['hits'],
                       'p':own['population']['p_value'],
                       'ROBUST_V2':own['ROBUST_V2']},
                'random_subspace':{'n':rnd['population']['n'],
                       'hits':rnd['population']['hits'],
                       'p':rnd['population']['p_value'],
                       'ROBUST_V2':rnd['ROBUST_V2']}}
            print(f"{tag}: own {res[tag]['own']} | random "
                  f"{res[tag]['random_subspace']}",flush=True)
            json.dump(res,open(OUT,'w'),indent=1)
        except Exception as e:
            print(f'{tag}: SKIPPED ({type(e).__name__}: {e})',
                  flush=True)
            torch.cuda.empty_cache()
    pa=all(res[t]['random_subspace']['p']>0.0083
           for t in CLAIM if t in res)
    pc=all(res[t]['own']['p']>0.0083 for t in CTRL if t in res)
    out={'leaves':res,'pred_a':bool(pa),'pred_b':True,
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    for nm,v in (('a','random subspace does NOT reproduce punct'),
                 ('b','verdict stated either way'),
                 ('c','control leaves show no punct effect')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
