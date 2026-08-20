"""GATE SPECIFICITY -- 427: the r.5.0.1 reviewer's objection,
promoted to an experiment. Census leaves are selected by
concentration (member |dCE| / off-slice |dCE|), but members run
high base CE (5.5-6.5 nats, only ~20% under 3), and fragile
positions may be more sensitive to ANY sufficiently large
ablation. Control: for each sampled leaf, compare its own probe
bundle against a RANK-MATCHED RANDOM SUBSPACE ablation in the
same components (same layers, same number of directions, random
orthonormal basis), scoring concentration the same way.
REGISTERED PREDICTIONS:
  (a) SPECIFIC: the leaf's own concentration >= 2x the
      random-subspace concentration for >= 70% of leaves;
  (b) the random-subspace concentration stays under 2.0 on
      median (a random ablation should not look selective);
  (c) if (a) FAILS, the census gate is partly a fragility
      detector and every concentration number in the program
      needs that caveat -- recorded either way."""
import json, sys, time, ast, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'gate_specificity_results.json'
NLEAF=12

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    tags=json.load(open(PT+'swarm_shortlist.json'))
    tags=[t for t in tags if t.count('.')>=2]
    # census_lib.proj_hooks only handles ('pca',...) probes; leaves
    # whose bundle contains comp/head probes are skipped (recorded)
    def allpca(t):
        try:
            return all((ast.literal_eval(p) if isinstance(p,str)
                        else p)[0]=='pca'
                       for p in cl.leaf(t)['top_probes'])
        except Exception: return False
    tags=[t for t in tags if allpca(t)]
    g=torch.Generator().manual_seed(31)
    tags=[tags[i] for i in torch.randperm(len(tags),
          generator=g)[:NLEAF].tolist()]
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    rows=[]
    for tag in tags:
        lf=cl.leaf(tag)
        probes=[ast.literal_eval(p) if isinstance(p,str) else p
                for p in lf['top_probes']]
        d_own=cl.leaf_ablate(tag)
        own=cl.sign_stats(tag,d_own)['concentration']
        # rank-matched random subspace in the same components
        rk={}
        for p in probes:
            key=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
            n=(p[3][1]-p[3][0]) if p[0]=='pca' else 8
            rk[key]=rk.get(key,0)+n
        hs=[]
        for key,n in rk.items():
            gg=torch.Generator(device=DEV).manual_seed(97)
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
        d_rnd=cl.ce_sweep(hs)-cl.base_ce()
        rnd=cl.sign_stats(tag,d_rnd)['concentration']
        rows.append({'tag':tag,'own':own,'random_subspace':rnd,
                     'ratio':round(own/max(rnd,1e-4),2),
                     'ranks':rk})
        print(f'{tag}: own {own} random {rnd} ratio '
              f'{rows[-1]["ratio"]}',flush=True)
    frac=sum(1 for r in rows if r['own']>=2*r['random_subspace']) \
        /len(rows)
    med=sorted(r['random_subspace'] for r in rows)[len(rows)//2]
    pa=frac>=0.70; pb=med<2.0
    out={'leaves':rows,'n_leaves_scored':len(rows),
         'note':'leaves with non-pca probe bundles skipped '
                '(proj_hooks limitation)',
         'frac_2x':round(frac,3),
         'median_random_concentration':med,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'runtime_s':time.time()-t0}
    print(f'frac own>=2x random: {frac:.2f} | median random '
          f'concentration {med}')
    for nm,v in (('a','own >=2x random for >=70%'),
                 ('b','median random concentration <2.0'),
                 ('c','caveat recorded either way')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
