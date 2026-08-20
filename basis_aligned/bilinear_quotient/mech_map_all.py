"""MECH MAP ALL -- 471: the enrichment table predicts causal
selectivity (469: rho 0.842, and 0.762 with the leaf's own
machinery excluded -- 470's correction), and on leaves whose table
is negative the whole profile is flat (range ~2 against 7.1), so
the tool's positives are worth chasing and its negatives are worth
believing. Use it at census scale.
Run the mechanism table over a large sample of the certified
shortlist and aggregate: how often does a leaf have ANY stable
enrichment, which writers recur across leaves, and is the
adjacency effect (425: a14 dominating a15's input reproduces on
unrelated leaves) the rule or the exception?
REGISTERED PREDICTIONS:
  (a) POSITIVES ARE RARE: fewer than 25% of sampled leaves carry
      at least one ENRICHED_STABLE2 component (the swarm's 7-leaf
      experience says most tables are negative);
  (b) CONCENTRATION: among the positive (leaf, writer) pairs, the
      top three writers account for >= 40% of them;
  (c) ADJACENCY: enriched writers sit within two layers of the
      component they feed in >= 60% of positive pairs -- i.e. the
      tool mostly finds local structure, which is what 425
      warned about."""
import json, os, sys, time, subprocess, collections
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_map_all_results.json'
NLEAF=60

def main():
    t0=time.time()
    import torch
    sl=json.load(open(PT+'swarm_shortlist.json'))
    sl=[t for t in sl if t.count('.')>=2]
    g=torch.Generator().manual_seed(41)
    sel=[sl[i] for i in torch.randperm(len(sl),generator=g)
         [:NLEAF].tolist()]
    done=0; res={}
    for tag in sel:
        f=PT+f'leaf_mech/{tag}.json'
        if not os.path.exists(f):
            r=subprocess.run([sys.executable,'-u',
                              PT+'leaf_input_decomp.py',tag],
                             cwd=PT,capture_output=True,text=True,
                             timeout=900)
            if not os.path.exists(f):
                print(f'{tag}: FAILED\n{r.stdout[-300:]}',
                      flush=True)
                continue
        try: t=json.load(open(f))
        except Exception: continue
        pos=[]
        for comp,tab in t['tables'].items():
            if tab.get('ENRICHED_STABLE2'):
                w=tab['top'][0][0]
                pos.append({'component':comp,'writer':w,
                            'min_ratio':tab['top_ratio_min'],
                            'threshold':tab.get('threshold_v2')})
        res[tag]={'machinery':t['machinery'],'positives':pos}
        done+=1
        if pos: print(f'{tag}: {pos}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    npos=[t for t in res if res[t]['positives']]
    pairs=[(t,p['component'],p['writer'])
           for t in res for p in res[t]['positives']]
    wc=collections.Counter(w for _,_,w in pairs)
    def lay(x):
        try: return int(x[1:])
        except Exception: return -99
    adj=sum(1 for _,c,w in pairs if abs(lay(c)-lay(w))<=2)
    frac_pos=len(npos)/max(len(res),1)
    top3=sum(c for _,c in wc.most_common(3))
    pa=frac_pos<0.25
    pb=(len(pairs)>0 and top3/len(pairs)>=0.40)
    pc=(len(pairs)>0 and adj/len(pairs)>=0.60)
    out={'n_leaves':len(res),'n_with_positive':len(npos),
         'frac_positive':round(frac_pos,3),
         'n_positive_pairs':len(pairs),
         'writer_counts':dict(wc.most_common(10)),
         'adjacency_fraction':round(adj/max(len(pairs),1),3),
         'leaves':res,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f'{len(res)} leaves scanned | {len(npos)} with a '
          f'positive ({frac_pos:.1%}) | {len(pairs)} positive '
          f'pairs')
    print('top writers:',wc.most_common(6))
    print(f'adjacency fraction {out["adjacency_fraction"]}')
    for nm,v in (('a','positives are rare (<25% of leaves)'),
                 ('b','top-3 writers hold >=40% of pairs'),
                 ('c','enriched writers are local (>=60%)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
