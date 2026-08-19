"""NET UTILITY -- user question (optimality): census circuits often
IMPROVE their members when deleted; is that consistent with being
learned? Test: corpus-wide signed net dCE of each leaf's own
ablation, for the 10 highest-concentration leaves. Optimality
predicts net damage > 0 globally (the machinery earns its keep in
expectation); member-tail improvement is a selection effect.
REGISTERED PREDICTIONS:
  (a) >=9/10 leaves are globally net-positive damage (deleting
      hurts the corpus overall);
  (b) for every leaf, global net is SMALLER in magnitude than
      member-mean |dCE| by >=3x (the census zooms into tails);
  (c) any net-negative leaf is named (a circuit the model would be
      better off without ON THIS CORPUS -- prime slack-harvest
      target)."""
import json, time, torch
import census_lib as cl
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'net_utility_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    eb=json.load(open(PT+'explainer_batch_results.json'))
    leaves=sorted(eb['leaves'],key=lambda r:-r['conc'])[:10]
    res=[]
    for r in leaves:
        tag=r['tag']
        d=cl.leaf_ablate(tag)
        lf=cl.leaf(tag)
        mm=torch.zeros(54272,dtype=torch.bool); mm[lf['member']]=True
        row={'tag':tag,'global_net':round(float(d.mean()),4),
             'member_mean':round(float(d[mm].mean()),3),
             'member_abs':round(float(d[mm].abs().mean()),3)}
        row['ratio']=round(row['member_abs']/max(abs(row['global_net']),
                                                 1e-4),1)
        res.append(row); print(row,flush=True)
    npos=sum(1 for r_ in res if r_['global_net']>0)
    pa=npos>=9
    pb=all(r_['ratio']>=3 for r_ in res)
    neg=[r_['tag'] for r_ in res if r_['global_net']<=0]
    out={'leaves':res,'n_net_positive':npos,'net_negative':neg,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f'net-positive damage: {npos}/10 | net-negative leaves: {neg}')
    print(f"(a) >=9/10 globally useful: {'HELD' if pa else 'FAILED'}")
    print(f"(b) member/|global| >=3x all: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
