"""PUNCT CONFIDENCE -- 463: the unselected run (462) refuted the
over-continuation story outright. Across ALL 1602 punctuation
targets on fresh text the intact model's top-1 is a
non-punctuation token only 23.5% of the time -- it gets
punctuation right three times out of four. The 75% figure was
pure selection.
What IS unselected and replicates: ablation damages punctuation
LESS than everything else. Bundle: punct -0.0098 against
non-punct +0.0153 (dissociation -0.025). Five components: punct
+0.325 against non-punct +0.746 (dissociation -0.422). Both
interventions spare punctuation; they differ only in overall
severity -- so 461's "conflict" was an artifact of my oracle
design (it compared ablate-only-at-punct against no-ablation,
not against ablate-everywhere).
Simplest remaining account: punctuation is spared because it is
PREDICTABLE -- high-confidence predictions survive damage, and
punctuation is high-confidence. If true, the sparing should
vanish once intact confidence is matched, and nothing about
punctuation per se is needed.
REGISTERED PREDICTIONS:
  (a) CONFIDENCE EXPLAINS IT: after matching punctuation and
      non-punctuation positions on intact top-1 probability
      (deciles), the punctuation sparing shrinks to < 25% of its
      unmatched value;
  (b) CONFIDENCE PREDICTS SPARING: across all positions, damage
      (dCE) decreases monotonically with intact confidence
      decile;
  (c) if (a) FAILS, punctuation retains sparing beyond
      confidence and something class-specific remains -- recorded
      either way."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_confidence_results.json'
TAG='r.13.2.1'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    ispunct=lambda t:(lambda s: bool(s) and
                      not any(c.isalnum() for c in s))(
                          cl.d1(int(t)).strip())
    fresh=cl.fineweb_rows(NFRESH)
    pm=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            pm[r,q]=ispunct(int(fresh[r,q+1]))
    conf=torch.zeros(NFRESH,T); dce=torch.zeros(NFRESH,T)
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]
        def fwd(mk):
            hs=cl.proj_hooks(cl.leaf(TAG)['top_probes']) if mk \
                else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                              reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
            return lg,c
        lg0,c0=fwd(False); _,c1=fwd(True)
        conf[i:i+4]=F.softmax(lg0,-1).max(-1).values.cpu()
        dce[i:i+4]=c1-c0
        print(f'batch {i} done',flush=True)
    cf=conf.reshape(-1); dd=dce.reshape(-1); pp=pm.reshape(-1)
    unmatched=float(dd[pp].mean()-dd[~pp].mean())
    qs=torch.quantile(cf,torch.linspace(0.1,0.9,9))
    dec=torch.bucketize(cf,qs)
    ladder=[]
    for b in range(10):
        msk=(dec==b)
        ladder.append({'decile':b,'n':int(msk.sum()),
                       'mean_conf':round(float(cf[msk].mean()),3),
                       'mean_dce':round(float(dd[msk].mean()),4)})
    # confidence-matched difference: within-decile, weighted by
    # the punctuation distribution
    num=den=0.0
    for b in range(10):
        mp=(dec==b)&pp; mn=(dec==b)&(~pp)
        if int(mp.sum())<5 or int(mn.sum())<5: continue
        w=float(mp.sum())
        num+=w*(float(dd[mp].mean())-float(dd[mn].mean()))
        den+=w
    matched=num/max(den,1e-9)
    md=[l['mean_dce'] for l in ladder]
    pa=abs(matched)<0.25*abs(unmatched)
    pb=all(md[i]>=md[i+1]-1e-6 for i in range(9))
    out={'n_punct':int(pp.sum()),'n_total':int(pp.numel()),
         'unmatched_punct_sparing':round(unmatched,4),
         'confidence_matched_sparing':round(matched,4),
         'ratio_matched_to_unmatched':round(
             matched/unmatched,3) if unmatched else None,
         'confidence_ladder':ladder,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'runtime_s':time.time()-t0}
    print(f'unmatched sparing {unmatched:+.4f} | '
          f'confidence-matched {matched:+.4f} '
          f'({out["ratio_matched_to_unmatched"]} of it)')
    print('dCE by confidence decile:',md)
    for nm,v in (('a','confidence explains the sparing'),
                 ('b','damage falls with confidence'),
                 ('c','verdict recorded either way')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
