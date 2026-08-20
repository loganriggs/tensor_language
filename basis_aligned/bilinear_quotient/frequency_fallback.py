"""FREQUENCY FALLBACK -- 455: the punctuation effect is NOT
leaf-specific after all (454): mean-ablating whole components
reproduces it for a3, a7, a6 -- and for m7, which belongs to NONE
of the three bundles (excess 0.254 and 0.208, p=0.0005/0.0024).
Only a12 is clean. So several different ablations help at
punctuation targets, while 16-dimensional RANDOM subspaces do not
(453). A single explanation covers both facts and connects to
442: the sink constant is a UNIGRAM-FREQUENCY PRIOR, and damaging
the model makes it fall back toward that prior -- which helps
wherever the true next token is high-frequency, and punctuation
is the most frequent class there is.
Test the fallback story directly.
  (1) Bin every member target by corpus unigram frequency and
      measure help-rate per bin under a whole-component ablation.
  (2) Ask whether punctuation still carries excess AFTER
      conditioning on frequency.
  (3) Measure whether ablation moves the model's predictions
      TOWARD the unigram distribution (KL to unigram).
REGISTERED PREDICTIONS:
  (a) LADDER: help-rate rises monotonically across four unigram
      frequency quartiles under ablation;
  (b) EXPLAINED: within the top frequency quartile, punctuation
      targets show excess <= 0.10 over non-punctuation targets --
      i.e. frequency explains the punctuation effect;
  (c) TOWARD THE PRIOR: mean KL(prediction || unigram) DROPS
      under ablation relative to intact."""
import json, time, math, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'frequency_fallback_results.json'
TAG='r.13.2.1'; COMP='a7'
NR=24

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    rows=cl.rows()
    # unigram frequency over the whole census corpus
    flat=rows[:,:257].reshape(-1)
    cnt=torch.bincount(flat,minlength=50257).float()
    uni=cnt/cnt.sum()
    loguni=(uni+1e-12).log()
    # member-level analysis under a whole-component ablation
    mus=cl.comp_means()
    mod={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    mod.update({f'm{li}':m.transformer.h[li].mlp
                for li in range(18)})[COMP] if False else None
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def hooks():
        mu=mus[COMP].to(DEV)
        def fh(mo,i_,o_,mu=mu):
            y,v1=o_
            return (mu.expand_as(y).to(y.dtype),v1)
        return [MODS[COMP].register_forward_hook(fh)]
    d=cl.ce_sweep(hooks())-cl.base_ce()
    lf=cl.leaf(TAG); mem=lf['member']
    qs=uni.quantile(torch.tensor([0.0]))  # placeholder
    # frequency of each member's TARGET token
    tgt=torch.tensor([int(rows[int(g)//256,int(g)%256+1])
                      for g in mem.tolist()])
    f=uni[tgt]
    qb=torch.quantile(f,torch.tensor([0.25,0.5,0.75]))
    bins=torch.bucketize(f,qb)
    ispunct=torch.tensor([
        (lambda s: bool(s) and not any(c.isalnum() for c in s))(
            cl.d1(int(t)).strip()) for t in tgt])
    dm=d[mem]
    ladder=[]
    for b in range(4):
        msk=(bins==b)
        ladder.append({'bin':b,'n':int(msk.sum()),
                       'help_rate':round(float((dm[msk]<0)
                                               .float().mean()),3),
                       'median_freq':round(float(f[msk].median()),6)})
        print(f"freq bin {b}: n {ladder[-1]['n']} help "
              f"{ladder[-1]['help_rate']}",flush=True)
    top=(bins==3)
    pt=top&ispunct; npt=top&(~ispunct)
    hp=float((dm[pt]<0).float().mean()) if int(pt.sum()) else 0.0
    hn=float((dm[npt]<0).float().mean()) if int(npt.sum()) else 0.0
    print(f'top-frequency quartile: punct {hp:.3f} (n={int(pt.sum())})'
          f' vs non-punct {hn:.3f} (n={int(npt.sum())})',flush=True)
    # KL to unigram, intact vs ablated
    kls={'intact':0.0,'ablated':0.0}; nk=0
    for i in range(0,NR,4):
        bb=rows[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        for arm in ('intact','ablated'):
            hs=hooks() if arm=='ablated' else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            lp=F.log_softmax(lg,dim=-1)
            p=lp.exp()
            kl=(p*(lp-loguni.to(DEV)[None,None,:])).sum(-1)
            kls[arm]+=float(kl.mean()); 
            for h in hs: h.remove()
        nk+=1
        print(f'batch {i} kl done',flush=True)
    KL={k:round(v/max(nk,1),4) for k,v in kls.items()}
    hr=[l['help_rate'] for l in ladder]
    pa=all(hr[i]<=hr[i+1]+1e-9 for i in range(3))
    pb=(hp-hn)<=0.10
    pc=KL['ablated']<KL['intact']
    out={'ladder':ladder,'top_quartile_punct_help':round(hp,3),
         'top_quartile_nonpunct_help':round(hn,3),
         'punct_excess_within_top_quartile':round(hp-hn,3),
         'kl_to_unigram':KL,'component':COMP,'tag':TAG,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('help-rate ladder:',hr,'| KL to unigram:',KL)
    for nm,v in (('a','help-rate rises with unigram frequency'),
                 ('b','frequency explains the punct effect'),
                 ('c','ablation moves predictions toward the prior')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
