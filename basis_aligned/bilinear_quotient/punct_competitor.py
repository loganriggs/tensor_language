"""PUNCT COMPETITOR -- 457: the punctuation effect is neither
leaf-specific (455) nor explained by unigram frequency (456:
within the TOP frequency quartile, punctuation still helps 77.8%
against 54.4% for non-punctuation, and the help-rate ladder across
frequency quartiles is flat-to-noisy at 0.50/0.55/0.48/0.59).
Damage does move predictions toward the prior (KL to unigram
4.957 -> 4.751) but only by 4%, nowhere near enough.
Next hypothesis, and it is about the model rather than the
metric: at punctuation targets the intact model is systematically
OVER-CONFIDENT in a wrong continuation -- it wants to keep the
phrase going -- and damaging almost any machinery lets the
punctuation win. If so the effect is a real deficiency that
ablation partially repairs, and the competitor should be
identifiable.
Method: at punctuation member positions where ablation helps,
record the intact model's top-1 token and how its probability
moves under ablation, against non-punctuation helped positions as
the control class.
REGISTERED PREDICTIONS:
  (a) WRONG CONTINUATION: at punct positions where ablation
      helps, the intact top-1 is a NON-punctuation token in
      >= 60% of cases;
  (b) TARGETED SUPPRESSION: ablation lowers that competitor's
      probability by more than it lowers the probability of a
      random other token by a factor >= 3;
  (c) CONTROL: at non-punct helped positions the intact top-1 is
      not systematically punctuation (< 30%), i.e. the asymmetry
      is specific."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_competitor_results.json'
TAG='r.13.2.1'; COMP='a7'

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    rows=cl.rows()
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    def hooks():
        mu=mus[COMP].to(DEV)
        def fh(mo,i_,o_,mu=mu):
            y,v1=o_
            return (mu.expand_as(y).to(y.dtype),v1)
        return [MODS[COMP].register_forward_hook(fh)]
    lf=cl.leaf(TAG); mem=lf['member'].tolist()
    d=cl.ce_sweep(hooks())-cl.base_ce()
    ispunct=lambda t:(lambda s: bool(s) and
                      not any(c.isalnum() for c in s))(
                          cl.d1(int(t)).strip())
    # group members by row so each forward serves many members
    byrow={}
    for g in mem: byrow.setdefault(g//256,[]).append(g%256)
    rowsel=sorted(byrow)[:40]
    stats={'punct':{'n':0,'top1_nonpunct':0,'dcomp':0.0,
                    'drand':0.0},
           'nonpunct':{'n':0,'top1_punct':0}}
    g0=torch.Generator().manual_seed(3)
    for i in range(0,len(rowsel),4):
        rid=torch.tensor(rowsel[i:i+4])
        bb=rows[rid,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=len(rid)
        def probs(ab):
            hs=hooks() if ab else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            for h in hs: h.remove()
            return F.softmax(lg,dim=-1)
        p0=probs(False); p1=probs(True)
        for b,r in enumerate(rid.tolist()):
            for pos in byrow[r]:
                gi=r*256+pos
                if float(d[gi])>=0: continue      # only helped
                tgt=int(rows[r,pos+1])
                t1=int(p0[b,pos].argmax())
                if ispunct(tgt):
                    s=stats['punct']; s['n']+=1
                    s['top1_nonpunct']+=int(not ispunct(t1))
                    s['dcomp']+=float(p1[b,pos,t1]-p0[b,pos,t1])
                    rt=int(torch.randint(0,50257,(1,),
                                         generator=g0))
                    s['drand']+=float(p1[b,pos,rt]-p0[b,pos,rt])
                else:
                    s=stats['nonpunct']; s['n']+=1
                    s['top1_punct']+=int(ispunct(t1))
        print(f'rows {i} done',flush=True)
    P=stats['punct']; N=stats['nonpunct']
    fr=P['top1_nonpunct']/max(P['n'],1)
    dc=P['dcomp']/max(P['n'],1); dr=P['drand']/max(P['n'],1)
    fn=N['top1_punct']/max(N['n'],1)
    pa=fr>=0.60
    pb=abs(dc)>=3*max(abs(dr),1e-9)
    pc=fn<0.30
    out={'punct_helped_n':P['n'],
         'punct_top1_nonpunct_frac':round(fr,3),
         'mean_dprob_competitor':round(dc,5),
         'mean_dprob_random':round(dr,8),
         'nonpunct_helped_n':N['n'],
         'nonpunct_top1_punct_frac':round(fn,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f"punct helped n={P['n']}: top-1 is non-punct "
          f"{fr:.3f}; competitor dprob {dc:+.5f} vs random "
          f"{dr:+.2e}")
    print(f"non-punct helped n={N['n']}: top-1 is punct {fn:.3f}")
    for nm,v in (('a','intact top-1 is a wrong continuation'),
                 ('b','ablation suppresses that competitor'),
                 ('c','control: no reverse asymmetry')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
