"""HEAD 12.6 TARGETS -- 489: the corrected classification (488)
rules out the obvious identities. At match positions head 12.6
reads the induction target (successor of a repeat) 9.3% of the
time -- 8.5x its 1.1% null, but nowhere near dominant -- the
repeat itself 3.3%, and something LOCAL only 15.5% against 43.5%
for a control head in the same layer. Seventy-one percent of its
top reads are "other": distant, and spread thinly (its top six
offsets, -1 through -9, cover only 22% of reads).
So 12.6 is a DIFFUSE LONG-RANGE reader whose damage is
match-specific -- a head type this program has not catalogued.
Characterise it by WHAT it reads rather than WHERE: at match
positions, take the token at its top read and compare the class
distribution against the corpus base rate.
Classes: punctuation, newline, digit, capitalised, space-word,
subword, plus a rarity split (bottom-quartile unigram frequency).
Control: head 12.3, same layer.
REGISTERED PREDICTIONS:
  (a) SELECTIVE: at least one token class is enriched >= 2x over
      its corpus base rate among 12.6's read targets;
  (b) SALIENCE HYPOTHESIS: rare tokens (bottom-quartile unigram
      frequency) are the most enriched class -- a long-range head
      that ignores position but seeks informative tokens;
  (c) CONTROL: head 12.3 shows a weaker top enrichment than
      12.6."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_12_6_targets_results.json'
NR=16
HEADS=[6,3]

def classes(tok):
    s=cl.d1(int(tok)); st=s.strip()
    return {'punct':bool(st) and not any(c.isalnum() for c in st),
            'newline':chr(10) in s,'digit':st.isdigit(),
            'capitalized':s.startswith(' ') and bool(st)
                          and st[:1].isupper(),
            'space_word':s.startswith(' ') and st.isalpha(),
            'subword':(not s.startswith(' ')) and st.isalpha()}

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    allrows=cl.rows()
    flat=allrows[:,:257].reshape(-1)
    V=m.lm_head.weight.shape[0]
    cnt=torch.bincount(flat,minlength=V).float()[:V]
    uni=cnt/cnt.sum()
    rare_thr=float(torch.quantile(uni[uni>0],0.25))
    KINDS=list(classes(0).keys())+['rare']
    base={k:0 for k in KINDS}; nbase=0
    for r in range(NR):
        for q in range(T):
            t=int(ROWS[r,q]); c=classes(t)
            for k in c:
                if c[k]: base[k]+=1
            if float(uni[t])<=rare_thr: base['rare']+=1
            nbase+=1
    baserate={k:base[k]/max(nbase,1) for k in KINDS}
    print('corpus base rates:',{k:round(v,3)
                                for k,v in baserate.items()},
          flush=True)
    got={h:{k:0 for k in KINDS} for h in HEADS}
    n={h:0 for h in HEADS}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        hh=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hh.remove()
        X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        for hd in HEADS:
            def r3(w):
                return are(F.rms_norm(w(X).view(B,T,9,128),
                           (128,))[:,:,hd][:,:,None],cos,sin)[:,:,0]
            qf,kf=r3(at.c_q),r3(at.c_k)
            q2,k2=r3(at.c_q2),r3(at.c_k2)
            pat=((torch.einsum('bqd,bkd->bqk',qf.float(),
                               kf.float())/128)
                 *(torch.einsum('bqd,bkd->bqk',q2.float(),
                                k2.float())/128)) \
                *torch.tril(torch.ones(T,T,device=DEV))
            for b in range(B):
                toks=ROWS[i+b,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]; prev=last.get(t); last[t]=q
                    if prev is None or prev+1>=q or q<8: continue
                    k=int(pat[b,q,:q].abs().argmax())
                    tk=int(ROWS[i+b,k]); c=classes(tk)
                    for kk in c:
                        if c[kk]: got[hd][kk]+=1
                    if float(uni[tk])<=rare_thr:
                        got[hd]['rare']+=1
                    n[hd]+=1
        print(f'batch {i} done',flush=True)
    out={}
    for hd in HEADS:
        rates={k:got[hd][k]/max(n[hd],1) for k in KINDS}
        enr={k:round(rates[k]/max(baserate[k],1e-6),2)
             for k in KINDS}
        top=max(enr,key=enr.get)
        out[f'12.{hd}']={'n':n[hd],
                         'rates':{k:round(v,3)
                                  for k,v in rates.items()},
                         'enrichment':enr,'top_class':top,
                         'top_enrichment':enr[top]}
        print(f"12.{hd}: enrichment {enr} | top {top} "
              f"{enr[top]}x",flush=True)
    a=out['12.6']
    pa=a['top_enrichment']>=2.0
    pb=(a['top_class']=='rare')
    pc=a['top_enrichment']>out['12.3']['top_enrichment']
    out.update({'base_rates':{k:round(v,3)
                              for k,v in baserate.items()},
                'rare_threshold':rare_thr,
                'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc),'runtime_s':time.time()-t0})
    for nm,v in (('a','some class enriched >=2x'),
                 ('b','rare tokens are the top class'),
                 ('c','12.6 is more selective than the control')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
